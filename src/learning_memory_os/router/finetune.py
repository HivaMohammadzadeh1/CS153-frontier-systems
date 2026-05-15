"""LoRA fine-tuning of a HF model for context-routing.

Reads JSONL trajectories, serializes to input/target pairs, runs SFT with PEFT/LoRA,
saves a LoRA adapter directory.
"""

from dataclasses import dataclass
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType as PeftTaskType

from ..trajectories.schemas import Trajectory
from ..trajectories.serializer import trajectory_to_training_pair


def _detect_device_and_precision() -> tuple[str, dict]:
    """Detect torch device + return precision kwargs for TrainingArguments.

    Returns (device_name, training_args_kwargs).
    - CUDA: use bf16=True (fast, well-supported)
    - MPS:  no precision flags; transformers >=4.41 auto-detects MPS and
            bf16 in TrainingArguments is unreliable there — use fp32 default.
    - CPU:  no precision flags, fp32 default.
    """
    if torch.cuda.is_available():
        return "cuda", {"bf16": True}
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # transformers 4.41+ dropped use_mps_device; MPS is auto-detected.
        # bf16 in TrainingArguments is unreliable on MPS; rely on fp32 default.
        return "mps", {}
    return "cpu", {}


@dataclass
class RouterFineTuneConfig:
    hf_model: str
    lora_r: int
    lora_alpha: int
    batch_size: int
    max_seq_len: int
    use_4bit_base: bool
    epochs: int = 2
    lr: float = 2e-4


def _load_pairs(jsonl_path: Path) -> list[dict]:
    pairs = []
    with jsonl_path.open() as f:
        for line in f:
            t = Trajectory.model_validate_json(line)
            pairs.append(trajectory_to_training_pair(t))
    return pairs


def _format_for_sft(pair: dict, eos_token: str) -> dict:
    return {"text": f"{pair['input']}\n{pair['target']}{eos_token}"}


def _maybe_quant_config():
    """Return a BitsAndBytesConfig if bitsandbytes is available, else None."""
    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes  # noqa: F401  (import check)
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    except ImportError:
        return None


def finetune(
    cfg: RouterFineTuneConfig,
    *,
    trajectories_path: Path,
    output_dir: Path,
    seed: int = 42,
) -> Path:
    device_name, precision_kwargs = _detect_device_and_precision()
    print(f"[finetune] detected device: {device_name}  precision_kwargs: {precision_kwargs}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Choose load dtype: bf16 on CUDA/MPS (MPS supports bf16 at the model level),
    # fp32 on CPU.
    if device_name in ("cuda", "mps"):
        load_dtype = torch.bfloat16
    else:
        load_dtype = torch.float32
    load_kwargs: dict = {"torch_dtype": load_dtype}

    if cfg.use_4bit_base:
        qc = _maybe_quant_config()
        if qc is not None:
            load_kwargs["quantization_config"] = qc
        else:
            print("WARN: 4-bit quant requested but bitsandbytes unavailable; loading bf16 instead.")

    base = AutoModelForCausalLM.from_pretrained(cfg.hf_model, **load_kwargs)

    # Explicitly move to MPS if needed (defensive; Trainer also does this).
    if device_name == "mps":
        base = base.to("mps")

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=PeftTaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(base, lora_cfg)

    pairs = _load_pairs(trajectories_path)
    raw = [_format_for_sft(p, tokenizer.eos_token) for p in pairs]
    ds = Dataset.from_list(raw)

    def tokenize(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=cfg.max_seq_len,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])

    output_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        seed=seed,
        **precision_kwargs,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "adapter")
    return output_dir / "adapter"

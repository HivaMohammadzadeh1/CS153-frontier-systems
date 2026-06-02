"""Load a LoRA adapter on top of its base model and route greedily.

No unit test: this requires real model weights + a GPU, exactly like
router/finetune.py. It is exercised on the cluster via scripts/eval_routers.py.
"""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from ..trajectories.schemas import StudentState, PoolItem, TaskType
from .prompt import format_router_input, parse_router_output


class FineTunedRouter:
    def __init__(self, adapter_dir: Path, base_model: str):
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Pick the best available device. bf16 on CUDA; float32 on MPS/CPU
        # (MPS bf16 is flaky and CPU has no bf16 speedup).
        if torch.cuda.is_available():
            device, dtype = "cuda", torch.bfloat16
        elif torch.backends.mps.is_available():
            device, dtype = "mps", torch.float32
        else:
            device, dtype = "cpu", torch.float32

        # transformers >=5 renamed the load kwarg torch_dtype -> dtype; support both.
        try:
            base = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
        except TypeError:
            base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
        self.model = PeftModel.from_pretrained(base, adapter_dir).to(device)
        self.model.eval()

    @torch.no_grad()
    def route(
        self,
        *,
        student_state: StudentState,
        task_type: TaskType,
        task_text: str,
        budget: int,
        candidate_pool: list[PoolItem],
        max_new_tokens: int = 128,
    ) -> list[str]:
        prompt = format_router_input(
            student_state=student_state,
            task_type=task_type,
            task_text=task_text,
            budget=budget,
            candidate_pool=candidate_pool,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = out_ids[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return parse_router_output(text)

"""Smoke-test the fine-tuned LoRA router adapter.

Loads the adapter onto the base model via PEFT and generates a routing decision.
Since learning_memory_os.router.infer does not yet exist, this script uses
the prompt module directly.
"""

from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Adjust these to match the actual smoke run.
ADAPTER_DIR = Path("data/router_checkpoints/qwen2_5_0_5b/adapter")
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Detect device
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"[infer] device: {device}")

# Load tokenizer + base model + adapter
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model = model.to(device)
model.eval()

# Build prompt using the router prompt module
import sys
sys.path.insert(0, "src")
from learning_memory_os.router.prompt import format_router_input, parse_router_output
from learning_memory_os.trajectories.schemas import StudentState, PoolItem, TaskType

state = StudentState(
    student_id="s",
    mastery={},
    active_misconceptions=[],
    recent_episodic_ids=[],
)
pool = [
    PoolItem(id="aaaa1111", title="KV cache", body_excerpt="K and V tensors", token_estimate=100),
    PoolItem(id="bbbb2222", title="Random", body_excerpt="unrelated content", token_estimate=100),
]
prompt = format_router_input(
    student_state=state,
    task_type=TaskType.EXPLAIN,
    task_text="What is a KV cache?",
    budget=300,
    candidate_pool=pool,
)
print("[infer] prompt (last 200 chars):", prompt[-200:])

inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("Raw generated:", repr(generated))
selected = parse_router_output(generated)
print("Selected IDs:", selected)

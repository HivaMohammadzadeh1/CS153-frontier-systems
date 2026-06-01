#!/usr/bin/env bash
# Bootstrap a DigitalOcean GPU Droplet to serve the fine-tuned 7B router via vLLM.
#
# Run as root on a fresh GPU Droplet created from DO's "AI/ML Ready" image
# (NVIDIA driver + CUDA preinstalled). It (1) merges the LoRA adapter into the
# base model and (2) serves the merged model with vLLM's OpenAI-compatible API
# on :8000, as a systemd service that survives reboots.
#
#   HF_TOKEN=hf_xxx bash bootstrap.sh
#
# Override any of: BASE, ADAPTER, MERGED_DIR, SERVED_NAME.
set -euo pipefail

BASE="${BASE:-Qwen/Qwen2.5-7B-Instruct}"
ADAPTER="${ADAPTER:-hivamoh/lmos-router-qwen2_5_7b}"   # private HF adapter repo
MERGED_DIR="${MERGED_DIR:-/opt/lmos-router-merged}"
SERVED_NAME="${SERVED_NAME:-lmos-router-7b}"
: "${HF_TOKEN:?set HF_TOKEN (read access to the private adapter repo)}"
export HF_TOKEN HF_HUB_ENABLE_HF_TRANSFER=1

echo "[bootstrap] base=$BASE adapter=$ADAPTER -> $MERGED_DIR"
nvidia-smi -L || { echo "ERROR: no GPU/driver — use DO's AI/ML Ready GPU image" >&2; exit 1; }

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3-venv python3-pip git

python3 -m venv /opt/venv
source /opt/venv/bin/activate
pip install --upgrade pip
pip install "vllm>=0.6" "transformers>=4.47" "peft>=0.14" "huggingface_hub[hf_transfer]"

# 1) Merge the LoRA adapter into the base -> standalone model (no PEFT at serve time).
echo "[bootstrap] merging adapter into base…"
python - <<PY
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base, adapter, out = "$BASE", "$ADAPTER", "$MERGED_DIR"
tok = AutoTokenizer.from_pretrained(adapter)
try:
    m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu")
except TypeError:
    m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="cpu")
m = PeftModel.from_pretrained(m, adapter)
m.merge_and_unload().save_pretrained(out, safe_serialization=True)
tok.save_pretrained(out)
print("merged ->", out)
PY

# 2) Serve the merged model with vLLM (OpenAI-compatible) as a systemd service.
#    Optional bearer auth: set VLLM_API_KEY in the environment before running.
API_KEY_LINE=""
[ -n "${VLLM_API_KEY:-}" ] && API_KEY_LINE="Environment=VLLM_API_KEY=${VLLM_API_KEY}"

cat >/etc/systemd/system/lmos-router.service <<EOF
[Unit]
Description=LMOS context router (vLLM OpenAI server)
After=network-online.target
Wants=network-online.target

[Service]
${API_KEY_LINE}
ExecStart=/opt/venv/bin/vllm serve ${MERGED_DIR} \\
  --host 0.0.0.0 --port 8000 \\
  --served-model-name ${SERVED_NAME} \\
  --max-model-len 4096 --gpu-memory-utilization 0.90
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now lmos-router
echo "[bootstrap] done. vLLM is starting (first start loads the model ~1-2 min)."
echo "[bootstrap] test once up:  curl http://localhost:8000/v1/models"

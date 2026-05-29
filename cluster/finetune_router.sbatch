#!/bin/bash
# Fine-tune ONE router size on the H100 cluster via pyxis/enroot.
#
# Parameterized by SIZE_ID (an id from config/router_sizes.yaml). Designed to be
# submitted by cluster/submit_sweep.sh, which overrides --time and --job-name and
# exports SIZE_ID/REPO/TRAJ/EPOCHS per size. Can also be submitted directly:
#
#   sbatch --job-name ft-qwen2_5_0_5b --time 01:00:00 \
#          --export=ALL,SIZE_ID=qwen2_5_0_5b cluster/finetune_router.sbatch
#
# Does NOT modify any project code: it only invokes the existing
# scripts/finetune_router.py CLI inside an NGC PyTorch container.
#
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16          # REQUIRED: container squashfs build is single-threaded
#SBATCH --time=04:00:00             # safe default; submit_sweep.sh overrides per size
#SBATCH --output=%x-%j.out
#SBATCH --job-name=ft-router

set -euo pipefail

SIZE_ID="${SIZE_ID:?set SIZE_ID, e.g. qwen2_5_0_5b}"
REPO="${REPO:-$HOME/CS153-frontier-systems}"
TRAJ="${TRAJ:-$REPO/data/trajectories/val.jsonl}"
EPOCHS="${EPOCHS:-2}"

echo "[finetune_router.sbatch] SIZE_ID=$SIZE_ID REPO=$REPO TRAJ=$TRAJ EPOCHS=$EPOCHS"
test -f "$TRAJ" || { echo "ERROR: trajectories file not found: $TRAJ (run cluster/stage.sh first)" >&2; exit 2; }

# NGC ships a CUDA-matched torch. Install ONLY the extra deps on top of it.
# Do NOT run `uv sync` here — it would reinstall/clobber NGC's torch.
# bitsandbytes is strictly needed only for the 7B 4-bit base; if its install
# fails, finetune.py already warns and falls back to bf16 for the smaller sizes.
srun --gres=gpu:1 --cpus-per-task=16 \
  --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
  --container-mounts="$REPO:$REPO" \
  --container-workdir="$REPO" \
  bash -lc "
    set -euo pipefail
    pip install --no-cache-dir 'peft>=0.19' 'datasets>=4' accelerate bitsandbytes
    export PYTHONPATH=src
    python -m scripts.finetune_router \
      --size '$SIZE_ID' \
      --trajectories '$TRAJ' \
      --epochs '$EPOCHS'
  "

echo "[finetune_router.sbatch] done: adapter at $REPO/data/router_checkpoints/$SIZE_ID/adapter"

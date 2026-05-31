#!/bin/bash
# Evaluate the fine-tuned router adapters on the cluster (GPU inference only).
#
# Runs scripts/eval_routers.py with --no-frontier: it loads every adapter under
# data/router_checkpoints/<size>/adapter and scores it against the held-out split,
# writing data/eval/router_results.adapters.json. The frontier-API baseline is NOT
# run here — it needs network + ANTHROPIC_API_KEY and runs on the login pod instead
# (see cluster/README.md), since compute nodes are not assumed to have egress.
#
#   sbatch --time 02:00:00 cluster/eval_routers.sbatch
#
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --job-name=eval-routers

set -euo pipefail

REPO="${REPO:-$HOME/CS153-frontier-systems}"
TEST="${TEST:-$REPO/data/trajectories/val.jsonl}"
LIMIT="${LIMIT:-500}"
OUT="${OUT:-$REPO/data/eval/router_results.adapters.json}"

echo "[eval_routers.sbatch] REPO=$REPO TEST=$TEST LIMIT=$LIMIT OUT=$OUT"
test -f "$TEST" || { echo "ERROR: test trajectories not found: $TEST" >&2; exit 2; }

srun --gres=gpu:1 --cpus-per-task=16 \
  --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
  --container-mounts="$REPO:$REPO" \
  --container-workdir="$REPO" \
  bash -lc "
    set -euo pipefail
    # Pin to versions compatible with the NGC container's (alpha) torch
    # (same as cluster/finetune_router.sbatch).
    # eval_routers.py imports learning_memory_os.config (pydantic-settings),
    # which the NGC image doesn't ship — add it.
    pip install --no-cache-dir 'transformers==4.47.1' 'peft==0.14.0' 'datasets<4' 'accelerate<2' bitsandbytes pydantic-settings
    export PYTHONPATH=src
    python -m scripts.eval_routers \
      --no-frontier \
      --test '$TEST' \
      --limit '$LIMIT' \
      --out '$OUT'
  "

echo "[eval_routers.sbatch] done: adapter metrics at $OUT"

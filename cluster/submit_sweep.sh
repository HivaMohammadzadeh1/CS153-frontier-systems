#!/usr/bin/env bash
# Submit the router fine-tune sweep on the cluster, dependency-chained so the
# sizes run ONE AT A TIME on a single GPU (CLAUDE.md: fewer/larger jobs, single-node).
#
# Run this ON THE LOGIN POD (it calls sbatch), from the repo root.
#
#   bash cluster/submit_sweep.sh                 # full chained sweep (0.5B -> 1.5B -> 3B -> 7B)
#   bash cluster/submit_sweep.sh qwen2_5_0_5b    # just one size (use for the smoke test)
#   bash cluster/submit_sweep.sh qwen2_5_3b qwen2_5_7b   # a subset, still chained in order given
set -euo pipefail

REPO="${REPO:-$HOME/CS153-frontier-systems}"
TRAJ="${TRAJ:-$REPO/data/trajectories/val.jsonl}"
EPOCHS="${EPOCHS:-2}"
SBATCH_SCRIPT="$REPO/cluster/finetune_router.sbatch"

test -f "$SBATCH_SCRIPT" || { echo "ERROR: $SBATCH_SCRIPT not found (run cluster/stage.sh?)" >&2; exit 2; }
test -f "$TRAJ" || { echo "ERROR: trajectories not found: $TRAJ (run cluster/stage.sh first)" >&2; exit 2; }

# Per-size walltime requests (with headroom over the H100 estimates in the spec).
declare -A WALLTIME=(
  [qwen2_5_0_5b]=01:00:00
  [qwen2_5_1_5b]=01:00:00
  [qwen2_5_3b]=02:00:00
  [qwen2_5_7b]=04:00:00
)
DEFAULT_ORDER=(qwen2_5_0_5b qwen2_5_1_5b qwen2_5_3b qwen2_5_7b)

SELECTED=("$@")
if [[ ${#SELECTED[@]} -eq 0 ]]; then
  SELECTED=("${DEFAULT_ORDER[@]}")
fi

dep=""
for size in "${SELECTED[@]}"; do
  wt="${WALLTIME[$size]:-04:00:00}"
  # NOTE: do NOT use sbatch --export on this cluster — any explicit --export
  # makes SLURM try to retrieve the user's login env, which fails here
  # (user_env_retrieval_failed → job held). Instead export the vars into the
  # submitting shell and let the default (ALL) propagation carry them.
  args=(
    --job-name "ft-$size"
    --time "$wt"
  )
  if [[ -n "$dep" ]]; then
    args+=(--dependency="afterany:$dep")
  fi
  jid=$(SIZE_ID="$size" REPO="$REPO" TRAJ="$TRAJ" EPOCHS="$EPOCHS" \
        sbatch --parsable "${args[@]}" "$SBATCH_SCRIPT")
  echo "submitted $size (walltime $wt) as job $jid${dep:+ — runs after $dep}"
  dep="$jid"
done

echo
echo "Monitor with:  squeue -u \$USER   |   sacct -u \$USER -S today"
echo "Adapters will land in: $REPO/data/router_checkpoints/<size>/adapter"

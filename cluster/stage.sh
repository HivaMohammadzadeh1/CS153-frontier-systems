#!/usr/bin/env bash
# Stage code + trajectories onto the cluster login pod (shared Weka home).
# Run this FROM THE DEV MACHINE (it uses kubectl). No SSH, per CLAUDE.md.
#
#   export CLUSTER_USER=<your cluster username>
#   bash cluster/stage.sh
#
# Overridable via env: BRANCH, REPO_URL, REMOTE_REPO, LOCAL_TRAJ.
set -euo pipefail

: "${CLUSTER_USER:?export CLUSTER_USER=<your cluster username> first}"
BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
REPO_URL="${REPO_URL:-$(git config --get remote.origin.url)}"
REMOTE_REPO="${REMOTE_REPO:-/home/$CLUSTER_USER/CS153-frontier-systems}"
LOCAL_TRAJ="${LOCAL_TRAJ:-data/trajectories/val.jsonl}"

test -n "$REPO_URL" || { echo "ERROR: could not determine REPO_URL; set it explicitly" >&2; exit 2; }
test -f "$LOCAL_TRAJ" || { echo "ERROR: local trajectories not found: $LOCAL_TRAJ" >&2; exit 2; }

POD=$(kubectl get pod -n slurm -l "stanford/user=$CLUSTER_USER" -o jsonpath='{.items[0].metadata.name}')
test -n "$POD" || { echo "ERROR: no login pod found for stanford/user=$CLUSTER_USER" >&2; exit 2; }
echo "login pod: $POD"
echo "branch:    $BRANCH"
echo "repo url:  $REPO_URL"

# Clone or fast-forward the repo on the pod, as the cluster user so ownership is correct.
kubectl exec -n slurm "$POD" -c login -- runuser -u "$CLUSTER_USER" -- bash -lc "
  set -euo pipefail
  if [ -d '$REMOTE_REPO/.git' ]; then
    cd '$REMOTE_REPO'
    git fetch --all --prune
    git checkout '$BRANCH'
    git pull --ff-only
  else
    git clone --branch '$BRANCH' '$REPO_URL' '$REMOTE_REPO'
  fi
  mkdir -p '$REMOTE_REPO/data/trajectories'
"

# val.jsonl is gitignored, so copy it in separately.
kubectl cp "$LOCAL_TRAJ" "slurm/$POD:$REMOTE_REPO/data/trajectories/val.jsonl" -c login
echo "staged $LOCAL_TRAJ -> $REMOTE_REPO/data/trajectories/val.jsonl"
echo
echo "Next: kubectl exec -it -n slurm $POD -c login -- runuser -u $CLUSTER_USER -- bash -l"
echo "      then from $REMOTE_REPO run the smoke test, then the sweep (see cluster/README.md)."

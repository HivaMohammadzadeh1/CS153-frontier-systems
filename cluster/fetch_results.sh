#!/usr/bin/env bash
# Pull trained LoRA adapters back from the cluster to the dev machine.
# Run this FROM THE DEV MACHINE (it uses kubectl).
#
#   export CLUSTER_USER=<your cluster username>
#   bash cluster/fetch_results.sh
#
# Overridable via env: REMOTE_REPO, DEST_PARENT.
set -euo pipefail

: "${CLUSTER_USER:?export CLUSTER_USER=<your cluster username> first}"
REMOTE_REPO="${REMOTE_REPO:-/home/$CLUSTER_USER/CS153-frontier-systems}"
DEST_PARENT="${DEST_PARENT:-data}"

POD=$(kubectl get pod -n slurm -l "stanford/user=$CLUSTER_USER" -o jsonpath='{.items[0].metadata.name}')
test -n "$POD" || { echo "ERROR: no login pod found for stanford/user=$CLUSTER_USER" >&2; exit 2; }
echo "login pod: $POD"

mkdir -p "$DEST_PARENT"
# Copies the whole router_checkpoints tree (LoRA adapters are small, tens of MB each).
kubectl cp "slurm/$POD:$REMOTE_REPO/data/router_checkpoints" "$DEST_PARENT/router_checkpoints" -c login
echo "fetched adapters -> $DEST_PARENT/router_checkpoints"
echo
echo "Verify:"
echo "  ls $DEST_PARENT/router_checkpoints/*/adapter/adapter_config.json"

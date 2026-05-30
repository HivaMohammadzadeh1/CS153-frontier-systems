# Cluster execution layer — router fine-tuning sweep

Runbook for fine-tuning the four context-router sizes on the shared H100 SLURM
cluster. This layer is **purely additive**: it invokes the existing
`scripts/finetune_router.py` CLI inside an NGC container. No project code changes.

See `CLAUDE.md` for the full cluster reference (access model, partitions, budget).
Design rationale: `docs/superpowers/specs/2026-05-29-slurm-finetuning-execution-design.md`.

## What's here

| File | Runs on | Purpose |
|------|---------|---------|
| `stage.sh` | dev machine | git clone/pull the repo on the pod + `kubectl cp` the gitignored `val.jsonl` |
| `finetune_router.sbatch` | cluster (sbatch) | one GPU job, fine-tunes one `SIZE_ID` in an NGC container |
| `submit_sweep.sh` | login pod | submit all four sizes, dependency-chained on a single GPU |
| `eval_routers.sbatch` | cluster (sbatch) | GPU job, scores every trained adapter on the held-out split |
| `fetch_results.sh` | dev machine | `kubectl cp` the trained adapters back |

Scope: **training + evaluation.** Training produces the adapters; evaluation scores
them and plots the accuracy-vs-cost frontier. The eval splits across two machines: the
GPU adapter inference runs as a cluster job (`eval_routers.sbatch`, `--no-frontier`),
while the frontier-API baseline (`--no-adapters`) runs on the login pod because it
needs network + `ANTHROPIC_API_KEY` and compute nodes are not assumed to have egress.

Trains on the existing `data/trajectories/val.jsonl` (5,000 trajectories). To train on
a larger regenerated set later, pass `TRAJ=/path/to/other.jsonl` — no script changes.

## Prerequisites

- `kubectl` configured against your Omniva kubeconfig on the dev machine.
- Your cluster username:

  ```bash
  export CLUSTER_USER=<your username>   # replaces the MY_USERNAME placeholder in CLAUDE.md
  ```

## Procedure

### 1. Sanity-check pyxis/enroot (once per cluster session)

```bash
POD=$(kubectl get pod -n slurm -l stanford/user=$CLUSTER_USER -o jsonpath='{.items[0].metadata.name}')
kubectl exec -it -n slurm "$POD" -c login -- runuser -u "$CLUSTER_USER" -- \
  srun --partition=small --gres=gpu:1 --cpus-per-task=16 \
    --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
    python -c "import torch; print(torch.cuda.device_count())"
```

Should print `1`. First run builds the squashfs (~3 min with 16 CPUs).

### 2. Stage code + data (from the dev machine)

```bash
bash cluster/stage.sh
```

Clones/fast-forwards the repo into `/home/$CLUSTER_USER/CS153-frontier-systems` on the
pod (Weka, visible to workers) and copies `val.jsonl` into place.

### 3. Check GPU-hour budget (on the login pod)

```bash
sshare -u $USER
sacctmgr show qos qos-$USER format=GrpTRESMins
```

The full sweep costs ≈ 3 GPU-hours. Don't submit if the remaining budget is tight.

### 4. Smoke test — the smallest model only (first real verification)

From the repo root on the login pod:

```bash
cd /home/$CLUSTER_USER/CS153-frontier-systems
bash cluster/submit_sweep.sh qwen2_5_0_5b
squeue -u $USER
```

Wait for it to finish, then confirm it produced an adapter:

```bash
ls data/router_checkpoints/qwen2_5_0_5b/adapter/adapter_config.json
```

If this fails, fix before chaining the rest — don't burn hours on a broken pipeline.

### 5. Full sweep

```bash
bash cluster/submit_sweep.sh
```

Submits 0.5B → 1.5B → 3B → 7B, each waiting for the previous (`--dependency=afterany`),
so one GPU is used at a time. Monitor:

```bash
squeue -u $USER
sacct -u $USER -S today
tail -f ft-qwen2_5_3b-*.out
```

If the 7B OOMs even with its 4-bit base, cut it — three sizes are still a valid result.

### 6. Fetch adapters back (from the dev machine)

```bash
bash cluster/fetch_results.sh
ls data/router_checkpoints/*/adapter/adapter_config.json
```

Each line is one successful fine-tune.

### 7. Evaluate the adapters + plot the frontier

Two parts, because the GPU compute nodes aren't assumed to reach the Anthropic API.

**(a) Adapter inference — GPU job (login pod):**

```bash
cd /home/$CLUSTER_USER/CS153-frontier-systems
sbatch cluster/eval_routers.sbatch        # writes data/eval/router_results.adapters.json
squeue -u $USER
```

**(b) Frontier-API baseline — login pod (no GPU, needs the key):**

```bash
export ANTHROPIC_API_KEY=<your key>
PYTHONPATH=src python -m scripts.eval_routers \
  --no-adapters --limit 500 \
  --out data/eval/router_results.frontier.json
```

**(c) Merge the two result sets and plot:**

```bash
PYTHONPATH=src python - <<'PY'
import json, pathlib
rows = []
for f in ["data/eval/router_results.adapters.json", "data/eval/router_results.frontier.json"]:
    p = pathlib.Path(f)
    if p.exists():
        rows += json.loads(p.read_text())
pathlib.Path("data/eval/router_results.json").write_text(json.dumps(rows, indent=2))
print(f"merged {len(rows)} rows")
PY
PYTHONPATH=src python -m scripts.plot_pareto   # writes data/eval/pareto.png
```

Then `bash cluster/fetch_results.sh` already brings back `data/router_checkpoints`;
to also pull the plot: `kubectl cp slurm/$POD:/home/$CLUSTER_USER/CS153-frontier-systems/data/eval data/eval -c login`.

## Notes

- **Container deps:** the job `pip install`s only `peft`, `datasets`, `accelerate`,
  `bitsandbytes` on top of NGC's CUDA-matched torch. It does **not** run `uv sync`
  (which would clobber that torch). `bitsandbytes` matters only for the 7B 4-bit base.
- **Single GPU, single node** throughout — matches the partitions' `MaxNodes=1`.
- **Username** is never hardcoded; every script reads `$CLUSTER_USER` / `$USER`.

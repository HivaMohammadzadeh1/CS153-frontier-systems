# SLURM Execution Layer for Router Fine-Tuning — Design Spec

**Date:** 2026-05-29
**Status:** Draft for review
**Related:** `docs/superpowers/plans/2026-05-13-plan-3-router-finetuning.md`, `CLAUDE.md` (cluster section)

## 1. Problem

The router fine-tuning pipeline already exists and is committed: `src/learning_memory_os/router/finetune.py` (LoRA SFT with CUDA/MPS/CPU auto-detect), `scripts/finetune_router.py` (CLI), `config/router_sizes.yaml` (four sizes: 0.5B / 1.5B / 3B / 7B), and 5,000 generated trajectories in `data/trajectories/val.jsonl`. The code runs today on the dev Mac via MPS, slowly, and only the 0.5B adapter has been produced (a smoke test).

There is **no path to run the real four-size sweep on the H100 cluster.** `CLAUDE.md` specifies the actual compute target — a shared 32×H100 SLURM cluster reached by `kubectl exec` into a login pod, where all GPU work goes through `sbatch`, never the login pod itself — but the repo contains zero `sbatch` scripts and no cluster runbook.

This spec adds that missing execution layer.

## 2. Goal & Non-Goals

**Goal:** A purely additive set of cluster scripts + a runbook that take the *unchanged* Python pipeline and run the four-size LoRA sweep on the H100 cluster, producing four LoRA adapters retrievable back to the dev machine.

**Non-goals (explicitly out of scope):**
- **No changes to `finetune.py` or `finetune_router.py`.** They already auto-detect CUDA and accept `--size` / `--trajectories` / `--out` / `--epochs`. The cluster layer only *invokes* them.
- **No trajectory regeneration.** Train on the existing 5,000-trajectory `val.jsonl`. (The scripts stay parameterized on `--trajectories` so a larger regenerated set can be dropped in later with no script changes — the "if needed" path.)
- **No corpus expansion** (CS336 Spring 2026 refresh, systems-book content). Deferred to a separate spec; the user will provide licensed book text later.
- **No cluster-side evaluation / Pareto plot.** The eval half of Plan 3 (`router/infer.py`, `router/frontier_api.py`, `eval/router_eval.py`, `eval/pareto.py`, `scripts/eval_routers.py`, `scripts/plot_pareto.py`) **does not exist yet**. A cluster eval job has nothing to call. Eval-on-cluster is a documented follow-up (§8), blocked on implementing those modules first.

## 3. Constraints (from CLAUDE.md)

- Access is `kubectl exec` into a login pod (`slurm-login-<user>-<hash>`). **No SSH, no port-forward to external services.** File movement is `kubectl cp` or network pull (git/HTTPS/S3/HF Hub).
- **Never run GPU work on the login pod.** All training goes through `sbatch` / `srun` to a partition.
- **Single-node only** (`MaxNodes=1`). Default `--gres=gpu:1`; multi-GPU multiplies GPU-hour consumption.
- **Containers via pyxis/enroot.** Always pass `--cpus-per-task=16` on container jobs (first-use squashfs build is single-threaded; 16 CPUs ≈ 3 min vs. ~30 min). Use the registry-URI form for NGC images: `nvcr.io#nvidia/pytorch:24.12-py3`.
- **GPU-hour budget** is QoS-capped. Prefer fewer, larger jobs; check `sshare -u $USER` before a sweep.
- Default partition `small` (24h max walltime).
- Home (`/home/<user>`) is shared Weka, mounted on login pod and all workers — code/data staged there is visible to jobs.

## 4. Architecture

All new files live under a new top-level `cluster/` directory. Nothing else in the repo changes.

```
cluster/
  finetune_router.sbatch   # one parameterized GPU job: srun NGC container, install extra deps, run finetune CLI for one SIZE_ID
  submit_sweep.sh          # submit the four sizes as a dependency-chained sequence on a single GPU
  stage.sh                 # on the login pod: git clone/pull the branch + kubectl-cp the gitignored val.jsonl into place
  fetch_results.sh         # kubectl-cp the trained adapters back to the dev machine
  README.md                # the end-to-end kubectl runbook (the human-facing operating guide)
```

### 4.1 Orchestration: parameterized job + dependency-chained submit (Approach A)

A single `finetune_router.sbatch` is parameterized by a `SIZE_ID` (passed via `--export` or positional arg) so one script trains any size. `submit_sweep.sh` submits the four sizes in order with `--dependency=afterany:<prev_jobid>` so they run **one at a time on a single GPU**, honoring CLAUDE.md's "fewer, larger jobs / single-GPU" guidance while keeping each size independently re-runnable.

Rationale vs. alternatives:
- **Rejected — SLURM array job (`--array=0-3`):** forces one walltime/memory spec sized for the worst case (7B + 4-bit), so the three small models over-reserve and per-size tweaks are awkward.
- **Rejected — one monolithic job trains all four sequentially:** fewest prolog/epilog cycles, but a single timeout/OOM risks the whole sweep and holds the GPU reservation contiguously.
- **Chosen — A:** the 7B is the size most likely to need a second attempt; re-running it must not require redoing the cheap three.

### 4.2 `finetune_router.sbatch` — anatomy

```bash
#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16          # REQUIRED for container squashfs build speed
#SBATCH --time=<per-size, see §6>
#SBATCH --output=%x-%j.out
#SBATCH --job-name=ft-router

set -euo pipefail
SIZE_ID="${SIZE_ID:?set SIZE_ID, e.g. qwen2_5_0_5b}"
REPO="${REPO:-$HOME/CS153-frontier-systems}"
TRAJ="${TRAJ:-$REPO/data/trajectories/val.jsonl}"
EPOCHS="${EPOCHS:-2}"

srun --gres=gpu:1 --cpus-per-task=16 \
  --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
  --container-mounts="$REPO:$REPO" \
  --container-workdir="$REPO" \
  bash -lc '
    set -euo pipefail
    # NGC ships a CUDA-matched torch. Install ONLY the extra deps on top of it.
    # Do NOT run `uv sync` — it would reinstall/clobber NGC torch.
    pip install --no-cache-dir "peft>=0.19" "datasets>=4" accelerate bitsandbytes
    export PYTHONPATH=src
    python -m scripts.finetune_router --size "'"$SIZE_ID"'" \
      --trajectories "'"$TRAJ"'" --epochs "'"$EPOCHS"'"
  '
```

Key decisions:
- **Container = NGC PyTorch**, torch comes from the image. Extra deps (`peft`, `datasets`, `accelerate`, `bitsandbytes`) installed at job start. `bitsandbytes` is only strictly needed for 7B 4-bit; if it fails to install it is non-fatal for the three smaller sizes, and `finetune.py` already warns and falls back to bf16 when bitsandbytes is unavailable.
- **`PYTHONPATH=src`** to import the project (src layout) without a full editable install.
- **Bind-mount the repo** so the adapter written to `data/router_checkpoints/<size>/adapter` lands on Weka and survives the job.

### 4.3 Username handling

No script hardcodes the `MY_USERNAME` placeholder. Scripts that run on the login pod or invoke `kubectl` read `${CLUSTER_USER:?set CLUSTER_USER}`; the runbook tells the operator to `export CLUSTER_USER=<their_username>` once. Pod selection uses the documented label: `kubectl get pod -n slurm -l stanford/user=$CLUSTER_USER`.

## 5. Data & code flow

1. **Code → pod:** `stage.sh` runs `git clone`/`git pull` of the working branch into `$HOME/CS153-frontier-systems` on the pod (Weka, shared to workers).
2. **Trajectories → pod:** `data/trajectories/val.jsonl` is gitignored, so `stage.sh` `kubectl cp`s it from the dev machine into `$REPO/data/trajectories/val.jsonl`.
3. **Train:** `submit_sweep.sh` submits the four jobs; each writes `data/router_checkpoints/<size>/adapter/` on Weka.
4. **Adapters → dev machine:** `fetch_results.sh` `kubectl cp`s `data/router_checkpoints/` back. Adapters are small (LoRA only, tens of MB).

## 6. Sweep sizing & budget

Estimated single-H100 wall time over 5K trajectories, with walltime requests carrying headroom:

| Size | Est. wall time | `--time` request | Notes |
|------|----------------|------------------|-------|
| 0.5B | ~10 min | 01:00:00 | smoke-test target |
| 1.5B | ~20 min | 01:00:00 | |
| 3B   | ~40 min | 02:00:00 | |
| 7B   | ~90 min | 04:00:00 | 4-bit base; OOM-prone |

Total ≈ **3 GPU-hours**. The runbook checks remaining budget with `sshare -u $USER` and `sacctmgr show qos qos-$USER format=GrpTRESMins` **before** submitting. If 7B OOMs even at 4-bit, it is cut from the sweep — three sizes still produce a valid downstream result (consistent with Plan 3's contingency).

## 7. Operating procedure (runbook outline)

`cluster/README.md` documents, in order:
1. `export CLUSTER_USER=<username>`; resolve the login pod name.
2. Sanity-check pyxis/enroot end-to-end (the `torch.cuda.device_count()` one-liner from CLAUDE.md).
3. `bash cluster/stage.sh` — clone/pull repo + cp `val.jsonl`.
4. **Smoke first:** submit *only* the 0.5B job; confirm it completes and writes an adapter. This is the first real on-hardware verification.
5. Check budget; run `bash cluster/submit_sweep.sh` for the full chained sweep.
6. Monitor: `squeue -u $CLUSTER_USER`, `sacct -u $CLUSTER_USER -S today`, tail `ft-router-*.out`.
7. `bash cluster/fetch_results.sh` — pull adapters back.
8. Verify: each `data/router_checkpoints/<size>/adapter/adapter_config.json` exists.

## 8. Deferred follow-ups

- **Cluster-side eval + Pareto plot.** Blocked on implementing Plan 3 Tasks 9–10 (`router/infer.py`, `router/frontier_api.py`, `eval/router_eval.py`, `eval/pareto.py`, `scripts/eval_routers.py`, `scripts/plot_pareto.py`). Once they exist, add `cluster/eval_routers.sbatch` (GPU job for adapter inference) and run the frontier-API baseline + Pareto plot on the login pod (CPU + network + `ANTHROPIC_API_KEY`, no GPU).
- **Corpus expansion** (CS336 Spring 2026, systems books with user-provided licensed text) — separate spec.
- **Train on a larger regenerated trajectory set** — drop a new JSONL in and re-run; scripts are already parameterized.

## 9. Verification plan

Honest about limits: the dev machine has no configured cluster access (the `MY_USERNAME` placeholder is unedited), so the author cannot submit to the real cluster. Verifiable locally before the user runs anything:
- `bash -n` syntax-check every `cluster/*.sbatch` and `cluster/*.sh`.
- Confirm the `python -m scripts.finetune_router` invocation matches the actual CLI signature in `scripts/finetune_router.py` (flags: `--size`, `--trajectories`, `--out`, `--epochs`).
- Confirm container image reference, `--cpus-per-task=16`, and single-GPU usage match CLAUDE.md rules.

The **first real verification is operator-run**: the 0.5B smoke job (step 4 of the runbook) before the full sweep is chained.

## 10. Success criteria

- The four `cluster/` scripts + README exist, are syntax-clean, and reference the existing CLI/config unchanged.
- Following the runbook on the cluster, the 0.5B smoke job produces an adapter.
- The full sweep produces adapters for every size that fits hardware (≥3 of 4), retrievable to the dev machine.
- No edits to `finetune.py`, `finetune_router.py`, `config/router_sizes.yaml`, or any trajectory data.

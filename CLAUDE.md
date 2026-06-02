# CLAUDE.md

Instructions for Claude Code or Claude-backed agents working in this repository.

## Start Here

This project is Learning Memory OS, a CS153 final project due on 2026-05-29. The immediate objective is the Week 6 MVP described in:

- `docs/superpowers/plans/2026-05-13-plan-1-mvp-system.md`

The deeper research framing is in:

- `docs/superpowers/specs/2026-05-12-learning-memory-os-design.md`

Before editing, read the relevant section of the plan and continue from the earliest unchecked task unless the user asks for something more specific.

## Operating Mode

Act as an implementation partner, not a brainstorming assistant. When the user asks to move the project forward, choose the next concrete task from the plan, implement it, run the smallest meaningful verification, and report exactly what changed.

Use this default loop:

1. Identify the next blocked milestone.
2. Make the smallest coherent code/doc change.
3. Run the targeted test or smoke check.
4. Update docs/checklists when the implementation state changes.
5. Leave clear next steps.

## Non-Negotiables

- Do not invent a new architecture without reconciling it with the spec and plan.
- Do not require live Anthropic/OpenAI calls in unit tests.
- Do not commit `.env`, API keys, logs, generated trajectories, local database files, or notebooks with private data.
- Do not spend time on UI polish before the ingestion -> memory -> selector -> tutor -> logging path works.
- Do not blur claims in the writeup: synthetic training is oracle distillation; student-zero is n=1.

## Preferred Commands

```bash
uv sync
uv run pytest
uv run pytest tests/unit -v
docker compose up -d db
uv run python -m scripts.migrate        # apply all migrations/*.sql (idempotent)
docker compose exec db psql -U lmos -d learning_memory_os -c "\\dt"
```

If `uv` dependencies are not initialized yet, follow Task 1 in the MVP plan first.

## Claude-Specific Guidance

- Keep responses terse and action-oriented.
- When generating code, include tests or explain why the change is doc-only.
- When touching LLM prompts, keep expected JSON contracts explicit and validated by Pydantic.
- When implementing agent behavior, make context selection observable: return or log selected item IDs, scores, budget usage, and omitted high-ranking candidates.
- When unsure between research ambition and MVP reliability, choose MVP reliability and document the deferred ambition.

## High-Value Next Work

Until Plan 1 is done, the most useful tasks are:

- project scaffold with `uv`, Docker Postgres, `.env.example`, and `.gitignore`
- database schema migration for four-tier memory
- typed schemas for artifacts and memory items
- Anthropic and OpenAI wrappers with mocked tests
- semantic/student/episodic/intervention store methods
- heuristic scoring and budgeted packing tests
- tutor REPL that logs every interaction to JSONL



Drop this file at the root of your project repo so Claude Code (or any AI coding assistant) understands the cluster environment you're working on. Edit the `MY_USERNAME` line before checking in.

---

## Environment

- **Cluster**: shared 32× H100 SLURM cluster, 4 nodes × 8 GPUs (H100 80GB HBM3).
- **My username on the cluster**: `MY_USERNAME` ← edit this
- **My home directory**: `/home/MY_USERNAME` (on a shared Weka filesystem mounted on the login pod and all workers).
- **Access**: via Omniva-issued kubeconfig. Connection model is `kubectl exec` into my LoginSet pod; there is no SSH and no port-forward to external services.
- **My login pod selector**: `kubectl get pod -n slurm -l stanford/user=MY_USERNAME`
- **Pod name pattern**: `slurm-login-MY_USERNAME-<hash>`

## How to do anything compute-heavy

NEVER run training, evaluation, container pulls, or other GPU-bound work inside the login pod. The login pod is a small CPU shell with no GPU. All compute goes through `sbatch` to a SLURM partition. Single-node only.

### Submitting a job

```bash
sbatch myjob.sbatch
squeue -u $USER        # see my queue
sacct -u $USER -S today
scancel <jobid>
```

### sbatch template

```bash
#!/bin/bash
#SBATCH --partition=small        # see "Partitions" below
#SBATCH --gres=gpu:1             # request N GPUs (1–8)
#SBATCH --time=00:30:00          # max walltime; required
#SBATCH --output=%x-%j.out
#SBATCH --job-name=my-run

# ... my commands here ...
```

### With a container (pyxis + enroot)

```bash
srun --gres=gpu:1 --cpus-per-task=16 \
  --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
  python train.py
```

**Two rules that will bite you if you skip them:**

- **Always pass `--cpus-per-task=16` on container jobs.** The first time an image is used on a worker, enroot builds a squashfs of it (~20 GB for PyTorch). That build is single-threaded by default and uses only the CPUs SLURM gave the job — with the default 2 CPUs, it takes ~30 minutes. With 16 CPUs, ~3 minutes. After the first build the squashfs is cached on that worker for free.
- **Image reference syntax rule.** For Docker Hub images use the **bare name** (`alpine:latest`, `python:3.12-slim`). For any other registry use the `<registry>#<path>` URI form (`nvcr.io#nvidia/pytorch:24.12-py3`). The specific combination `docker.io#library/<name>` breaks enroot's manifest pipeline (JSON parse error). NGC images are still preferred for ML work — they ship CUDA + NCCL + cuDNN matched to host drivers and avoid Docker Hub rate limits.

Common working images:
- `nvcr.io#nvidia/pytorch:24.12-py3` — PyTorch 2.x + CUDA + NCCL + most ML libs
- `nvcr.io#nvidia/cuda:12.6.0-base-ubuntu22.04` — minimal CUDA base
- `nvcr.io#nvidia/cuda:12.6.0-devel-ubuntu22.04` — has nvcc + headers for compiling

### Sanity-check the cluster

If something feels broken, run this from the login pod to verify pyxis + container imports work end-to-end:

```bash
srun --partition=small --gres=gpu:1 --cpus-per-task=16 \
  --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
  python -c "import torch; print(torch.cuda.device_count())"
```

Should print `1`. If it hangs at `pyxis: importing` for more than ~5 min on the first run, ping the admin — the worker's enroot scratch directory may have lost its permissions.

## Partitions

| Partition | Max walltime | Use for |
|---|---|---|
| `small`  | 24h | single-GPU jobs, quick experiments |
| `medium` | 5d  | multi-GPU runs, longer training |
| `big`    | 5d  | large reserved slots — restricted access |

Default partition is `small`. If a longer run is needed, use `medium`.

## GPU-hour budget

I have a per-user GPU-hour cap enforced by SLURM QoS. When the cap is hit, new jobs queue indefinitely until reset.

```bash
sshare -u $USER                                    # remaining budget
sacctmgr show qos qos-$USER format=GrpTRESMins     # absolute cap (in minutes)
```

When suggesting workloads:
- Prefer fewer, larger jobs over many small ones (fewer prolog/epilog cycles).
- Check feasibility against the remaining budget before suggesting a sweep.
- Default to `--gres=gpu:1` unless multi-GPU is required; multi-GPU multiplies hour consumption.

## Storage

- `/home/MY_USERNAME` — my primary workspace. Persistent across pods + nodes.
- `/home/_shared/models` — read-only shared HuggingFace cache. Use it if a model is already there before downloading.
- `/home/_shared/datasets` — read-only shared dataset cache.
- Soft cap is ~1 TB per user. Avoid hoarding checkpoints; delete or move old runs.

## Login-pod context

- Container runs as `root` by default. To act as me (so file ownership in `/home/MY_USERNAME` is correct), run interactive shells via `runuser`:
  ```bash
  kubectl exec -it -n slurm <my-pod> -c login -- runuser -u MY_USERNAME -- bash -l
  ```
- I have these pre-installed on the login pod: `python3`, `pip`, `uv`, `git`, `git-lfs`, `gh`, `rsync`, `aws` (CLI v2), `huggingface_hub[cli]`, `wandb`, `transformers`, `tokenizers`, `datasets`, `vim`, `tmux`, `htop`, `jq`.
- For heavy Python deps (torch, vllm, deepspeed, etc.), put them in the container image used by `srun --container-image=`, not on the login pod.

## Copying files

```bash
# from my laptop into my pod's home dir
kubectl cp local-file.py slurm/<my-pod>:/home/MY_USERNAME/local-file.py -c login

# from pod back to laptop
kubectl cp slurm/<my-pod>:/home/MY_USERNAME/results.tar.gz . -c login
```

## Constraints to remember when suggesting code

- **No SSH-into-the-cluster** patterns. No `ssh slurm-controller`, no rsync-over-ssh. All file movement uses `kubectl cp` or pulls/pushes via the network (HTTPS, S3, HF Hub).
- **No port-forwarding to external services.** I can't expose a Jupyter or TensorBoard port to the public internet. For interactive notebooks, run Jupyter inside an `srun --pty` session and port-forward via `kubectl port-forward <my-pod>` to my laptop only.
- **Single-node only.** The Stanford partitions (`small`, `medium`, `big`) all have `MaxNodes=1`. Don't suggest multi-node DDP. Use 1–8 GPUs on one node with `torchrun --standalone --nproc-per-node=N`.
- **No `sudo` on the login pod** (well, I'm root, but anything I install isn't persistent across pod restarts — bake into container images or use `pip install --user`).
- **All my actions on the cluster are audit-logged in Teleport.** Don't suggest probing other namespaces or other students' pods.

## Things that will likely come up

- **NCCL / multi-GPU**: the workers have RoCE v2 over 8× ConnectX-7 NDR (400 GbE). NCCL env is pre-injected by Kyverno policy on GPU jobs. Don't override `NCCL_IB_*` or `NCCL_SOCKET_IFNAME` unless I know what I'm doing.
- **Container caching**: enroot caches imported containers under `/run/enroot/${UID}` (per-job ephemeral). The first pull of a large image takes minutes; subsequent jobs reuse the cache for the duration of the worker pod's life.
- **Out-of-memory / OOM**: SLURM allocates the full node's memory by default when you take a GPU. If a job crashes with OOM, it's usually CUDA OOM (model too big for GPU memory), not host OOM.

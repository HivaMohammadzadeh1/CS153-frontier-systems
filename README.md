# Memex — a context-routed mastery tutor for ML systems engineering

> *Not a chatbot that remembers you — a learning OS that **models what you know**, **routes only the context you need** (with a model we fine-tuned ourselves), and **proves your interview-readiness improving over time**.*

CS 153 (Building Frontier-Model Applications) final project. Memex is an adaptive
tutor for ML-systems engineering (the CS336 / CS349D body of knowledge). It pairs a
frontier LLM for explanations with a **persistent, structured learner model** and a
**learned context-selection router**, then packages the result as a monetizable
interview-readiness product.

---

## Why this isn't just a Claude Project / Custom GPT

A generic "chat with memory" (Claude Projects, ChatGPT memory) answers questions and
keeps loose notes. Memex differs structurally:

| | Claude Project / ChatGPT | **Memex** |
|---|---|---|
| Learner model | opaque, uncontrollable notes | **explicit per-concept mastery + confidence, misconceptions, prerequisites, spaced-repetition schedule** |
| Context selection | stuff the window (opaque, costly) | **a fine-tuned router** selects the optimal subset under a token budget — *observable, scored, cheap* |
| Outcomes | none | **interview-readiness % over time** — proof you improved (a stateless tutor can't produce this) |
| Pedagogy | reactive Q&A | misconception **diagnostics**, **adaptive-difficulty** quizzes, spaced-repetition resurfacing |
| Ownership | vendor silo | **self-hostable**; your data; per-user trajectory capture for further fine-tuning |

The thing we sell is the **differentiator**: the outcomes-proven readiness report — not the answer text (which is a commodity).

---

## How it works (the three lenses)

### [1] Research — a learned context router (oracle distillation)
The core research question: *can a small model match a frontier oracle at **context selection** — picking which memory items to put in a tutor's context under a token budget?*
- **Data:** 5,000 routing trajectories (`data/trajectories/val.jsonl`) — `(student_state, task, candidate_pool, budget) → oracle_selection`.
- **Training:** LoRA SFT of four Qwen2.5 sizes (0.5B / 1.5B / 3B / 7B) on a shared **32×H100 SLURM cluster** (`cluster/`, `scripts/finetune_router.py`, `src/learning_memory_os/router/finetune.py`).
- **Result:** held-out **selection Jaccard vs. the oracle** — 0.5B 0.33 → 1.5B 0.39 → 3B 0.55 → **7B 0.81**, with an accuracy-vs-latency **Pareto** (`scripts/eval_routers.py`, `scripts/plot_pareto.py`, `data/eval/pareto.png`). The 7B dominates the 3B. Adapters published to the HF Hub (private).

### [2] Application / Product
- **Backend:** FastAPI (`src/learning_memory_os/api.py`) over Postgres + **pgvector**; four-tier memory (semantic / student-mastery / episodic / intervention) plus an XTrace tier.
- **Frontend:** a no-build SPA (`web/`) — Home, **Interview Readiness**, Profile, Chat, Progress, Path, Your AI. Auth (sessions), and **Stripe billing** (`src/learning_memory_os/billing.py`) gating the Pro readiness report server-side.
- **Deployment:** self-host the trained router via **DigitalOcean GPU Droplet + vLLM** (`deploy/digitalocean/`) or Modal; the product can call it as a remote OpenAI-compatible endpoint.

### [3] Agent / automation
- **TutorAgent** (`src/learning_memory_os/agents/tutor.py`): mastery-aware prompting (skips mastered topics, targets weak ones, confronts active misconceptions), with a **diagnostic remediation** loop and **adaptive-difficulty** quizzes (difficulty scales with mastery + recent scores).
- **RoutingEngine** (heuristic, scored on relevance/recency/misconception/prerequisite/reuse) **or** the fine-tuned router, selectable per request.
- **Trace capture** (`src/learning_memory_os/memory/trace.py`): every turn is logged as a training trajectory for future per-user fine-tuning.

---

## Setup

Prereqs: Python 3.11+, [`uv`](https://github.com/astral-sh/uv), Docker.

```bash
git clone https://github.com/HivaMohammadzadeh1/CS153-frontier-systems.git
cd CS153-frontier-systems

cp .env.example .env          # fill in ANTHROPIC_API_KEY, OPENAI_API_KEY, DATABASE_URL
docker compose up -d db       # Postgres + pgvector on :5433
uv sync                       # install deps
uv run python scripts/migrate.py   # apply migrations/ (schema, auth, billing, mastery_history)
uv run pytest                 # (optional) tests
```

Optional env (off by default): `STRIPE_SECRET_KEY` / `STRIPE_PRICE_ID` / `STRIPE_WEBHOOK_SECRET` (billing), `LMOS_ROUTER_ENDPOINT` (use a remotely-hosted fine-tuned router).

## Usage

```bash
uv run python -m scripts.serve --port 8000     # web app -> http://localhost:8000
```
Sign up, then: **Chat** (ask anything; pick a topic; hit 🎯 Test for an adaptive quiz),
**🧭 Readiness** (your interview-readiness %, gaps, drill plan — Pro), **Progress** /
**Path** (mastery + curriculum map). The "Context for this answer" panel shows *why*
each memory item was selected.

Other entry points:
- `uv run streamlit run scripts/app.py` — backend-dev Streamlit view.
- Fine-tune the routers on the cluster: see `cluster/README.md`.
- Evaluate + plot the Pareto: `scripts/eval_routers.py`, `scripts/plot_pareto.py`.
- Merge a LoRA for serving: `python -m scripts.merge_adapter --size qwen2_5_7b --out data/merged/qwen2_5_7b`.
- Deploy the 7B router: `deploy/digitalocean/README.md`.

## Repository layout

```
src/learning_memory_os/   backend: api, agents, router, selector, memory, eval, trajectories, auth, billing
web/                      no-build SPA (index.html, app.js, styles.css)
scripts/                  CLIs: serve, finetune_router, eval_routers, merge_adapter, generate_trajectories, …
cluster/                  SLURM fine-tuning sweep (sbatch + runbook) for the H100 cluster
deploy/digitalocean/      GPU-Droplet + vLLM hosting for the trained router
config/                   router_sizes.yaml, topics.yaml (the curriculum)
migrations/               Postgres schema (001–008)
data/trajectories/        routing training/eval data
docs/superpowers/specs/   design specs for each subsystem
```

## AI usage disclosure

- **Built with [Claude Code](https://claude.com/claude-code)** (Anthropic) as the primary coding agent, under human direction, throughout the project. Design specs in `docs/superpowers/specs/` were drafted collaboratively.
- **Runtime models:** the **tutor's explanations** are generated by **Anthropic Claude** (Opus/Sonnet); **embeddings** use OpenAI `text-embedding-3-small`; the **context routers** are **our own** fine-tunes of **Qwen2.5-Instruct** (LoRA), trained for this project.
- No project code was copied from external repositories; third-party libraries are used via their public packages (see citations).

## Citations & acknowledgements

- **Curriculum** distilled from Stanford **CS336** (Language Modeling from Scratch) and **CS349D** (Systems for ML); topic list in `config/topics.yaml`.
- **Base models:** [Qwen2.5-Instruct](https://huggingface.co/Qwen) (Alibaba). **Serving:** [vLLM](https://github.com/vllm-project/vllm). **Fine-tuning:** 🤗 [PEFT](https://github.com/huggingface/peft) + [Transformers](https://github.com/huggingface/transformers).
- **Stack:** FastAPI, PostgreSQL + [pgvector](https://github.com/pgvector/pgvector), Stripe, Streamlit, Tailwind.
- **Compute:** shared 32×H100 SLURM cluster (Omniva / Teleport access), provided for CS 153.

## External resources

- GitHub: https://github.com/HivaMohammadzadeh1/CS153-frontier-systems
- Fine-tuned router adapters (HF Hub, private): `hivamoh/lmos-router-qwen2_5_{0_5b,1_5b,3b,7b}`
- Demo video: *(link in Gradescope submission)*

## License / academic honesty

Coursework for CS 153. Built during the course. See AI usage disclosure above.

# AGENTS.md

Project-level instructions for coding agents working on Learning Memory OS.

## Project Thesis

Learning Memory OS is a CS153 frontier systems project: a context-routed tutoring system for ML systems engineers. The system maintains structured learner memory, selects task-specific context under budgets, tutors the author as student-zero, and later benchmarks learned context routers against retrieval and heuristic baselines.

The 17-day deliverable is not a generic chatbot. It is an evidence-producing system with:

- ingestion from course and paper materials into structured artifacts
- four-tier memory: semantic, student, episodic, intervention
- heuristic context selection for the MVP
- tutor agent responses grounded in selected context
- interaction logs suitable for ablations, student-zero evaluation, and the final writeup

## Canonical Sources

Read these before making architectural decisions:

- `docs/superpowers/specs/2026-05-12-learning-memory-os-design.md` - product and research spec
- `docs/superpowers/plans/2026-05-13-plan-1-mvp-system.md` - current implementation plan
- `README.md` - public project overview; keep it concise and demo-oriented

If code and docs disagree, prefer the implementation plan for immediate MVP work and update the docs in the same change if the decision is intentional.

## Current Priority

Ship Plan 1 first: ingestion -> four-tier memory -> heuristic selector -> tutor agent -> JSONL logs.

Do not start router fine-tuning, curriculum-scale content generation, a dashboard, or a polished app until the Plan 1 end-to-end loop works on the four seed topics.

The MVP success path is:

1. start Postgres with pgvector
2. ingest one seed topic into structured artifacts
3. persist artifacts and learner state
4. ask a tutor question
5. select context under a token budget
6. generate a tutor response
7. log selected context, dropped context, response, latency, and cost metadata

## Execution Rules

- Work in small, reviewable commits aligned with the task order in the plan.
- Prefer tests before or alongside implementation for pure logic, schema contracts, and wrappers.
- Keep modules separated by responsibility:
  - `schemas/` owns Pydantic contracts
  - `memory/` owns Postgres access
  - `selector/` is mostly pure scoring and packing logic
  - `ingestion/` converts source text to artifacts
  - `agents/` orchestrates selector calls and LLM responses
  - `logging_utils/` records evaluation-ready traces
- Avoid hidden global state. Pass settings, clients, and student IDs explicitly unless an existing helper establishes a clear pattern.
- Favor deterministic behavior in tests. Mock Anthropic/OpenAI calls; do not require live API keys for unit tests.
- Keep all generated runtime data out of git: `.env`, `.venv/`, `logs/`, `data/runtime/`, caches, and database volumes.

## Implementation Style

- Python 3.11+.
- Package manager: `uv`.
- Runtime stack: Anthropic SDK, OpenAI SDK, Pydantic v2, psycopg, pgvector, structlog, Typer.
- Database: Postgres 16 with pgvector via Docker.
- Tests: `uv run pytest`.
- Lint/format target: use Ruff once configured.

Prefer typed, narrow functions over large framework objects. The selector and packer should be easy to test without a database.

## Research Guardrails

- Preserve the central empirical claim: compare context-routing strategies under token, latency, and cost budgets.
- Log enough metadata now so later ablations are possible. Do not throw away candidate scores, selected IDs, dropped IDs, budgets, or task type.
- Treat student-zero results as descriptive evidence, not statistical proof.
- Keep synthetic-router supervision clearly framed as distillation from a stronger oracle unless real user trajectories are added later.

## Definition of Done

A task is complete when:

- the code path is implemented
- targeted tests pass
- the relevant README/docs are updated if behavior or setup changed
- secrets are not committed
- the next agent can tell what changed and what remains

For MVP milestones, prefer a working vertical slice over broad incomplete scaffolding.


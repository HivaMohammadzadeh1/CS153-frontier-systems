Fix th# SKILLS.md

Project-specific playbooks for agents working on Learning Memory OS.

These are not separate products. They are reusable modes of work that help move the project from plan to demo without losing the research thread.

## Skill: MVP Plan Executor

Use when the user asks to "move forward," "continue," or "do the next task."

Workflow:

1. Open `docs/superpowers/plans/2026-05-13-plan-1-mvp-system.md`.
2. Find the earliest unchecked task that is not blocked by missing credentials, GPU access, or external course material.
3. Implement only that task or one coherent subtask.
4. Run the targeted verification named in the plan.
5. Update the checkbox state only for work actually completed.

Output should include changed files, verification result, and the next concrete task.

## Skill: Architecture Keeper

Use when adding modules, changing data flow, or resolving design ambiguity.

Guardrails:

- Keep the four-tier memory boundary intact.
- Keep selector logic testable without Postgres.
- Keep agents thin: agents call selectors and LLM wrappers; they should not perform direct global retrieval.
- Keep ingestion schema-first: extracted artifacts must validate before persistence.
- Keep logs evaluation-ready.

Decision rule: if a shortcut would make the final ablation or student-zero trace impossible, do not take it.

## Skill: Backend Implementer

Use for database, schemas, service modules, and CLIs.

Checklist:

- Add or update Pydantic models before relying on implicit dictionaries.
- Mock external APIs in unit tests.
- Use parameterized SQL or typed client calls; no string-built SQL from user text.
- Keep database migrations idempotent where practical.
- Ensure CLIs fail with actionable messages when `.env`, Docker, or API keys are missing.

Preferred verification:

```bash
uv run pytest tests/unit -v
docker compose exec db psql -U lmos -d learning_memory_os -c "\\dt"
```

## Skill: Context Selector Engineer

Use for `selector/scoring.py`, `selector/pack.py`, and future router baselines.

MVP heuristic features:

- semantic relevance
- recency decay
- active misconception boost
- prerequisite boost
- successful reuse boost
- token-budget-aware packing

Test cases should cover:

- stable tie-breaking
- budget overflow
- zero-token or missing-token edge cases
- misconception-priority behavior
- redundant candidates not crowding out prerequisites

Always expose enough scoring detail for logs and later ablation tables.

## Skill: Evaluation and Logging Engineer

Use whenever tutor sessions, quizzes, selector calls, or generated trajectories are added.

Each interaction log should capture:

- timestamp
- student ID
- task type
- user query or task prompt
- candidate item IDs
- per-candidate score components
- selected item IDs
- dropped high-ranking item IDs
- token budget and estimated token usage
- model name
- latency
- estimated cost if available
- final response
- optional user feedback or correction

Prefer append-only JSONL for the MVP. Do not replace this with a dashboard until the logs support the ablation tables.

## Skill: Research Writeup Steward

Use when editing README, project page copy, report sections, or demo scripts.

Keep the story precise:

- Long-term mission: train ML systems engineers.
- Short-term artifact: context-routed tutor with learner memory.
- Technical centerpiece: context selection under budgets.
- Headline result: accuracy-vs-cost Pareto frontier for routing strategies and model sizes.
- Evidence: synthetic eval, student-zero trajectory, optional qualitative cohort.

Avoid overstating:

- n=1 student-zero evidence
- synthetic-only router training
- any result not yet measured

## Skill: Scope Controller

Use near deadlines or when tasks sprawl.

Cut in this order if behind:

1. dashboard polish
2. cohort evaluation
3. CS336 assignment case study
4. router sizes beyond the minimum viable sweep
5. combinatorial selector

Never cut:

- at least one working tutor loop
- interaction logging
- one learned-router result if router training has started
- student-zero trace
- demo video and final writeup


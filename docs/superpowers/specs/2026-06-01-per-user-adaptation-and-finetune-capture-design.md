# Per-User Adaptation + Fine-Tune Capture — Design Spec

Date: 2026-06-01
Status: Implemented

## Goal

Make the agent **adapt to each user automatically** and **capture each user's real
memory/history/context in a fine-tune-ready format** — usable for both a context
router fine-tune (now) and a tutor fine-tune (later). Approach C: capture once,
richly.

## Components

- **Learning trace store** (`memory/trace.py`, migration `005 learning_traces`):
  one row per chat turn — a superset of the synthetic `Trajectory`
  (student_state, task, candidate_pool, selection) plus tutor/RL fields
  (reply, model, reward). `record_turn` (best-effort, never breaks a reply),
  `attach_reward` (labels the latest turn), `export_trajectories` (maps rows →
  `Trajectory`, with `min_reward` filtering), `count`/`recent`/`delete_for_student`.
  Rows use `clock_timestamp()` so turns in one transaction stay ordered.
- **Always-on personalization** (`agents/profile.py` → `LearnerProfile`):
  consolidates decayed mastery, strengths/gaps, active misconceptions, and
  due-for-review into one snapshot built every turn. Drives the tutor's STUDENT
  PROFILE block (now including due-for-review) and backs `GET /api/student/{id}/profile`.
- **Capture + reward wiring** (`api.py`): `/chat` builds the profile, routes,
  then records the turn. `/feedback` (👍/👎 → ±1) and `/quiz/score` (score) call
  `attach_reward`. The quiz endpoint now passes the raw score to `update_mastery`
  (relies on the A3 confidence-weighted blend) instead of a second manual EMA.
- **Export path**: `scripts/export_traces.py` + `GET /api/student/{id}/traces/export`
  → `Trajectory` JSONL that `scripts/finetune_router.py` consumes directly.
- **"Your AI" view** (frontend, `web/`): a new nav entry showing what Memex has
  learned (mastery ring, strengths/working-on, misconceptions, due-for-review),
  how much data is captured (trace count + recent turns with reward dots), and
  **consent controls** (Export my data / Reset my data → `DELETE /traces`).

## Data flow
`chat → build_profile → route → record_turn → reply`; later `feedback|quiz →
attach_reward`; `export → JSONL → finetune_router`.

## Error handling
Capture is wrapped in try/except (logs `trace_capture_failed`, never breaks chat).
Migration idempotent; reward nullable; export tolerates missing fields.

## Testing
+5 tests: trace record/export round-trip, reward-latest + `min_reward` filter,
delete, profile strength/weakness classification, profile misconceptions + prompt
block. Full suite green (131). Endpoints + "Your AI" view verified live
(light + dark) against seeded data.

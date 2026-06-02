# Memory System Improvements — Design Spec

Date: 2026-05-31
Status: Implemented

## Goal

Make the multi-tier memory system **correct** and **pedagogically smarter**, in two
tracks, without breaking the working tutor loop (was 112 tests green).

## Track A — correctness & retrieval quality (no schema change)

- **A1 · Misconception signal fix.** The selector's misconception boost (weight
  0.8) compared a `semantic_items` id against a set of *misconception-row ids* —
  different ID spaces, so it never fired. Fix: build the boost set from each
  active misconception's `concept_id` (exact, 1.0) and the topic that concept (or
  the misconception itself) belongs to (fallback, 0.4). `ScoringContext` field
  renamed `active_misconception_titles` → `misconception_concept_ids` (+ a new
  `misconception_topics`) so the contract is self-describing.
- **A2 · MMR diversity in packing.** `pack_under_budget` gains a `diversity` (λ)
  param; each pick maximizes `(1-λ)·score − λ·max_cosine_to_selected`, suppressing
  near-duplicate chunks. Falls back to greedy when λ=0 or embeddings are absent.
  Engine default λ=0.7.
- **A3 · Confidence-weighted mastery.** `update_mastery` blends new evidence with
  the prior in proportion to confidence (atomic SQL upsert), so one noisy quiz
  can't wipe history; confidence accrues toward 1.0.

## Track B — spaced repetition / forgetting (migrations 003 + 004, small UI)

- **B1 · Decay-on-read** (`memory/decay.py`): effective mastery =
  `score · 0.5^(age_days / half_life)`, half-life 3–33 days scaling with
  score×confidence. Stored scores are never mutated; `/progress` returns the
  decayed value.
- **B2 · SM-2 scheduling** (migration 003: `reps`, `interval_days`,
  `next_review_at`): passing grades (≥0.6) grow the interval (1→3→×2, capped 365d);
  failing resets to 1 day.
- **B3 · Due-for-review surfacing**: `StudentStore.due_for_review()` +
  `GET /api/student/{id}/review` + a Home "Due for review" tile that seeds a
  review session.
- **B4 · Selector resurfacing**: a low-weight (0.3) `review_due` signal so the
  tutor naturally revisits decaying concepts.

### Misconception → topic wiring (migration 004)

So A1 actually fires in the real flow, misconceptions gained a `topic_id` column;
the diagnostic-turn endpoint threads the active topic when it records a confirmed
misconception, and the API folds misconception `topic_id`s into the boost set.

## Error handling
All new signals default to 0 when data is absent; MMR degrades to greedy without
embeddings; decay returns the raw score when `last_updated`/columns are null;
pre-migration rows are backfilled idempotently.

## Testing
13 new tests: misconception exact/topic/none + review-due signals, MMR drops a
duplicate, decay monotonicity/half-life, mastery blend math, SM-2 advance/reset,
due-filtering, misconception-topic round-trip. Full suite green (126).

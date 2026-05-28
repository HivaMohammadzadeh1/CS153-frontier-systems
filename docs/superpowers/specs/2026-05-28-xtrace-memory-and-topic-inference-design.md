# XTrace long-horizon memory + conversation-level topic inference

Date: 2026-05-28
Status: approved, in implementation

## Why

Two gaps in the week-6 MVP surfaced while testing the Streamlit demo:

1. The four Postgres memory tiers track per-concept mastery and per-session episodics, but they don't store free-form natural-language facts about *what the student has said about themselves and their work*. Recall is keyed on concept IDs, not on "the student previously mentioned they're implementing a KV cache."
2. The "Topic focus" dropdown defaults to `(global vector search)`. When a user asks an anaphoric question like "what's the most common misconception about this topic?", the tutor has no idea what "this topic" refers to and falls back to asking the user to clarify. From the screenshot, this is the dominant failure mode.

This design adds a fifth memory tier backed by XTrace Memory Manager
(`mem.xtrace.ai`) for long-horizon free-form student recall, and a
conversation-level topic-inference pass that replaces the dropdown's manual
default.

## Non-goals

- Migrating existing Postgres episodic data into XTrace.
- Tuning XTrace recall hyperparameters beyond a top-K and similarity threshold
  pair chosen by inspection.
- Per-`(student_id, topic_id)` namespacing in XTrace. We use one namespace per
  student so cross-topic recall works.
- A TypeScript sidecar for XTrace (their Python SDK is roadmapped but we call
  the REST API directly).
- Touching the production-style FastAPI frontend; this work targets the
  Streamlit demo path that the screenshot came from.

## Architecture

```
                              ┌──────────────────────────────┐
   user turn ──▶ tutor REPL ──┤  topic inference (new)        │
                              │  embed(last 4 turns + curr)   │
                              │  cosine vs topic centroids    │
                              └──────────────┬────────────────┘
                                             │ topic_id, confidence
                                             ▼
                              ┌──────────────────────────────┐
                              │  selector.engine             │
                              │  ├─ semantic store (Postgres)│
                              │  ├─ student store  (Postgres)│
                              │  ├─ episodic store (Postgres)│
                              │  ├─ intervention store (PG)  │
                              │  └─ xtrace store (NEW, REST) │ ◀── recall(student_id, query)
                              └──────────────┬───────────────┘
                                             │ scored, budgeted pack
                                             ▼
                                       tutor prompt
                                             │
   xtrace ingest (async) ◀──────────── user turn (Fact)
   xtrace ingest (on session end) ◀── session summary (Episode)
```

## Components

### 1. `src/learning_memory_os/memory/xtrace.py` (new)

Thin REST wrapper around XTrace Memory Manager.

Public interface:

```python
class XTraceClient:
    def __init__(self, api_key: str, org_id: str, base_url: str, *, client: httpx.Client | None = None): ...
    def ingest_fact(self, student_id: str, text: str) -> None: ...
    def ingest_episode(self, student_id: str, summary: str) -> None: ...
    def recall(self, student_id: str, query: str, *, k: int = 5) -> list[XTraceMemoryItem]: ...
```

- One `httpx.Client` per `XTraceClient` instance. Initialized from
  `XTRACE_API_KEY`, `XTRACE_ORG_ID`, `XTRACE_BASE_URL` env vars.
- `XTraceMemoryItem` is a Pydantic model with `id`, `text`, `kind`
  (`fact` | `artifact` | `episode`), `similarity`, `created_at`.
- All recall errors return `[]` and log a warning. Ingest errors log and
  return. A circuit breaker (instance-level `_unhealthy` flag with
  exponential backoff) prevents repeated HTTP attempts when the service is
  down.
- No live HTTP in unit tests. Tests mock at `httpx.MockTransport`.

### 2. `src/learning_memory_os/agents/topic_inference.py` (new)

```python
@dataclass
class TopicInferenceResult:
    topic_id: str | None
    confidence: float          # 0–1 cosine similarity to top centroid
    decision: Literal["auto", "inferred", "ask"]

def infer_topic(
    conversation: list[Message],
    topics: list[Topic],
    embed_fn: Callable[[str], list[float]],
    *,
    history_turns: int = 4,
) -> TopicInferenceResult: ...
```

- Builds an embedding from `history_turns` most recent user turns + current
  turn (newline-joined).
- Per-topic centroid = mean of seed-doc embeddings, computed once at startup
  and cached on a `TopicCentroids` object passed in (not recomputed each call).
- Decision thresholds:
  - confidence ≥ 0.6 → `auto` (use silently).
  - 0.4 ≤ confidence < 0.6 → `inferred` (use, show "(inferred)" in sidebar).
  - confidence < 0.4 → `ask` (fall through to current "which topic?"
    behavior).
- Embedding failures raise; caller catches and treats as `ask`.

### 3. `src/learning_memory_os/selector/engine.py` (changed)

The selector grows one more candidate source. Each `XTraceMemoryItem` becomes
a `MemoryItem(kind="xtrace", text=hit.text, score_input=hit.similarity)` and
goes through the existing scoring and budgeted packing. The "show context
analysis" panel already reads from selector output, so XTrace hits become
observable for free (with `kind="xtrace"` as the badge).

XTrace candidate scoring: `score = hit.similarity * recency_decay(hit.created_at)`
with the same decay function used for episodic items.

### 4. `src/learning_memory_os/agents/tutor.py` (changed)

- Before retrieval, if `topic_id is None` and the sidebar is on `(auto)`,
  call `infer_topic`. Record the result on the turn metadata.
- After generating the reply, fire-and-forget call
  `xtrace.ingest_fact(student_id, user_message)` on a background thread.
  Errors logged, never raised.
- On explicit "End session" button (new) or after N idle minutes, call
  `xtrace.ingest_episode(student_id, summary)` where `summary` is generated
  by an LLM pass over the session transcript.

### 5. `scripts/app.py` (changed)

- Replace `(global vector search)` with `(auto)` as the first dropdown option.
- When `(auto)` is selected, show `inferred: <topic> (conf: 0.74)` immediately
  below the dropdown, refreshing on each turn.
- Manual selection still overrides inference.
- Add an "End session" button next to "Clear chat" that triggers episode
  ingest.

## Data flow per turn

1. User submits a message.
2. If sidebar is on `(auto)`, `infer_topic` runs and returns
   `(topic_id, confidence, decision)`.
3. Selector pulls candidates from all five stores:
   - Postgres (4 tiers): unchanged.
   - XTrace: `recall(student_id, user_message, k=5)`.
4. Selector scores, packs to budget, returns the chosen pack.
5. Tutor generates a reply with the packed context.
6. Background task: `xtrace.ingest_fact(student_id, user_message)`. Logs on
   failure.
7. On session end (button or idle): `xtrace.ingest_episode(student_id, summary)`.

## Error handling

| Failure | Behavior |
|---|---|
| XTrace recall HTTP 5xx / timeout | Return `[]`, log warning, set `_unhealthy=True` for 60s. |
| XTrace ingest failure | Log, return. Tutor reply is not affected. |
| Embedding API failure during topic inference | Treat as decision=`ask`. |
| Topic centroids unavailable at startup | Disable inference; sidebar shows `(global vector search)` as fallback. |
| Confidence < 0.4 | decision=`ask`; tutor asks the user to clarify topic (current behavior). |

## Testing

- `tests/unit/test_xtrace_client.py` — `httpx.MockTransport` fixture, assert
  request shape (auth header, JSON body), parse responses into Pydantic
  models, exercise error paths (4xx, 5xx, timeout) returning `[]` /
  no-raise.
- `tests/unit/test_topic_inference.py` — fixture conversations + fixture
  centroids, assert top-1 topic and decision-bucket boundaries (0.4, 0.6).
- `tests/unit/test_selector_with_xtrace.py` — selector receives a mocked
  XTrace candidate, verify it competes for budget and appears in the
  returned pack with `kind="xtrace"`.
- `tests/integration/test_xtrace_live.py` (skipped unless
  `XTRACE_LIVE_TESTS=1`) — round-trip ingest + recall against the real
  service.

No live API calls in `uv run pytest` defaults.

## Environment

New env vars in `.env.example` (and required for the Streamlit demo):

```
XTRACE_API_KEY=
XTRACE_ORG_ID=
XTRACE_BASE_URL=https://api.mem.xtrace.ai
```

If `XTRACE_API_KEY` is unset, the tutor logs a warning at startup and the
XTrace candidate source returns `[]` — the Postgres-only path keeps working.

## Risks

- **XTrace Python SDK gap.** We call REST directly. If their API shape
  changes before they ship the SDK, we own the migration.
- **Recall noise.** Per-turn writes may flood the index with low-signal user
  questions. Mitigation: the end-of-session Episode is the high-signal
  artifact; recall ranks by similarity so noise is naturally deprioritized.
- **Inference miscalibration.** Cosine thresholds (0.4, 0.6) are chosen by
  inspection. We log per-turn (`topic_id`, `confidence`, `decision`) to JSONL
  so we can re-tune from real session data.
- **Streamlit auto-rerun.** Topic inference must be cached per `(student_id,
  message_id)` to avoid re-embedding on every Streamlit rerun.

## What changes for the writeup

The four-tier memory framing becomes a five-tier one. The new tier has a
clear job description that the existing four don't cover: free-form
long-horizon recall of what the student said, distinct from per-concept
mastery scoring. The writeup will note that XTrace is a hosted dependency,
not first-party.

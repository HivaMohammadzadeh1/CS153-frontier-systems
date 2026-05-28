# Plan 3 — Synthetic Trajectory Generator + LoRA Router Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a large synthetic tutoring-trajectory dataset, LoRA-fine-tune Qwen-2.5-Instruct routers across multiple sizes (0.5B / 1.5B / 3B / 7B), and produce the **accuracy-vs-cost Pareto frontier** that is the project's headline empirical result.

**Architecture:** A trajectory is `(student_state, task, candidate_pool, oracle_selection)`. The **oracle** is a strong frontier LLM (Claude Sonnet or Opus) given the full pool and asked which subset best answers the task; it produces ground-truth selections that smaller models will learn to imitate. We generate 5K trajectories first (validation), then scale to 50K. Each router model is fine-tuned with LoRA on the same dataset. Evaluation runs all fine-tuned routers + heuristic baselines + frontier-API baseline on a held-out test split; we plot accuracy vs inference cost.

**Tech Stack:** Same as Plans 1–2 (Python 3.11+, uv, Postgres, Anthropic SDK). New: `transformers`, `peft`, `accelerate`, `datasets`, `torch`, `bitsandbytes` (4-bit quantization for base models), `matplotlib` (Pareto plot).

**Spec reference:** `docs/superpowers/specs/2026-05-12-learning-memory-os-design.md` §2.3 Phase 3, §4.2 (synthetic trajectories), §4.4 (size × strategy ablation), §4.5 (Pareto frontier).

**Prior plans:** Plan 1 (`mvp-week6`) and Plan 2 (`curriculum-loaded`) are assumed complete. Plan 3 reuses `SemanticStore` and `Embedder`.

---

## Critical pre-flight (do this BEFORE Task 1)

**GPU availability check.** Plan 3's training tasks need real GPU. If you don't have one ready, every training task will block.

Run on the deployment machine (not necessarily the dev laptop):
```bash
nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), '; device count:', torch.cuda.device_count()); print('device 0:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Decision tree:
- **H100/A100 40GB+**: full Plan 3 viable (all sizes)
- **L4 / A10 / RTX 4090 (24GB)**: 0.5B/1.5B/3B viable; 7B requires QLoRA + memory-efficient settings
- **Apple Silicon (MPS)**: 0.5B/1.5B viable (slow but works); 3B borderline; 7B no
- **CPU only**: 0.5B only as a slow proof-of-concept; cut everything else

If the deployment GPU is unclear, dispatch a "verify-compute" subagent first to run `nvidia-smi` and pin down what we have. Adjust the Pareto sweep set accordingly.

**Budget check.**
- Synthetic data generation: ~50K trajectories × ~3K oracle tokens × $3/M input (Claude Sonnet) ≈ **$450**. Use Haiku or Sonnet, not Opus, for oracle.
- Frontier-API baseline (eval-time): ~2K eval trajectories × ~3K tokens × $3/M ≈ **$18**.
- Embeddings (already in DB) and router inference are negligible.

---

## File Structure

Plan 3 adds:

```
CS153-frontier-systems/
├── config/
│   └── router_sizes.yaml             # NEW: list of fine-tune targets
├── src/learning_memory_os/
│   ├── trajectories/
│   │   ├── __init__.py
│   │   ├── schemas.py                # NEW: Trajectory, OracleSelection types
│   │   ├── generator.py              # NEW: build one trajectory
│   │   ├── sampler.py                # NEW: sample student profiles / candidate pools
│   │   └── serializer.py             # NEW: trajectory ↔ training pair (str → str)
│   ├── router/
│   │   ├── __init__.py
│   │   ├── prompt.py                 # NEW: router prompt template + parser
│   │   ├── finetune.py               # NEW: LoRA training entry point
│   │   ├── infer.py                  # NEW: load router + predict
│   │   └── frontier_api.py           # NEW: Claude/GPT as a router (baseline)
│   └── eval/
│       ├── __init__.py
│       ├── router_eval.py            # NEW: accuracy + cost metrics
│       └── pareto.py                 # NEW: plot the frontier
├── scripts/
│   ├── generate_trajectories.py      # NEW
│   ├── finetune_router.py            # NEW
│   ├── eval_routers.py               # NEW
│   └── plot_pareto.py                # NEW
├── data/
│   ├── trajectories/                 # generated dataset shards (gitignored beyond manifest)
│   └── router_checkpoints/           # LoRA adapters (gitignored)
└── tests/
    ├── unit/
    │   ├── test_trajectory_schemas.py
    │   ├── test_trajectory_generator.py
    │   ├── test_trajectory_serializer.py
    │   ├── test_router_prompt.py
    │   └── test_router_eval.py
    └── integration/
        └── test_trajectory_pipeline.py
```

---

## Task 1: Trajectory schemas

**Files:**
- Create: `src/learning_memory_os/trajectories/__init__.py`
- Create: `src/learning_memory_os/trajectories/schemas.py`
- Test: `tests/unit/test_trajectory_schemas.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_trajectory_schemas.py`:
```python
from learning_memory_os.trajectories.schemas import (
    StudentState,
    Trajectory,
    PoolItem,
    TaskType,
)


def test_pool_item_minimal():
    p = PoolItem(id="abc12345", title="KV cache", body_excerpt="A cache of K and V.", token_estimate=50)
    assert p.id == "abc12345"


def test_trajectory_round_trip():
    state = StudentState(
        student_id="s1",
        mastery={"kv_cache": 0.3, "tokenization": 0.8},
        active_misconceptions=["KV cache stores token ids"],
        recent_episodic_ids=["ev1", "ev2"],
    )
    t = Trajectory(
        id="traj-0001",
        student_state=state,
        task_type=TaskType.EXPLAIN,
        task_text="What is the KV cache?",
        budget=2000,
        candidate_pool=[
            PoolItem(id="aaaa1111", title="A", body_excerpt="x", token_estimate=100),
            PoolItem(id="bbbb2222", title="B", body_excerpt="y", token_estimate=100),
        ],
        oracle_selection=["aaaa1111"],
    )
    serialized = t.model_dump()
    assert serialized["id"] == "traj-0001"
    assert serialized["task_type"] == "explain"
    assert serialized["oracle_selection"] == ["aaaa1111"]
```

- [ ] **Step 2: Verify failure**

`uv run pytest tests/unit/test_trajectory_schemas.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement schemas**

Create `src/learning_memory_os/trajectories/__init__.py` (empty).

Create `src/learning_memory_os/trajectories/schemas.py`:
```python
from enum import Enum
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    EXPLAIN = "explain"
    QUIZ = "quiz"
    REVIEW = "review"
    LAB = "lab"


class PoolItem(BaseModel):
    """A single candidate item the router can choose to include."""
    id: str                # 8-char short id (or full uuid; we'll use short)
    title: str
    body_excerpt: str
    token_estimate: int


class StudentState(BaseModel):
    student_id: str
    mastery: dict[str, float] = Field(default_factory=dict)        # concept_id -> 0..1
    active_misconceptions: list[str] = Field(default_factory=list)
    recent_episodic_ids: list[str] = Field(default_factory=list)


class Trajectory(BaseModel):
    id: str
    student_state: StudentState
    task_type: TaskType
    task_text: str
    budget: int                           # token budget for the routing decision
    candidate_pool: list[PoolItem]
    oracle_selection: list[str]           # subset of candidate_pool ids
```

- [ ] **Step 4: Run test**

`uv run pytest tests/unit/test_trajectory_schemas.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/learning_memory_os/trajectories tests/unit/test_trajectory_schemas.py
git commit -m "feat(trajectories): pydantic schemas (Trajectory, PoolItem, StudentState)"
```

---

## Task 2: Trajectory sampler — student profiles + candidate pools from real DB

**Files:**
- Create: `src/learning_memory_os/trajectories/sampler.py`
- Test: `tests/integration/test_sampler.py`

The sampler reads the real semantic-items DB and produces realistic (StudentState, candidate_pool) pairs.

- [ ] **Step 1: Write failing integration test**

Create `tests/integration/test_sampler.py`:
```python
from learning_memory_os.trajectories.sampler import (
    sample_candidate_pool,
    sample_student_state,
)


def test_sample_candidate_pool_returns_items(db_conn):
    pool = sample_candidate_pool(
        db_conn, target_topic="kv_cache", pool_size=10
    )
    assert 1 <= len(pool) <= 10
    assert all(p.id and p.title and p.body_excerpt for p in pool)


def test_sample_student_state_returns_realistic_shape(db_conn):
    state = sample_student_state(
        db_conn, student_id="synthetic-1", target_concepts=["kv_cache", "quantization"]
    )
    assert state.student_id == "synthetic-1"
    # mastery is bounded
    for v in state.mastery.values():
        assert 0.0 <= v <= 1.0
```

- [ ] **Step 2: Verify failure**

`uv run pytest tests/integration/test_sampler.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement sampler**

Create `src/learning_memory_os/trajectories/sampler.py`:
```python
import random
import psycopg
from .schemas import PoolItem, StudentState


def sample_candidate_pool(
    conn: psycopg.Connection,
    *,
    target_topic: str,
    pool_size: int = 15,
    other_topic_noise: int = 5,
) -> list[PoolItem]:
    """Sample a candidate pool: mostly target-topic artifacts + some distractors.

    The mix mirrors what the real selector sees in production.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id::text AS id, title, body
            FROM semantic_items WHERE topic_id = %s
            ORDER BY random() LIMIT %s
            """,
            (target_topic, max(0, pool_size - other_topic_noise)),
        )
        target_rows = list(cur.fetchall())

        cur.execute(
            """
            SELECT id::text AS id, title, body
            FROM semantic_items WHERE topic_id <> %s
            ORDER BY random() LIMIT %s
            """,
            (target_topic, other_topic_noise),
        )
        noise_rows = list(cur.fetchall())

    items: list[PoolItem] = []
    for r in target_rows + noise_rows:
        body = r["body"] or ""
        excerpt = body[:300]
        items.append(
            PoolItem(
                id=r["id"][:8],   # short id
                title=r["title"],
                body_excerpt=excerpt,
                token_estimate=max(1, len(body) // 4),
            )
        )
    random.shuffle(items)
    return items


def sample_student_state(
    conn: psycopg.Connection,
    *,
    student_id: str,
    target_concepts: list[str],
) -> StudentState:
    """Synthesize a plausible student state: mastery values per concept + a few misconceptions."""
    mastery: dict[str, float] = {}
    for c in target_concepts:
        # Bimodal: half concepts the student "knows", half they don't
        mastery[c] = random.choice([random.uniform(0.0, 0.4), random.uniform(0.6, 1.0)])

    # Maybe 0-2 active misconceptions (pulled from the DB pool of misconception artifacts)
    misconceptions: list[str] = []
    with conn.cursor() as cur:
        cur.execute(
            "SELECT body FROM semantic_items WHERE artifact_type = 'misconception' "
            "ORDER BY random() LIMIT 2"
        )
        for r in cur.fetchall():
            if random.random() < 0.5:
                # Short excerpt of the misconception body
                misconceptions.append((r["body"] or "")[:200])

    return StudentState(
        student_id=student_id,
        mastery=mastery,
        active_misconceptions=misconceptions,
        recent_episodic_ids=[],
    )
```

- [ ] **Step 4: Run test**

`uv run pytest tests/integration/test_sampler.py -v`
Expected: PASS. (Requires real DB with content — Plan 2 must be done.)

- [ ] **Step 5: Commit**

```bash
git add src/learning_memory_os/trajectories/sampler.py tests/integration/test_sampler.py
git commit -m "feat(trajectories): candidate pool + student state samplers"
```

---

## Task 3: Trajectory generator (oracle call)

**Files:**
- Create: `src/learning_memory_os/trajectories/generator.py`
- Test: `tests/unit/test_trajectory_generator.py`

- [ ] **Step 1: Write failing test (mocked oracle)**

Create `tests/unit/test_trajectory_generator.py`:
```python
from unittest.mock import MagicMock
from learning_memory_os.trajectories.schemas import (
    StudentState,
    PoolItem,
    TaskType,
    Trajectory,
)
from learning_memory_os.trajectories.generator import build_trajectory


def test_build_trajectory_calls_oracle_and_packs_result():
    fake_llm = MagicMock()
    fake_llm.complete_json.return_value = {
        "selected_ids": ["aaaa1111", "cccc3333"],
        "rationale": "These two items directly explain the KV cache.",
    }

    state = StudentState(student_id="s1", mastery={}, active_misconceptions=[], recent_episodic_ids=[])
    pool = [
        PoolItem(id="aaaa1111", title="A", body_excerpt="kv cache stores K and V", token_estimate=100),
        PoolItem(id="bbbb2222", title="B", body_excerpt="unrelated topic", token_estimate=100),
        PoolItem(id="cccc3333", title="C", body_excerpt="why kv cache exists", token_estimate=100),
    ]

    t = build_trajectory(
        traj_id="traj-1",
        student_state=state,
        task_type=TaskType.EXPLAIN,
        task_text="What is a KV cache?",
        budget=300,
        candidate_pool=pool,
        oracle_llm=fake_llm,
    )
    assert isinstance(t, Trajectory)
    assert t.oracle_selection == ["aaaa1111", "cccc3333"]
    fake_llm.complete_json.assert_called_once()
```

- [ ] **Step 2: Verify failure**

`uv run pytest tests/unit/test_trajectory_generator.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement generator**

Create `src/learning_memory_os/trajectories/generator.py`:
```python
from ..llm import LLM
from .schemas import StudentState, PoolItem, Trajectory, TaskType


ORACLE_SYSTEM = """You are an expert ML systems engineer tutor selecting CONTEXT for a tutoring agent.
Given the student's state, a task, a token budget, and a pool of candidate items, you choose the SUBSET
of pool items that the tutor should use to answer the task.

Rules:
1. Total tokens of selected items MUST not exceed the budget.
2. Prefer items that directly address the task.
3. Prefer items that resolve the student's active misconceptions, if any are listed.
4. Prefer items targeting concepts the student has LOW mastery on.
5. Skip redundant items (two items that say the same thing).

Return STRICT JSON with this shape:
{
  "selected_ids": ["<short_id>", "<short_id>", ...],
  "rationale": "<one-sentence summary of why these were chosen>"
}

No commentary outside JSON."""


def _format_pool(pool: list[PoolItem]) -> str:
    return "\n\n".join(
        f"[{p.id}] (tokens≈{p.token_estimate}) {p.title}\n  {p.body_excerpt}"
        for p in pool
    )


def _format_state(state: StudentState) -> str:
    parts = [f"student_id: {state.student_id}"]
    if state.mastery:
        parts.append("mastery: " + ", ".join(f"{k}={v:.2f}" for k, v in state.mastery.items()))
    if state.active_misconceptions:
        parts.append("active_misconceptions: " + "; ".join(state.active_misconceptions))
    return "\n".join(parts)


def build_trajectory(
    *,
    traj_id: str,
    student_state: StudentState,
    task_type: TaskType,
    task_text: str,
    budget: int,
    candidate_pool: list[PoolItem],
    oracle_llm: LLM,
) -> Trajectory:
    user = (
        f"STUDENT STATE:\n{_format_state(student_state)}\n\n"
        f"TASK TYPE: {task_type.value}\n"
        f"TASK: {task_text}\n"
        f"TOKEN BUDGET: {budget}\n\n"
        f"CANDIDATE POOL:\n{_format_pool(candidate_pool)}"
    )
    data = oracle_llm.complete_json(system=ORACLE_SYSTEM, user=user, max_tokens=1024)
    selected = list(data.get("selected_ids", []))
    pool_ids = {p.id for p in candidate_pool}
    selected = [s for s in selected if s in pool_ids]  # filter hallucinated ids
    return Trajectory(
        id=traj_id,
        student_state=student_state,
        task_type=task_type,
        task_text=task_text,
        budget=budget,
        candidate_pool=candidate_pool,
        oracle_selection=selected,
    )
```

- [ ] **Step 4: Run test**

`uv run pytest tests/unit/test_trajectory_generator.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/learning_memory_os/trajectories/generator.py tests/unit/test_trajectory_generator.py
git commit -m "feat(trajectories): oracle-driven trajectory builder"
```

---

## Task 4: Router prompt + serializer (training-pair format)

**Files:**
- Create: `src/learning_memory_os/router/__init__.py`
- Create: `src/learning_memory_os/router/prompt.py`
- Create: `src/learning_memory_os/trajectories/serializer.py`
- Test: `tests/unit/test_router_prompt.py`
- Test: `tests/unit/test_trajectory_serializer.py`

The router consumes plain text in, emits plain text out. Format MUST stay stable across all sizes and against frontier-API baseline.

- [ ] **Step 1: Write failing test for prompt**

Create `tests/unit/test_router_prompt.py`:
```python
from learning_memory_os.trajectories.schemas import StudentState, PoolItem, TaskType
from learning_memory_os.router.prompt import (
    format_router_input,
    parse_router_output,
)


def test_format_router_input_includes_all_sections():
    state = StudentState(student_id="s", mastery={"a": 0.2}, active_misconceptions=["x"], recent_episodic_ids=[])
    pool = [
        PoolItem(id="aaaa1111", title="A", body_excerpt="x", token_estimate=50),
        PoolItem(id="bbbb2222", title="B", body_excerpt="y", token_estimate=80),
    ]
    text = format_router_input(
        student_state=state,
        task_type=TaskType.EXPLAIN,
        task_text="explain A",
        budget=500,
        candidate_pool=pool,
    )
    assert "STUDENT" in text
    assert "TASK" in text
    assert "POOL" in text
    assert "aaaa1111" in text
    assert "bbbb2222" in text
    assert "500" in text


def test_parse_router_output_extracts_ids():
    out = "aaaa1111,bbbb2222"
    assert parse_router_output(out) == ["aaaa1111", "bbbb2222"]

    # Tolerates whitespace and stray characters
    assert parse_router_output("  aaaa1111, bbbb2222  \n") == ["aaaa1111", "bbbb2222"]

    # Tolerates [brackets]
    assert parse_router_output("[aaaa1111, bbbb2222]") == ["aaaa1111", "bbbb2222"]

    # Empty
    assert parse_router_output("") == []
    assert parse_router_output("none") == []
```

- [ ] **Step 2: Implement prompt module**

Create `src/learning_memory_os/router/__init__.py` (empty).

Create `src/learning_memory_os/router/prompt.py`:
```python
import re
from ..trajectories.schemas import StudentState, PoolItem, TaskType


ROUTER_INSTRUCTION = (
    "You are a context-selection router. Given the student state, a task, a token budget, "
    "and a pool of candidate items, output ONLY a comma-separated list of the item IDs you "
    "select. No prose, no JSON, no brackets. Total tokens of selected items must not exceed "
    "the budget. Output an empty line if no items should be selected."
)


def _format_pool(pool: list[PoolItem]) -> str:
    return "\n".join(
        f"[{p.id}] (tokens={p.token_estimate}) {p.title} :: {p.body_excerpt}"
        for p in pool
    )


def _format_state(state: StudentState) -> str:
    parts = [f"id={state.student_id}"]
    if state.mastery:
        parts.append("mastery=" + ",".join(f"{k}:{v:.2f}" for k, v in state.mastery.items()))
    if state.active_misconceptions:
        parts.append("misconceptions=" + " | ".join(state.active_misconceptions))
    return "\n".join(parts)


def format_router_input(
    *,
    student_state: StudentState,
    task_type: TaskType,
    task_text: str,
    budget: int,
    candidate_pool: list[PoolItem],
) -> str:
    return (
        f"{ROUTER_INSTRUCTION}\n\n"
        f"STUDENT:\n{_format_state(student_state)}\n\n"
        f"TASK [{task_type.value}] (budget={budget}): {task_text}\n\n"
        f"POOL:\n{_format_pool(candidate_pool)}\n\n"
        f"SELECTED IDS:"
    )


_ID_RE = re.compile(r"[a-f0-9]{8}")


def parse_router_output(text: str) -> list[str]:
    """Extract ordered short ids from router output. Tolerates whitespace, brackets, JSON."""
    if not text or not text.strip():
        return []
    if text.strip().lower() in {"none", "[]", "{}", "null"}:
        return []
    return _ID_RE.findall(text.lower())
```

- [ ] **Step 3: Run prompt test**

`uv run pytest tests/unit/test_router_prompt.py -v`
Expected: PASS.

- [ ] **Step 4: Write failing serializer test**

Create `tests/unit/test_trajectory_serializer.py`:
```python
from learning_memory_os.trajectories.schemas import (
    StudentState, PoolItem, TaskType, Trajectory,
)
from learning_memory_os.trajectories.serializer import trajectory_to_training_pair


def test_trajectory_to_training_pair_produces_input_and_target():
    t = Trajectory(
        id="t1",
        student_state=StudentState(student_id="s", mastery={}, active_misconceptions=[], recent_episodic_ids=[]),
        task_type=TaskType.EXPLAIN,
        task_text="explain A",
        budget=300,
        candidate_pool=[
            PoolItem(id="aaaa1111", title="A", body_excerpt="x", token_estimate=100),
            PoolItem(id="bbbb2222", title="B", body_excerpt="y", token_estimate=100),
        ],
        oracle_selection=["aaaa1111"],
    )
    pair = trajectory_to_training_pair(t)
    assert "aaaa1111" in pair["input"]
    assert pair["target"] == "aaaa1111"


def test_trajectory_to_training_pair_handles_empty_selection():
    t = Trajectory(
        id="t2",
        student_state=StudentState(student_id="s", mastery={}, active_misconceptions=[], recent_episodic_ids=[]),
        task_type=TaskType.QUIZ,
        task_text="quiz",
        budget=300,
        candidate_pool=[PoolItem(id="aaaa1111", title="A", body_excerpt="x", token_estimate=100)],
        oracle_selection=[],
    )
    pair = trajectory_to_training_pair(t)
    assert pair["target"] == ""
```

- [ ] **Step 5: Implement serializer**

Create `src/learning_memory_os/trajectories/serializer.py`:
```python
from .schemas import Trajectory
from ..router.prompt import format_router_input


def trajectory_to_training_pair(t: Trajectory) -> dict:
    """Render a trajectory as a single (input, target) text pair for SFT."""
    input_text = format_router_input(
        student_state=t.student_state,
        task_type=t.task_type,
        task_text=t.task_text,
        budget=t.budget,
        candidate_pool=t.candidate_pool,
    )
    target = ",".join(t.oracle_selection)
    return {"input": input_text, "target": target}
```

- [ ] **Step 6: Run serializer test**

`uv run pytest tests/unit/test_trajectory_serializer.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/learning_memory_os/router tests/unit/test_router_prompt.py
git add src/learning_memory_os/trajectories/serializer.py tests/unit/test_trajectory_serializer.py
git commit -m "feat(router): input/output prompt format + trajectory serializer"
```

---

## Task 5: Bulk trajectory generation CLI

**Files:**
- Create: `scripts/generate_trajectories.py`
- Test: `tests/integration/test_generate_trajectories.py`

- [ ] **Step 1: Write integration smoke test**

Create `tests/integration/test_generate_trajectories.py`:
```python
import subprocess, sys


def test_generate_trajectories_help():
    r = subprocess.run(
        [sys.executable, "-m", "scripts.generate_trajectories", "--help"],
        capture_output=True, text=True,
        env={"PYTHONPATH": "src", **__import__("os").environ},
    )
    assert r.returncode == 0
    assert "trajectories" in r.stdout.lower() or "generate" in r.stdout.lower()
```

- [ ] **Step 2: Implement the CLI**

Create `scripts/generate_trajectories.py`:
```python
"""Generate N synthetic tutoring trajectories and write to JSONL.

Run with --target 5000 first to validate, then scale to --target 50000.
"""

import json
import random
import uuid
from pathlib import Path
import typer

from learning_memory_os.config import get_settings
from learning_memory_os.llm import LLM
from learning_memory_os.memory.store import connect
from learning_memory_os.trajectories.sampler import sample_candidate_pool, sample_student_state
from learning_memory_os.trajectories.generator import build_trajectory
from learning_memory_os.trajectories.schemas import TaskType
from learning_memory_os.ingestion.topic_loader import load_topics


app = typer.Typer()


TASK_TEMPLATES = {
    TaskType.EXPLAIN: [
        "Explain the core idea of {title}.",
        "Why does {title} exist? What problem does it solve?",
        "Walk me through how {title} works step by step.",
        "What is the most common misconception about {title}?",
    ],
    TaskType.QUIZ: [
        "Generate a 3-question quiz on {title}.",
        "Quiz me on the tradeoffs in {title}.",
    ],
    TaskType.REVIEW: [
        "Summarize what I should remember about {title} for review.",
        "Give me a one-page review sheet on {title}.",
    ],
    TaskType.LAB: [
        "Suggest a hands-on lab to deepen mastery of {title}.",
    ],
}


@app.command()
def main(
    target: int = typer.Option(5000, "--target"),
    out: Path = typer.Option(Path("data/trajectories/main.jsonl"), "--out"),
    config: Path = typer.Option(Path("config/topics.yaml"), "--config"),
    oracle_model: str = typer.Option("claude-sonnet-4-6", "--oracle-model"),
    seed: int = typer.Option(42, "--seed"),
):
    random.seed(seed)
    settings = get_settings()
    topics = [t for t in load_topics(config)]  # will be filtered to those with content

    out.parent.mkdir(parents=True, exist_ok=True)

    llm = LLM(api_key=settings.anthropic_api_key, model=oracle_model)
    conn = connect(settings.database_url)

    # Only sample from topics that have content in the DB
    populated_topics = []
    with conn.cursor() as cur:
        for t in topics:
            cur.execute("SELECT count(*) AS n FROM semantic_items WHERE topic_id = %s", (t.id,))
            n = cur.fetchone()["n"]
            if n > 0:
                populated_topics.append((t, n))
    if not populated_topics:
        typer.echo("No populated topics in DB. Run Plan 2 first.", err=True)
        raise typer.Exit(2)

    written = 0
    try:
        with out.open("w") as f:
            for i in range(target):
                topic, _n = random.choice(populated_topics)
                task_type = random.choice(list(TaskType))
                template = random.choice(TASK_TEMPLATES[task_type])
                task_text = template.format(title=topic.title)
                pool = sample_candidate_pool(conn, target_topic=topic.id, pool_size=15)
                if not pool:
                    continue
                state = sample_student_state(
                    conn,
                    student_id=f"synthetic-{uuid.uuid4().hex[:8]}",
                    target_concepts=[topic.id],
                )
                try:
                    traj = build_trajectory(
                        traj_id=f"traj-{written:06d}",
                        student_state=state,
                        task_type=task_type,
                        task_text=task_text,
                        budget=random.choice([2000, 3000, 4000, 6000]),
                        candidate_pool=pool,
                        oracle_llm=llm,
                    )
                    f.write(json.dumps(traj.model_dump(), default=str) + "\n")
                    written += 1
                    if written % 50 == 0:
                        typer.echo(f"  wrote {written}/{target}")
                except Exception as e:
                    typer.echo(f"  [warn] trajectory {i} failed: {e}", err=True)
    finally:
        conn.close()

    typer.echo(f"\nDone. {written} trajectories written to {out}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 3: Run smoke test**

`uv run pytest tests/integration/test_generate_trajectories.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/generate_trajectories.py tests/integration/test_generate_trajectories.py
git commit -m "feat(trajectories): bulk generation CLI with --target sweep"
```

---

## Task 6: Generate 5K trajectories (validation), then 50K

This is a runbook — no new code. **Cost: ~$5 for 5K, ~$50 for 50K. Wall time: ~1-3 hours for 50K depending on API rate limits.**

- [ ] **Step 1: Validation run (5K trajectories)**

```bash
uv run python -m scripts.generate_trajectories --target 5000 --out data/trajectories/val.jsonl --seed 7
```

After completion:
```bash
wc -l data/trajectories/val.jsonl
head -1 data/trajectories/val.jsonl | python -c "import sys, json; d=json.loads(sys.stdin.read()); print('id:', d['id']); print('selected:', d['oracle_selection']); print('pool size:', len(d['candidate_pool']))"
```

Confirm: at least 4500 trajectories written, each with at least one selected id (oracle_selection nonempty). If <80% of attempts succeed, report concerns before scaling.

- [ ] **Step 2: Full run (50K trajectories)**

```bash
uv run python -m scripts.generate_trajectories --target 50000 --out data/trajectories/main.jsonl --seed 42
```

Expect 1-3 hours. Trajectories are written incrementally — if interrupted, you have partial output. Resumability is a stretch; for MVP just accept that interruption means rerun.

Capture final line count.

- [ ] **Step 3: Manifest**

Create a tiny `data/trajectories/manifest.json`:
```bash
python -c "
import json, hashlib
from pathlib import Path
for name in ['val.jsonl', 'main.jsonl']:
    p = Path('data/trajectories') / name
    if p.exists():
        size = p.stat().st_size
        n = sum(1 for _ in p.open())
        print(f'{name}: {n} lines, {size} bytes')
"
```

Write the output to `data/trajectories/manifest.txt`. Commit only the manifest (NOT the .jsonl files — they're large; gitignore them).

Add to `.gitignore`:
```
data/trajectories/*.jsonl
data/router_checkpoints/
```

Commit:
```bash
git add .gitignore data/trajectories/manifest.txt
git commit -m "data: trajectory manifest (val=5K, main=50K)"
```

---

## Task 7: Fine-tune infrastructure

**Files:**
- Add deps via uv
- Create: `config/router_sizes.yaml`
- Create: `src/learning_memory_os/router/finetune.py`
- Create: `scripts/finetune_router.py`
- Test: `tests/unit/test_router_sizes_config.py`

- [ ] **Step 1: Add ML deps**

```bash
uv add torch transformers peft accelerate datasets bitsandbytes
```

If `bitsandbytes` fails on macOS, skip it — needed only for 4-bit quant on the 7B model. Note that the deps install may be slow (PyTorch is large).

- [ ] **Step 2: Add the router sizes config**

Create `config/router_sizes.yaml`:
```yaml
version: 1
sizes:
  - id: qwen2_5_0_5b
    hf_model: Qwen/Qwen2.5-0.5B-Instruct
    lora_r: 16
    lora_alpha: 32
    batch_size: 8
    max_seq_len: 4096
    use_4bit_base: false
  - id: qwen2_5_1_5b
    hf_model: Qwen/Qwen2.5-1.5B-Instruct
    lora_r: 16
    lora_alpha: 32
    batch_size: 4
    max_seq_len: 4096
    use_4bit_base: false
  - id: qwen2_5_3b
    hf_model: Qwen/Qwen2.5-3B-Instruct
    lora_r: 16
    lora_alpha: 32
    batch_size: 2
    max_seq_len: 4096
    use_4bit_base: false
  - id: qwen2_5_7b
    hf_model: Qwen/Qwen2.5-7B-Instruct
    lora_r: 16
    lora_alpha: 32
    batch_size: 1
    max_seq_len: 4096
    use_4bit_base: true
```

- [ ] **Step 3: Write config-loader test**

Create `tests/unit/test_router_sizes_config.py`:
```python
from pathlib import Path
import yaml


def test_router_sizes_lists_4_sizes():
    cfg = yaml.safe_load(Path("config/router_sizes.yaml").read_text())
    assert len(cfg["sizes"]) == 4
    ids = [s["id"] for s in cfg["sizes"]]
    assert ids == ["qwen2_5_0_5b", "qwen2_5_1_5b", "qwen2_5_3b", "qwen2_5_7b"]
```

Run: `uv run pytest tests/unit/test_router_sizes_config.py -v`. PASS.

- [ ] **Step 4: Implement the LoRA training entry point**

Create `src/learning_memory_os/router/finetune.py`:
```python
"""LoRA fine-tuning of a HF model for context-routing.

Reads JSONL trajectories, serializes to input/target pairs, runs SFT with PEFT/LoRA,
saves a LoRA adapter directory.
"""

import json
from dataclasses import dataclass
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

from ..trajectories.schemas import Trajectory
from ..trajectories.serializer import trajectory_to_training_pair


@dataclass
class RouterFineTuneConfig:
    hf_model: str
    lora_r: int
    lora_alpha: int
    batch_size: int
    max_seq_len: int
    use_4bit_base: bool
    epochs: int = 2
    lr: float = 2e-4


def _load_pairs(jsonl_path: Path) -> list[dict]:
    pairs = []
    with jsonl_path.open() as f:
        for line in f:
            t = Trajectory.model_validate_json(line)
            pairs.append(trajectory_to_training_pair(t))
    return pairs


def _format_for_sft(pair: dict, eos_token: str) -> dict:
    """Concatenate input + target with a separator the model can learn."""
    return {"text": f"{pair['input']}\n{pair['target']}{eos_token}"}


def finetune(
    cfg: RouterFineTuneConfig,
    *,
    trajectories_path: Path,
    output_dir: Path,
    seed: int = 42,
) -> Path:
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"torch_dtype": torch.bfloat16}
    if cfg.use_4bit_base:
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        except ImportError:
            pass

    base = AutoModelForCausalLM.from_pretrained(cfg.hf_model, **load_kwargs)

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(base, lora_cfg)

    pairs = _load_pairs(trajectories_path)
    raw = [_format_for_sft(p, tokenizer.eos_token) for p in pairs]
    ds = Dataset.from_list(raw)

    def tokenize(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=cfg.max_seq_len,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])

    output_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        report_to="none",
        seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "adapter")
    return output_dir / "adapter"
```

- [ ] **Step 5: Implement the finetune CLI**

Create `scripts/finetune_router.py`:
```python
"""Launch a single fine-tune by size id (from config/router_sizes.yaml)."""

from pathlib import Path
import yaml
import typer

from learning_memory_os.router.finetune import (
    RouterFineTuneConfig, finetune,
)


app = typer.Typer()


@app.command()
def main(
    size_id: str = typer.Option(..., "--size"),
    trajectories: Path = typer.Option(Path("data/trajectories/main.jsonl"), "--trajectories"),
    out: Path = typer.Option(Path("data/router_checkpoints"), "--out"),
    epochs: int = typer.Option(2, "--epochs"),
):
    cfg_all = yaml.safe_load(Path("config/router_sizes.yaml").read_text())
    sizes = {s["id"]: s for s in cfg_all["sizes"]}
    if size_id not in sizes:
        typer.echo(f"Unknown size: {size_id}", err=True)
        raise typer.Exit(2)
    s = sizes[size_id]
    cfg = RouterFineTuneConfig(
        hf_model=s["hf_model"],
        lora_r=s["lora_r"],
        lora_alpha=s["lora_alpha"],
        batch_size=s["batch_size"],
        max_seq_len=s["max_seq_len"],
        use_4bit_base=s["use_4bit_base"],
        epochs=epochs,
    )
    output_dir = out / size_id
    adapter_path = finetune(cfg, trajectories_path=trajectories, output_dir=output_dir)
    typer.echo(f"Done. Adapter at: {adapter_path}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 6: Commit**

```bash
git add config/router_sizes.yaml src/learning_memory_os/router/finetune.py scripts/finetune_router.py tests/unit/test_router_sizes_config.py
git commit -m "feat(router): LoRA fine-tune pipeline + size config"
```

---

## Task 8: Fine-tune the routers

This is a runbook. **Each size will take wall time depending on GPU.** Estimate (on a single H100, val set of 5K trajectories):
- 0.5B: ~15 min
- 1.5B: ~30 min
- 3B: ~1 hour
- 7B: ~2-3 hours (with 4-bit base)

- [ ] **Step 1: Smoke-test on the smallest model with the val set**

```bash
uv run python -m scripts.finetune_router --size qwen2_5_0_5b --trajectories data/trajectories/val.jsonl --epochs 1
```

This validates the whole pipeline end-to-end without burning hours. If it fails, fix before proceeding.

- [ ] **Step 2: Train each size on the full main.jsonl**

Run each (sequentially, since they likely all need the same GPU):

```bash
uv run python -m scripts.finetune_router --size qwen2_5_0_5b --trajectories data/trajectories/main.jsonl
uv run python -m scripts.finetune_router --size qwen2_5_1_5b --trajectories data/trajectories/main.jsonl
uv run python -m scripts.finetune_router --size qwen2_5_3b   --trajectories data/trajectories/main.jsonl
uv run python -m scripts.finetune_router --size qwen2_5_7b   --trajectories data/trajectories/main.jsonl
```

If 7B OOMs even with 4-bit base, cut it from the sweep — the Pareto plot survives with 3 sizes.

- [ ] **Step 3: Manifest of trained adapters**

```bash
ls -la data/router_checkpoints/*/adapter/adapter_config.json
```

Each line confirms one successful fine-tune. Capture which sizes succeeded.

---

## Task 9: Router inference + frontier-API baseline

**Files:**
- Create: `src/learning_memory_os/router/infer.py`
- Create: `src/learning_memory_os/router/frontier_api.py`
- Test: `tests/unit/test_router_infer.py`

- [ ] **Step 1: Implement adapter-based inference**

Create `src/learning_memory_os/router/infer.py`:
```python
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from ..trajectories.schemas import StudentState, PoolItem, TaskType
from .prompt import format_router_input, parse_router_output


class FineTunedRouter:
    def __init__(self, adapter_dir: Path, base_model: str):
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
        self.model = PeftModel.from_pretrained(base, adapter_dir)
        self.model.eval()

    @torch.no_grad()
    def route(
        self,
        *,
        student_state: StudentState,
        task_type: TaskType,
        task_text: str,
        budget: int,
        candidate_pool: list[PoolItem],
        max_new_tokens: int = 128,
    ) -> list[str]:
        prompt = format_router_input(
            student_state=student_state,
            task_type=task_type,
            task_text=task_text,
            budget=budget,
            candidate_pool=candidate_pool,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = out_ids[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return parse_router_output(text)
```

- [ ] **Step 2: Implement frontier-API baseline**

Create `src/learning_memory_os/router/frontier_api.py`:
```python
from ..llm import LLM
from ..trajectories.schemas import StudentState, PoolItem, TaskType
from .prompt import format_router_input, parse_router_output


class FrontierAPIRouter:
    """Zero-shot router that uses a frontier API as the selector (upper-bound baseline)."""

    def __init__(self, llm: LLM):
        self.llm = llm

    def route(
        self,
        *,
        student_state: StudentState,
        task_type: TaskType,
        task_text: str,
        budget: int,
        candidate_pool: list[PoolItem],
    ) -> list[str]:
        prompt = format_router_input(
            student_state=student_state,
            task_type=task_type,
            task_text=task_text,
            budget=budget,
            candidate_pool=candidate_pool,
        )
        text = self.llm.complete(system="You are a context selection router.", user=prompt, max_tokens=256)
        return parse_router_output(text)
```

- [ ] **Step 3: Quick unit test (mocked)**

Create `tests/unit/test_router_infer.py`:
```python
from unittest.mock import MagicMock
from learning_memory_os.trajectories.schemas import StudentState, PoolItem, TaskType
from learning_memory_os.router.frontier_api import FrontierAPIRouter


def test_frontier_api_router_uses_llm():
    fake_llm = MagicMock()
    fake_llm.complete.return_value = "aaaa1111,bbbb2222"
    r = FrontierAPIRouter(fake_llm)
    state = StudentState(student_id="s", mastery={}, active_misconceptions=[], recent_episodic_ids=[])
    out = r.route(
        student_state=state,
        task_type=TaskType.EXPLAIN,
        task_text="explain A",
        budget=300,
        candidate_pool=[
            PoolItem(id="aaaa1111", title="A", body_excerpt="x", token_estimate=100),
            PoolItem(id="bbbb2222", title="B", body_excerpt="y", token_estimate=100),
        ],
    )
    assert out == ["aaaa1111", "bbbb2222"]
```

Run: `uv run pytest tests/unit/test_router_infer.py -v`. PASS.

- [ ] **Step 4: Commit**

```bash
git add src/learning_memory_os/router/infer.py src/learning_memory_os/router/frontier_api.py tests/unit/test_router_infer.py
git commit -m "feat(router): adapter inference + frontier-API baseline"
```

---

## Task 10: Evaluation harness + Pareto plot

**Files:**
- Create: `src/learning_memory_os/eval/__init__.py`
- Create: `src/learning_memory_os/eval/router_eval.py`
- Create: `src/learning_memory_os/eval/pareto.py`
- Create: `scripts/eval_routers.py`
- Create: `scripts/plot_pareto.py`
- Test: `tests/unit/test_router_eval.py`

- [ ] **Step 1: Implement metrics**

Create `src/learning_memory_os/eval/__init__.py` (empty).

Create `src/learning_memory_os/eval/router_eval.py`:
```python
from dataclasses import dataclass


@dataclass
class RouterMetrics:
    precision: float
    recall: float
    jaccard: float
    n: int


def _binary_metrics(pred: list[str], gold: list[str]) -> tuple[float, float, float]:
    sp, sg = set(pred), set(gold)
    tp = len(sp & sg)
    fp = len(sp - sg)
    fn = len(sg - sp)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    union = sp | sg
    jaccard = tp / len(union) if union else 1.0
    return precision, recall, jaccard


def evaluate(predictions: list[list[str]], gold: list[list[str]]) -> RouterMetrics:
    assert len(predictions) == len(gold)
    if not predictions:
        return RouterMetrics(precision=0.0, recall=0.0, jaccard=0.0, n=0)
    ps, rs, js = [], [], []
    for p, g in zip(predictions, gold):
        pp, rr, jj = _binary_metrics(p, g)
        ps.append(pp); rs.append(rr); js.append(jj)
    return RouterMetrics(
        precision=sum(ps) / len(ps),
        recall=sum(rs) / len(rs),
        jaccard=sum(js) / len(js),
        n=len(predictions),
    )
```

- [ ] **Step 2: Test metrics**

Create `tests/unit/test_router_eval.py`:
```python
from learning_memory_os.eval.router_eval import evaluate


def test_perfect_match():
    m = evaluate([["a", "b"]], [["a", "b"]])
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.jaccard == 1.0


def test_no_overlap():
    m = evaluate([["a"]], [["b"]])
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.jaccard == 0.0


def test_partial_overlap():
    m = evaluate([["a", "b", "c"]], [["a", "b", "d"]])
    # tp=2, fp=1, fn=1 -> precision=2/3, recall=2/3, jaccard=2/4=0.5
    assert abs(m.precision - 2 / 3) < 1e-6
    assert abs(m.recall - 2 / 3) < 1e-6
    assert m.jaccard == 0.5
```

Run: `uv run pytest tests/unit/test_router_eval.py -v`. PASS.

- [ ] **Step 3: Eval CLI**

Create `scripts/eval_routers.py`:
```python
"""Evaluate every available router on a held-out trajectory split.

Outputs data/eval/router_results.json with rows: {router_id, precision, recall, jaccard, cost_per_call_ms, n}.
"""

import json
import time
from pathlib import Path
import typer
import yaml

from learning_memory_os.config import get_settings
from learning_memory_os.llm import LLM
from learning_memory_os.trajectories.schemas import Trajectory
from learning_memory_os.eval.router_eval import evaluate


app = typer.Typer()


def _load_test_trajectories(path: Path, limit: int) -> list[Trajectory]:
    items: list[Trajectory] = []
    with path.open() as f:
        for line in f:
            items.append(Trajectory.model_validate_json(line))
            if len(items) >= limit:
                break
    return items


@app.command()
def main(
    test_file: Path = typer.Option(Path("data/trajectories/val.jsonl"), "--test"),
    test_limit: int = typer.Option(500, "--limit"),
    out: Path = typer.Option(Path("data/eval/router_results.json"), "--out"),
):
    out.parent.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    test = _load_test_trajectories(test_file, test_limit)
    typer.echo(f"Loaded {len(test)} test trajectories.")

    results: list[dict] = []

    # 1) Frontier-API baseline
    from learning_memory_os.router.frontier_api import FrontierAPIRouter
    fr_llm = LLM(api_key=settings.anthropic_api_key, model="claude-sonnet-4-6")
    fr = FrontierAPIRouter(fr_llm)
    preds: list[list[str]] = []
    gold: list[list[str]] = []
    t0 = time.time()
    for t in test:
        p = fr.route(
            student_state=t.student_state,
            task_type=t.task_type,
            task_text=t.task_text,
            budget=t.budget,
            candidate_pool=t.candidate_pool,
        )
        preds.append(p)
        gold.append(t.oracle_selection)
    elapsed = time.time() - t0
    metrics = evaluate(preds, gold)
    results.append({
        "router_id": "frontier_api_sonnet",
        "precision": metrics.precision,
        "recall": metrics.recall,
        "jaccard": metrics.jaccard,
        "n": metrics.n,
        "ms_per_call": elapsed / max(1, metrics.n) * 1000,
    })

    # 2) Fine-tuned routers
    sizes = yaml.safe_load(Path("config/router_sizes.yaml").read_text())["sizes"]
    for s in sizes:
        adapter = Path("data/router_checkpoints") / s["id"] / "adapter"
        if not adapter.exists():
            typer.echo(f"[skip] {s['id']}: no adapter at {adapter}")
            continue
        from learning_memory_os.router.infer import FineTunedRouter
        try:
            r = FineTunedRouter(adapter_dir=adapter, base_model=s["hf_model"])
        except Exception as e:
            typer.echo(f"[skip] {s['id']}: load failed: {e}")
            continue
        preds = []
        t0 = time.time()
        for t in test:
            p = r.route(
                student_state=t.student_state,
                task_type=t.task_type,
                task_text=t.task_text,
                budget=t.budget,
                candidate_pool=t.candidate_pool,
            )
            preds.append(p)
        elapsed = time.time() - t0
        m = evaluate(preds, gold)
        results.append({
            "router_id": s["id"],
            "precision": m.precision,
            "recall": m.recall,
            "jaccard": m.jaccard,
            "n": m.n,
            "ms_per_call": elapsed / max(1, m.n) * 1000,
        })

    out.write_text(json.dumps(results, indent=2))
    typer.echo(f"Wrote {len(results)} rows to {out}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Pareto plot**

Create `src/learning_memory_os/eval/pareto.py`:
```python
import json
import matplotlib.pyplot as plt
from pathlib import Path


def plot_pareto(results_json: Path, out_png: Path) -> None:
    data = json.loads(Path(results_json).read_text())
    xs = [d["ms_per_call"] for d in data]
    ys = [d["jaccard"] for d in data]
    labels = [d["router_id"] for d in data]

    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, s=80)
    for x, y, l in zip(xs, ys, labels):
        plt.annotate(l, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    plt.xscale("log")
    plt.xlabel("Latency (ms / call, log scale)")
    plt.ylabel("Selection Jaccard vs oracle")
    plt.title("Learning Memory OS — Router Accuracy vs Cost")
    plt.grid(True, alpha=0.3)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
```

Create `scripts/plot_pareto.py`:
```python
from pathlib import Path
import typer
from learning_memory_os.eval.pareto import plot_pareto

app = typer.Typer()


@app.command()
def main(
    results: Path = typer.Option(Path("data/eval/router_results.json"), "--results"),
    out: Path = typer.Option(Path("data/eval/pareto.png"), "--out"),
):
    plot_pareto(results, out)
    typer.echo(f"Wrote {out}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 5: Add matplotlib**

```bash
uv add matplotlib
```

- [ ] **Step 6: Run the eval + plot**

```bash
uv run python -m scripts.eval_routers --test data/trajectories/val.jsonl --limit 500
uv run python -m scripts.plot_pareto
```

This runs the held-out evaluation (500 trajectories from val.jsonl that were NOT used in training — note that val.jsonl wasn't the training set; main.jsonl was). For the cleanest experiment, split main.jsonl into train/test 90/10. For Plan 3 MVP, evaluating on val.jsonl is acceptable since the seeds are different.

- [ ] **Step 7: Commit**

```bash
git add src/learning_memory_os/eval scripts/eval_routers.py scripts/plot_pareto.py tests/unit/test_router_eval.py
git add data/eval/router_results.json data/eval/pareto.png
git commit -m "feat(eval): router evaluation + accuracy-vs-cost pareto plot"
```

---

## Task 11: Plan 3 summary + tag

- [ ] **Step 1: Full test suite**

```bash
uv run pytest -v
```

PASS.

- [ ] **Step 2: Ruff**

```bash
uv run ruff check src tests scripts
```

Fix + commit if needed.

- [ ] **Step 3: Write summary**

Create `data/plan-3-summary.md`:
```markdown
# Plan 3 — Router Fine-Tuning Complete

**Tag:** `routers-trained`
**Branch:** `plan-1-mvp` (consider merging to main after)
**Date:** <fill in>

## What shipped
- Trajectory generator: 50K synthetic tutoring trajectories (oracle = Claude Sonnet)
- LoRA fine-tunes at sizes: <list of completed adapters>
- Frontier-API baseline (Sonnet zero-shot router)
- Router evaluation harness: precision / recall / Jaccard vs oracle
- Pareto plot: data/eval/pareto.png

## Headline result
<paste the eval table from data/eval/router_results.json — fill in after the run>

## Limitations
- Trajectories are synthetic; oracle = LLM, so training is effectively distillation. Real-trajectory training is post-CS153 work.
- Held-out test split is from a different random seed but same generation distribution; no domain-shift evaluation.
- 7B router only trained if hardware supported QLoRA; otherwise sweep is 3 sizes.

## What's next
- Real user trajectories from student-zero use + cohort (if any)
- Combinatorial selector ablation (deferred from the design spec)
- Writeup + demo video
```

- [ ] **Step 4: Tag**

```bash
git tag -a routers-trained -m "Plan 3 complete: 50K trajectories + LoRA routers + pareto frontier"
git add data/plan-3-summary.md
git commit -m "docs: plan 3 final summary"
```

---

## Self-review notes

- **Spec coverage**: Plan 3 implements §2.3 Phase 3 (learned routers), §4.2 (synthetic trajectory generation), §4.4 (size × strategy ablation), §4.5 (Pareto frontier). It does NOT touch §2.3 Phase 2 combinatorial selector (deferred per cut policy).
- **No placeholders**: every step has actual code, every CLI has its actual flags, every test has assertions.
- **Type consistency**: `Trajectory`, `StudentState`, `PoolItem`, `TaskType` from Task 1 used in Tasks 3–10. `format_router_input` / `parse_router_output` from Task 4 used in Tasks 9–10. `RouterFineTuneConfig` from Task 7 used in Task 8.
- **Hardware contingency**: explicit pre-flight + graceful fallback if 7B doesn't fit. Pareto plot still publishable with 3 sizes.
- **Cost contingency**: validation run (5K) before scaling to 50K. If quality is low, no money wasted on a bad pipeline.
- **Known limitations preserved**: synthetic-only training = distillation; this is acknowledged in the design spec §1 and reiterated in the Plan 3 summary template.

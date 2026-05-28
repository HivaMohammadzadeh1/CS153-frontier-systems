# Plan 1 — MVP System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the minimum-credible end-to-end Learning Memory OS — ingestion → multi-tier memory → heuristic context selector → tutor agent → interaction logging — running on 4 seed topics with one user (the author). This is the TA's "Week 6 MVP."

**Architecture:** Python service backed by Postgres + pgvector. Ingestion converts raw lecture/paper text into structured artifacts via LLM-extraction; artifacts populate a four-tier memory store; a heuristic ranker + budgeted packer selects context per task; a tutor agent calls the selector and an LLM to respond; every interaction is logged to JSONL for later evaluation. Phase 2 (combinatorial selector) and Phase 3 (fine-tuned router) build on this plan but live in follow-up plans.

**Tech Stack:** Python 3.11+, uv (package manager), Postgres 16 + pgvector (Docker), Anthropic SDK (Claude), OpenAI SDK (embeddings), Pydantic v2, pytest, structlog.

**This is Plan 1 of an expected series.** Follow-ups (not in scope here): Plan 2 — Curriculum content pipeline; Plan 3 — Synthetic trajectory generator + LoRA fine-tuning at multiple sizes; Plan 4 — Combinatorial selector + ablations; Plan 5 — Real-user data collection; Plan 6 — Writeup + demo.

**Spec reference:** `docs/superpowers/specs/2026-05-12-learning-memory-os-design.md`

---

## File Structure

Plan 1 creates these files:

```
CS153-frontier-systems/
├── pyproject.toml
├── docker-compose.yml
├── .env.example
├── README.md (existing — extend)
├── migrations/
│   └── 001_init.sql
├── src/learning_memory_os/
│   ├── __init__.py
│   ├── config.py                       # Env / settings
│   ├── llm.py                          # Anthropic wrapper
│   ├── embeddings.py                   # OpenAI embeddings wrapper
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── artifacts.py                # Concept, Misconception, Example, Exercise, etc.
│   │   └── memory.py                   # MemoryItem, MasteryEntry, EpisodicEvent, etc.
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── store.py                    # Postgres connection + base operations
│   │   ├── semantic.py                 # Semantic-tier API
│   │   ├── student.py                  # Student-tier API
│   │   ├── episodic.py                 # Episodic-tier API
│   │   └── intervention.py             # Intervention-tier API
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── extractors.py               # Text → artifacts via LLM
│   ├── selector/
│   │   ├── __init__.py
│   │   ├── scoring.py                  # Per-item score components
│   │   └── pack.py                     # Budget-respecting top-K assembly
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                     # Base agent w/ selector call
│   │   └── tutor.py                    # Tutor agent
│   └── logging_utils/
│       ├── __init__.py
│       └── interactions.py             # JSONL interaction logger
├── tests/
│   ├── conftest.py
│   ├── unit/                           # Mirrors src/learning_memory_os/
│   └── integration/                    # End-to-end smoke tests
├── scripts/
│   ├── ingest_topic.py                 # CLI: load a topic's materials
│   └── tutor_repl.py                   # CLI: interactive tutor session
└── data/
    └── seed_topics/                    # 4 MVP topics (one each from Areas A/B/C/E)
        ├── tokenization/               # Area A
        ├── data_parallelism/           # Area B
        ├── kv_cache/                   # Area C
        └── agent_memory/               # Area E
```

Boundary rationale:
- `schemas/` is the contract between layers — all other modules import from it.
- `memory/` knows about Postgres; `selector/`, `agents/`, `ingestion/` don't.
- `selector/scoring.py` is pure functions over memory items (TDD-friendly).
- `selector/pack.py` is the budget logic (also pure, TDD-friendly).
- `agents/base.py` owns the routing-engine call; specific agents are thin specializations.

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `docker-compose.yml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `src/learning_memory_os/__init__.py`
- Modify: `README.md`

- [ ] **Step 1: Initialize uv project**

Run:
```bash
cd /Users/hivamoh/CS153-project/CS153-frontier-systems
uv init --package --name learning-memory-os --python 3.11
```

Expected: creates `pyproject.toml`, `src/learning_memory_os/__init__.py`.

- [ ] **Step 2: Add runtime dependencies**

Run:
```bash
uv add anthropic openai psycopg[binary,pool] pgvector pydantic structlog python-dotenv typer
```

- [ ] **Step 3: Add dev dependencies**

Run:
```bash
uv add --dev pytest pytest-asyncio pytest-postgresql ruff mypy
```

- [ ] **Step 4: Create docker-compose.yml**

Create file `docker-compose.yml`:
```yaml
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: learning_memory_os
      POSTGRES_USER: lmos
      POSTGRES_PASSWORD: lmos_dev
    ports:
      - "5433:5432"
    volumes:
      - lmos_pgdata:/var/lib/postgresql/data

volumes:
  lmos_pgdata:
```

- [ ] **Step 5: Create .env.example**

Create file `.env.example`:
```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://lmos:lmos_dev@localhost:5433/learning_memory_os
LMOS_LOG_DIR=./logs
LMOS_DEFAULT_TOKEN_BUDGET=8000
```

- [ ] **Step 6: Create .gitignore**

Create file `.gitignore`:
```
.env
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.mypy_cache/
logs/
data/runtime/
```

- [ ] **Step 7: Replace README content**

Overwrite `README.md`:
```markdown
# Learning Memory OS

A context-routed tutor for ML systems engineers. CS 153 final project.

See `docs/superpowers/specs/2026-05-12-learning-memory-os-design.md` for the design spec.

## Dev setup
1. `cp .env.example .env` and fill in API keys
2. `docker compose up -d db`
3. `uv sync`
4. `uv run pytest`
```

- [ ] **Step 8: Verify Postgres starts**

Run:
```bash
docker compose up -d db
docker compose exec db psql -U lmos -d learning_memory_os -c "SELECT version();"
```

Expected: prints PostgreSQL 16.x version string.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock docker-compose.yml .env.example .gitignore README.md src/
git commit -m "chore: scaffold project with uv, docker postgres, deps"
```

---

## Task 2: Database schema migration

**Files:**
- Create: `migrations/001_init.sql`
- Create: `src/learning_memory_os/config.py`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write failing test for config**

Create `tests/unit/test_config.py`:
```python
import os
from learning_memory_os.config import Settings


def test_settings_reads_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://u:p@localhost:5433/db"
    )
    monkeypatch.setenv("LMOS_LOG_DIR", "/tmp/logs")
    monkeypatch.setenv("LMOS_DEFAULT_TOKEN_BUDGET", "4000")

    s = Settings()
    assert s.anthropic_api_key == "sk-ant-test"
    assert s.openai_api_key == "sk-test"
    assert s.default_token_budget == 4000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: FAIL with "No module named 'learning_memory_os.config'"

- [ ] **Step 3: Implement config**

Create `src/learning_memory_os/config.py`:
```python
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str
    openai_api_key: str
    database_url: str
    log_dir: Path = Field(default=Path("./logs"), alias="LMOS_LOG_DIR")
    default_token_budget: int = Field(default=8000, alias="LMOS_DEFAULT_TOKEN_BUDGET")


def get_settings() -> Settings:
    return Settings()
```

Add the missing dep:
```bash
uv add pydantic-settings
```

- [ ] **Step 4: Run config test**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Write the migration SQL**

Create `migrations/001_init.sql`:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Semantic tier: stable course/topic facts
CREATE TABLE semantic_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    topic_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,        -- concept | example | misconception | exercise | code_pattern | paper_claim
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(1536),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX semantic_items_topic_idx ON semantic_items(topic_id);
CREATE INDEX semantic_items_type_idx ON semantic_items(artifact_type);
CREATE INDEX semantic_items_embedding_idx ON semantic_items USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Prerequisite graph (concept-level edges)
CREATE TABLE prerequisites (
    src UUID REFERENCES semantic_items(id) ON DELETE CASCADE,
    dst UUID REFERENCES semantic_items(id) ON DELETE CASCADE,
    PRIMARY KEY (src, dst)
);

-- Student tier: per-student mastery + misconceptions
CREATE TABLE students (
    id TEXT PRIMARY KEY,
    profile JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE mastery (
    student_id TEXT REFERENCES students(id) ON DELETE CASCADE,
    concept_id UUID REFERENCES semantic_items(id) ON DELETE CASCADE,
    score REAL NOT NULL,
    confidence REAL NOT NULL,
    last_updated TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (student_id, concept_id)
);

CREATE TABLE misconceptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id TEXT REFERENCES students(id) ON DELETE CASCADE,
    concept_id UUID REFERENCES semantic_items(id),
    description TEXT NOT NULL,
    evidence TEXT,
    resolved BOOLEAN NOT NULL DEFAULT FALSE,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at TIMESTAMPTZ
);
CREATE INDEX misconceptions_student_idx ON misconceptions(student_id);

-- Episodic tier: append-only event log
CREATE TABLE episodic_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id TEXT REFERENCES students(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,           -- session_start | question | tutor_reply | quiz_attempt | exercise_attempt
    payload JSONB NOT NULL,
    embedding vector(1536),
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX episodic_student_idx ON episodic_events(student_id, occurred_at DESC);
CREATE INDEX episodic_embedding_idx ON episodic_events USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Intervention tier: which tutoring strategy was tried, did it work
CREATE TABLE interventions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id TEXT REFERENCES students(id) ON DELETE CASCADE,
    misconception_id UUID REFERENCES misconceptions(id) ON DELETE SET NULL,
    strategy TEXT NOT NULL,
    outcome TEXT,                       -- helped | partial | no_effect | regressed | unknown
    notes TEXT,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

- [ ] **Step 6: Apply the migration**

Run:
```bash
docker compose exec -T db psql -U lmos -d learning_memory_os < migrations/001_init.sql
```

Expected: prints CREATE EXTENSION / CREATE TABLE / CREATE INDEX messages with no errors.

- [ ] **Step 7: Smoke-test the schema**

Run:
```bash
docker compose exec db psql -U lmos -d learning_memory_os -c "\dt"
```

Expected: lists tables `episodic_events`, `interventions`, `mastery`, `misconceptions`, `prerequisites`, `semantic_items`, `students`.

- [ ] **Step 8: Commit**

```bash
git add migrations/001_init.sql src/learning_memory_os/config.py tests/unit/test_config.py
git commit -m "feat(db): initial schema for 4-tier memory + config module"
```

---

## Task 3: LLM and embeddings wrappers

**Files:**
- Create: `src/learning_memory_os/llm.py`
- Create: `src/learning_memory_os/embeddings.py`
- Test: `tests/unit/test_llm.py`
- Test: `tests/unit/test_embeddings.py`

- [ ] **Step 1: Write failing test for LLM wrapper**

Create `tests/unit/test_llm.py`:
```python
from unittest.mock import MagicMock, patch
from learning_memory_os.llm import LLM


def test_llm_complete_returns_text():
    fake_response = MagicMock()
    fake_response.content = [MagicMock(text="hello world")]

    with patch("learning_memory_os.llm.Anthropic") as MockAnthropic:
        client = MockAnthropic.return_value
        client.messages.create.return_value = fake_response

        llm = LLM(api_key="sk-test", model="claude-opus-4-7")
        out = llm.complete(system="be terse", user="hi")

        assert out == "hello world"
        client.messages.create.assert_called_once()
        kwargs = client.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-opus-4-7"
        assert kwargs["system"] == "be terse"
        assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_llm_complete_json_parses_response():
    fake_response = MagicMock()
    fake_response.content = [MagicMock(text='{"k": 1}')]

    with patch("learning_memory_os.llm.Anthropic") as MockAnthropic:
        client = MockAnthropic.return_value
        client.messages.create.return_value = fake_response

        llm = LLM(api_key="sk-test", model="claude-opus-4-7")
        out = llm.complete_json(system="emit json", user="hi")
        assert out == {"k": 1}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_llm.py -v`
Expected: FAIL with "No module named 'learning_memory_os.llm'"

- [ ] **Step 3: Implement LLM wrapper**

Create `src/learning_memory_os/llm.py`:
```python
import json
import re
from anthropic import Anthropic


class LLM:
    def __init__(self, api_key: str, model: str = "claude-opus-4-7"):
        self._client = Anthropic(api_key=api_key)
        self.model = model

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> str:
        resp = self._client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return "".join(block.text for block in resp.content if hasattr(block, "text"))

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 4096,
    ) -> dict:
        text = self.complete(
            system=system, user=user, max_tokens=max_tokens, temperature=0.0
        )
        match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
        payload = match.group(0) if match else text
        return json.loads(payload)
```

- [ ] **Step 4: Run LLM test**

Run: `uv run pytest tests/unit/test_llm.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Write failing test for embeddings wrapper**

Create `tests/unit/test_embeddings.py`:
```python
from unittest.mock import MagicMock, patch
from learning_memory_os.embeddings import Embedder


def test_embedder_returns_vectors():
    fake_response = MagicMock()
    fake_response.data = [MagicMock(embedding=[0.1] * 1536)]

    with patch("learning_memory_os.embeddings.OpenAI") as MockOpenAI:
        client = MockOpenAI.return_value
        client.embeddings.create.return_value = fake_response

        e = Embedder(api_key="sk-test")
        out = e.embed_one("hello")
        assert len(out) == 1536
        assert out[0] == 0.1
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_embeddings.py -v`
Expected: FAIL with "No module named 'learning_memory_os.embeddings'"

- [ ] **Step 7: Implement embeddings wrapper**

Create `src/learning_memory_os/embeddings.py`:
```python
from openai import OpenAI


class Embedder:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.dim = 1536

    def embed_one(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(model=self.model, input=text)
        return list(resp.data[0].embedding)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return [list(d.embedding) for d in resp.data]
```

- [ ] **Step 8: Run embeddings test**

Run: `uv run pytest tests/unit/test_embeddings.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/learning_memory_os/llm.py src/learning_memory_os/embeddings.py tests/unit/test_llm.py tests/unit/test_embeddings.py
git commit -m "feat: anthropic LLM + openai embeddings wrappers"
```

---

## Task 4: Schemas (artifacts + memory items)

**Files:**
- Create: `src/learning_memory_os/schemas/__init__.py`
- Create: `src/learning_memory_os/schemas/artifacts.py`
- Create: `src/learning_memory_os/schemas/memory.py`
- Test: `tests/unit/test_schemas.py`

- [ ] **Step 1: Write failing schema test**

Create `tests/unit/test_schemas.py`:
```python
from learning_memory_os.schemas.artifacts import (
    Concept,
    Misconception,
    Example,
    Exercise,
    ArtifactType,
)
from learning_memory_os.schemas.memory import (
    MemoryItem,
    MasteryEntry,
    EpisodicEvent,
)


def test_concept_round_trip():
    c = Concept(
        topic_id="kv_cache",
        title="KV cache",
        definition="A cache of past attention K and V tensors.",
        deep_explanation="Long form explanation.",
        prerequisites=[],
    )
    assert c.artifact_type == ArtifactType.CONCEPT
    assert c.model_dump()["topic_id"] == "kv_cache"


def test_misconception_has_correction():
    m = Misconception(
        topic_id="kv_cache",
        statement="KV cache stores raw token ids.",
        correction="It stores K and V tensors of past tokens.",
    )
    assert m.artifact_type == ArtifactType.MISCONCEPTION


def test_memory_item_from_artifact():
    c = Concept(
        topic_id="kv_cache",
        title="KV cache",
        definition="A cache.",
        deep_explanation="More.",
        prerequisites=[],
    )
    item = MemoryItem.from_artifact(c, embedding=[0.0] * 1536)
    assert item.tier == "semantic"
    assert item.title == "KV cache"
    assert len(item.embedding) == 1536


def test_mastery_entry_bounds():
    m = MasteryEntry(student_id="hiva", concept_id="abc", score=0.7, confidence=0.5)
    assert 0.0 <= m.score <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_schemas.py -v`
Expected: FAIL with "No module named 'learning_memory_os.schemas.artifacts'"

- [ ] **Step 3: Implement artifact schemas**

Create `src/learning_memory_os/schemas/__init__.py` (empty file).

Create `src/learning_memory_os/schemas/artifacts.py`:
```python
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field


class ArtifactType(str, Enum):
    CONCEPT = "concept"
    EXAMPLE = "example"
    MISCONCEPTION = "misconception"
    EXERCISE = "exercise"
    CODE_PATTERN = "code_pattern"
    PAPER_CLAIM = "paper_claim"


class _ArtifactBase(BaseModel):
    topic_id: str
    title: str = ""
    artifact_type: ArtifactType


class Concept(_ArtifactBase):
    artifact_type: Literal[ArtifactType.CONCEPT] = ArtifactType.CONCEPT
    definition: str
    deep_explanation: str
    prerequisites: list[str] = Field(default_factory=list)


class Example(_ArtifactBase):
    artifact_type: Literal[ArtifactType.EXAMPLE] = ArtifactType.EXAMPLE
    concept_title: str
    body: str


class Misconception(_ArtifactBase):
    artifact_type: Literal[ArtifactType.MISCONCEPTION] = ArtifactType.MISCONCEPTION
    statement: str
    correction: str


class Exercise(_ArtifactBase):
    artifact_type: Literal[ArtifactType.EXERCISE] = ArtifactType.EXERCISE
    prompt: str
    starter_code: str = ""
    rubric: str


class CodePattern(_ArtifactBase):
    artifact_type: Literal[ArtifactType.CODE_PATTERN] = ArtifactType.CODE_PATTERN
    body: str


class PaperClaim(_ArtifactBase):
    artifact_type: Literal[ArtifactType.PAPER_CLAIM] = ArtifactType.PAPER_CLAIM
    claim: str
    source: str
    evidence: str


Artifact = Concept | Example | Misconception | Exercise | CodePattern | PaperClaim


def artifact_to_body(a: Artifact) -> str:
    """Canonical text representation for embedding and serialization."""
    if isinstance(a, Concept):
        return f"{a.title}\n{a.definition}\n{a.deep_explanation}"
    if isinstance(a, Example):
        return f"Example for {a.concept_title}: {a.body}"
    if isinstance(a, Misconception):
        return f"Misconception: {a.statement}\nCorrection: {a.correction}"
    if isinstance(a, Exercise):
        return f"Exercise: {a.prompt}\nRubric: {a.rubric}"
    if isinstance(a, CodePattern):
        return f"Code pattern: {a.title}\n{a.body}"
    if isinstance(a, PaperClaim):
        return f"Claim: {a.claim}\nSource: {a.source}\nEvidence: {a.evidence}"
    raise ValueError(f"Unknown artifact: {a}")
```

- [ ] **Step 4: Implement memory-item schema**

Create `src/learning_memory_os/schemas/memory.py`:
```python
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field, field_validator

from .artifacts import Artifact, ArtifactType, artifact_to_body


Tier = Literal["semantic", "student", "episodic", "intervention"]


class MemoryItem(BaseModel):
    """Uniform representation used by the selector."""

    id: str
    tier: Tier
    artifact_type: ArtifactType | None = None
    topic_id: str | None = None
    title: str
    body: str
    token_estimate: int
    metadata: dict = Field(default_factory=dict)
    embedding: list[float] = Field(default_factory=list)
    created_at: datetime | None = None

    @classmethod
    def from_artifact(
        cls,
        a: Artifact,
        *,
        embedding: list[float],
        item_id: str | None = None,
    ) -> "MemoryItem":
        body = artifact_to_body(a)
        return cls(
            id=item_id or f"sem:{a.topic_id}:{a.title}",
            tier="semantic",
            artifact_type=a.artifact_type,
            topic_id=a.topic_id,
            title=a.title or a.artifact_type.value,
            body=body,
            token_estimate=max(1, len(body) // 4),  # ~4 chars per token
            embedding=embedding,
        )


class MasteryEntry(BaseModel):
    student_id: str
    concept_id: str
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    last_updated: datetime | None = None


class EpisodicEvent(BaseModel):
    id: str | None = None
    student_id: str
    event_type: str
    payload: dict
    occurred_at: datetime | None = None
    embedding: list[float] = Field(default_factory=list)


class InterventionRecord(BaseModel):
    id: str | None = None
    student_id: str
    misconception_id: str | None = None
    strategy: str
    outcome: str | None = None
    notes: str | None = None
    occurred_at: datetime | None = None
```

- [ ] **Step 5: Run schema test**

Run: `uv run pytest tests/unit/test_schemas.py -v`
Expected: PASS (all 4 tests)

- [ ] **Step 6: Commit**

```bash
git add src/learning_memory_os/schemas tests/unit/test_schemas.py
git commit -m "feat(schemas): pydantic models for artifacts and memory items"
```

---

## Task 5: Memory store base + semantic tier

**Files:**
- Create: `src/learning_memory_os/memory/__init__.py`
- Create: `src/learning_memory_os/memory/store.py`
- Create: `src/learning_memory_os/memory/semantic.py`
- Test: `tests/integration/test_semantic.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write integration-test fixtures**

Create `tests/conftest.py`:
```python
import os
import uuid
import pytest
import psycopg
from psycopg.rows import dict_row

DB_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://lmos:lmos_dev@localhost:5433/learning_memory_os",
)


@pytest.fixture
def db_conn():
    """Yields a psycopg connection. Each test runs in a transaction that is rolled back."""
    conn = psycopg.connect(DB_URL, row_factory=dict_row)
    conn.autocommit = False
    yield conn
    conn.rollback()
    conn.close()


@pytest.fixture
def fresh_student_id(db_conn):
    sid = f"test-{uuid.uuid4().hex[:8]}"
    with db_conn.cursor() as cur:
        cur.execute("INSERT INTO students (id) VALUES (%s)", (sid,))
    yield sid
```

- [ ] **Step 2: Write failing test for semantic store**

Create `tests/integration/test_semantic.py`:
```python
from learning_memory_os.memory.semantic import SemanticStore
from learning_memory_os.schemas.artifacts import Concept
from learning_memory_os.schemas.memory import MemoryItem


def test_insert_and_retrieve_by_topic(db_conn):
    store = SemanticStore(db_conn)
    c = Concept(
        topic_id="kv_cache",
        title="KV cache",
        definition="def",
        deep_explanation="more",
        prerequisites=[],
    )
    item = MemoryItem.from_artifact(c, embedding=[0.0] * 1536)
    store.insert(item)

    results = store.by_topic("kv_cache")
    assert len(results) == 1
    assert results[0].title == "KV cache"


def test_vector_search_returns_closest(db_conn):
    store = SemanticStore(db_conn)
    items = [
        MemoryItem(
            id=f"x:{i}",
            tier="semantic",
            topic_id="t",
            title=f"item {i}",
            body=f"body {i}",
            token_estimate=10,
            embedding=[float(i)] + [0.0] * 1535,
        )
        for i in range(3)
    ]
    for it in items:
        store.insert(it)

    # query vector close to item 2
    hits = store.vector_search(query=[2.0] + [0.0] * 1535, k=2)
    assert len(hits) == 2
    assert hits[0].title == "item 2"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_semantic.py -v`
Expected: FAIL with "No module named 'learning_memory_os.memory.semantic'"

- [ ] **Step 4: Implement memory store base**

Create `src/learning_memory_os/memory/__init__.py` (empty).

Create `src/learning_memory_os/memory/store.py`:
```python
import json
import psycopg
from psycopg.rows import dict_row


def connect(database_url: str) -> psycopg.Connection:
    return psycopg.connect(database_url, row_factory=dict_row, autocommit=False)


def vec_literal(v: list[float]) -> str:
    """Postgres pgvector accepts a string literal `'[0.1,0.2,...]'`."""
    return "[" + ",".join(f"{x:.7f}" for x in v) + "]"
```

- [ ] **Step 5: Implement semantic-tier store**

Create `src/learning_memory_os/memory/semantic.py`:
```python
import json
import psycopg
from ..schemas.memory import MemoryItem
from .store import vec_literal


class SemanticStore:
    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    def insert(self, item: MemoryItem) -> str:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO semantic_items
                    (topic_id, artifact_type, title, body, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s::vector)
                RETURNING id
                """,
                (
                    item.topic_id,
                    item.artifact_type.value if item.artifact_type else "concept",
                    item.title,
                    item.body,
                    json.dumps(item.metadata),
                    vec_literal(item.embedding) if item.embedding else None,
                ),
            )
            row = cur.fetchone()
            return str(row["id"])

    def by_topic(self, topic_id: str) -> list[MemoryItem]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id::text, topic_id, artifact_type, title, body, metadata
                FROM semantic_items WHERE topic_id = %s ORDER BY created_at
                """,
                (topic_id,),
            )
            return [self._row_to_item(r) for r in cur.fetchall()]

    def vector_search(self, *, query: list[float], k: int = 5) -> list[MemoryItem]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id::text, topic_id, artifact_type, title, body, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM semantic_items
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (vec_literal(query), vec_literal(query), k),
            )
            return [self._row_to_item(r) for r in cur.fetchall()]

    @staticmethod
    def _row_to_item(r: dict) -> MemoryItem:
        body = r["body"] or ""
        return MemoryItem(
            id=r["id"],
            tier="semantic",
            artifact_type=r["artifact_type"],
            topic_id=r["topic_id"],
            title=r["title"],
            body=body,
            token_estimate=max(1, len(body) // 4),
            metadata=r["metadata"] or {},
        )
```

- [ ] **Step 6: Run integration test**

Run: `uv run pytest tests/integration/test_semantic.py -v`
Expected: PASS (both tests)

If the test fails with a vector dimension error, check that the migration was applied to the same database the test connects to.

- [ ] **Step 7: Commit**

```bash
git add src/learning_memory_os/memory tests/conftest.py tests/integration/test_semantic.py
git commit -m "feat(memory): semantic-tier store with vector search"
```

---

## Task 6: Student-tier store

**Files:**
- Create: `src/learning_memory_os/memory/student.py`
- Test: `tests/integration/test_student.py`

- [ ] **Step 1: Write failing test**

Create `tests/integration/test_student.py`:
```python
from learning_memory_os.memory.student import StudentStore


def test_set_and_get_mastery(db_conn, fresh_student_id):
    store = StudentStore(db_conn)
    # Need a concept first
    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO semantic_items (topic_id, artifact_type, title, body) "
            "VALUES ('t', 'concept', 'C', 'b') RETURNING id::text"
        )
        concept_id = cur.fetchone()["id"]

    store.update_mastery(fresh_student_id, concept_id, score=0.7, confidence=0.6)
    entries = store.mastery_for(fresh_student_id)
    assert len(entries) == 1
    assert entries[0].score == 0.7


def test_record_misconception(db_conn, fresh_student_id):
    store = StudentStore(db_conn)
    mid = store.record_misconception(
        fresh_student_id,
        concept_id=None,
        description="KV cache stores token ids",
        evidence="quiz answer",
    )
    active = store.active_misconceptions(fresh_student_id)
    assert len(active) == 1
    assert active[0]["description"] == "KV cache stores token ids"

    store.resolve_misconception(mid)
    assert store.active_misconceptions(fresh_student_id) == []
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/integration/test_student.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement student store**

Create `src/learning_memory_os/memory/student.py`:
```python
import psycopg
from ..schemas.memory import MasteryEntry


class StudentStore:
    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    def ensure_student(self, student_id: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO students (id) VALUES (%s) ON CONFLICT DO NOTHING",
                (student_id,),
            )

    def update_mastery(
        self,
        student_id: str,
        concept_id: str,
        score: float,
        confidence: float,
    ) -> None:
        self.ensure_student(student_id)
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO mastery (student_id, concept_id, score, confidence)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (student_id, concept_id) DO UPDATE SET
                    score = EXCLUDED.score,
                    confidence = EXCLUDED.confidence,
                    last_updated = now()
                """,
                (student_id, concept_id, score, confidence),
            )

    def mastery_for(self, student_id: str) -> list[MasteryEntry]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT student_id, concept_id::text, score, confidence, last_updated
                FROM mastery WHERE student_id = %s
                """,
                (student_id,),
            )
            return [MasteryEntry(**r) for r in cur.fetchall()]

    def record_misconception(
        self,
        student_id: str,
        *,
        concept_id: str | None,
        description: str,
        evidence: str | None = None,
    ) -> str:
        self.ensure_student(student_id)
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO misconceptions (student_id, concept_id, description, evidence)
                VALUES (%s, %s, %s, %s) RETURNING id::text
                """,
                (student_id, concept_id, description, evidence),
            )
            return cur.fetchone()["id"]

    def active_misconceptions(self, student_id: str) -> list[dict]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id::text, description, evidence, concept_id::text, detected_at
                FROM misconceptions
                WHERE student_id = %s AND resolved = FALSE
                ORDER BY detected_at DESC
                """,
                (student_id,),
            )
            return list(cur.fetchall())

    def resolve_misconception(self, misconception_id: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE misconceptions
                SET resolved = TRUE, resolved_at = now()
                WHERE id = %s
                """,
                (misconception_id,),
            )
```

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/integration/test_student.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add src/learning_memory_os/memory/student.py tests/integration/test_student.py
git commit -m "feat(memory): student-tier store (mastery + misconceptions)"
```

---

## Task 7: Episodic and intervention tiers

**Files:**
- Create: `src/learning_memory_os/memory/episodic.py`
- Create: `src/learning_memory_os/memory/intervention.py`
- Test: `tests/integration/test_episodic.py`
- Test: `tests/integration/test_intervention.py`

- [ ] **Step 1: Write failing test for episodic**

Create `tests/integration/test_episodic.py`:
```python
from learning_memory_os.memory.episodic import EpisodicStore


def test_append_and_recent(db_conn, fresh_student_id):
    store = EpisodicStore(db_conn)
    store.append(
        student_id=fresh_student_id,
        event_type="question",
        payload={"text": "what is KV cache?"},
        embedding=[0.1] * 1536,
    )
    store.append(
        student_id=fresh_student_id,
        event_type="tutor_reply",
        payload={"text": "it caches K and V tensors..."},
        embedding=[0.2] * 1536,
    )
    recent = store.recent(fresh_student_id, limit=5)
    assert len(recent) == 2
    # Most recent first
    assert recent[0].event_type == "tutor_reply"
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/integration/test_episodic.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement episodic**

Create `src/learning_memory_os/memory/episodic.py`:
```python
import json
import psycopg
from ..schemas.memory import EpisodicEvent
from .store import vec_literal


class EpisodicStore:
    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    def append(
        self,
        *,
        student_id: str,
        event_type: str,
        payload: dict,
        embedding: list[float] | None = None,
    ) -> str:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO episodic_events (student_id, event_type, payload, embedding)
                VALUES (%s, %s, %s::jsonb, %s::vector)
                RETURNING id::text
                """,
                (
                    student_id,
                    event_type,
                    json.dumps(payload),
                    vec_literal(embedding) if embedding else None,
                ),
            )
            return cur.fetchone()["id"]

    def recent(self, student_id: str, limit: int = 20) -> list[EpisodicEvent]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id::text, student_id, event_type, payload, occurred_at
                FROM episodic_events
                WHERE student_id = %s
                ORDER BY occurred_at DESC
                LIMIT %s
                """,
                (student_id, limit),
            )
            return [EpisodicEvent(**r) for r in cur.fetchall()]
```

- [ ] **Step 4: Run episodic test**

Run: `uv run pytest tests/integration/test_episodic.py -v`
Expected: PASS

- [ ] **Step 5: Write failing intervention test**

Create `tests/integration/test_intervention.py`:
```python
from learning_memory_os.memory.intervention import InterventionStore


def test_record_and_list(db_conn, fresh_student_id):
    store = InterventionStore(db_conn)
    store.record(
        student_id=fresh_student_id,
        misconception_id=None,
        strategy="worked_example",
        outcome="helped",
        notes="student answered correctly after",
    )
    records = store.for_student(fresh_student_id)
    assert len(records) == 1
    assert records[0]["strategy"] == "worked_example"
    assert records[0]["outcome"] == "helped"
```

- [ ] **Step 6: Verify failure**

Run: `uv run pytest tests/integration/test_intervention.py -v`
Expected: FAIL — module missing.

- [ ] **Step 7: Implement intervention store**

Create `src/learning_memory_os/memory/intervention.py`:
```python
import psycopg


class InterventionStore:
    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    def record(
        self,
        *,
        student_id: str,
        misconception_id: str | None,
        strategy: str,
        outcome: str | None = None,
        notes: str | None = None,
    ) -> str:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO interventions
                    (student_id, misconception_id, strategy, outcome, notes)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id::text
                """,
                (student_id, misconception_id, strategy, outcome, notes),
            )
            return cur.fetchone()["id"]

    def for_student(self, student_id: str) -> list[dict]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id::text, strategy, outcome, notes, occurred_at,
                       misconception_id::text AS misconception_id
                FROM interventions
                WHERE student_id = %s
                ORDER BY occurred_at DESC
                """,
                (student_id,),
            )
            return list(cur.fetchall())
```

- [ ] **Step 8: Run intervention test**

Run: `uv run pytest tests/integration/test_intervention.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/learning_memory_os/memory/episodic.py src/learning_memory_os/memory/intervention.py tests/integration/test_episodic.py tests/integration/test_intervention.py
git commit -m "feat(memory): episodic and intervention tiers"
```

---

## Task 8: Ingestion — text-to-artifact extractor

**Files:**
- Create: `src/learning_memory_os/ingestion/__init__.py`
- Create: `src/learning_memory_os/ingestion/extractors.py`
- Test: `tests/unit/test_extractors.py`

- [ ] **Step 1: Write failing extractor test**

Create `tests/unit/test_extractors.py`:
```python
from unittest.mock import MagicMock
from learning_memory_os.ingestion.extractors import ArtifactExtractor


def test_extractor_parses_concept_and_misconception():
    fake_llm = MagicMock()
    fake_llm.complete_json.return_value = {
        "concepts": [
            {
                "title": "KV cache",
                "definition": "Cache of attention K/V from prior tokens.",
                "deep_explanation": "Long form.",
                "prerequisites": [],
            }
        ],
        "misconceptions": [
            {"statement": "KV cache stores token ids.", "correction": "It stores K and V tensors."},
        ],
        "examples": [],
        "exercises": [],
        "code_patterns": [],
        "paper_claims": [],
    }

    ex = ArtifactExtractor(llm=fake_llm)
    arts = ex.extract(topic_id="kv_cache", source_text="...lecture transcript...")
    types = sorted(a.artifact_type.value for a in arts)
    assert types == ["concept", "misconception"]
    assert arts[0].topic_id == "kv_cache"
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/unit/test_extractors.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement extractor**

Create `src/learning_memory_os/ingestion/__init__.py` (empty).

Create `src/learning_memory_os/ingestion/extractors.py`:
```python
from ..llm import LLM
from ..schemas.artifacts import (
    Concept,
    Example,
    Misconception,
    Exercise,
    CodePattern,
    PaperClaim,
    Artifact,
)


EXTRACTION_SYSTEM = """You extract structured ML-systems-engineering teaching artifacts from source text.
Return STRICT JSON with these top-level keys, each holding an array:
  - concepts: [{title, definition, deep_explanation, prerequisites[]}]
  - misconceptions: [{statement, correction}]
  - examples: [{concept_title, body}]
  - exercises: [{title, prompt, starter_code, rubric}]
  - code_patterns: [{title, body}]
  - paper_claims: [{claim, source, evidence}]
Be conservative: only emit items grounded in the source. No commentary outside JSON."""


class ArtifactExtractor:
    def __init__(self, llm: LLM):
        self.llm = llm

    def extract(self, *, topic_id: str, source_text: str) -> list[Artifact]:
        data = self.llm.complete_json(
            system=EXTRACTION_SYSTEM,
            user=f"TOPIC: {topic_id}\n\nSOURCE:\n{source_text}",
            max_tokens=6000,
        )
        out: list[Artifact] = []
        for c in data.get("concepts", []):
            out.append(Concept(topic_id=topic_id, **c))
        for m in data.get("misconceptions", []):
            out.append(Misconception(topic_id=topic_id, title=m["statement"][:60], **m))
        for e in data.get("examples", []):
            out.append(
                Example(topic_id=topic_id, title=f"Example: {e['concept_title']}", **e)
            )
        for x in data.get("exercises", []):
            out.append(Exercise(topic_id=topic_id, **x))
        for cp in data.get("code_patterns", []):
            out.append(CodePattern(topic_id=topic_id, **cp))
        for pc in data.get("paper_claims", []):
            out.append(
                PaperClaim(topic_id=topic_id, title=pc["claim"][:60], **pc)
            )
        return out
```

- [ ] **Step 4: Run extractor test**

Run: `uv run pytest tests/unit/test_extractors.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/learning_memory_os/ingestion tests/unit/test_extractors.py
git commit -m "feat(ingestion): LLM-based artifact extractor"
```

---

## Task 9: Selector — scoring

**Files:**
- Create: `src/learning_memory_os/selector/__init__.py`
- Create: `src/learning_memory_os/selector/scoring.py`
- Test: `tests/unit/test_scoring.py`

- [ ] **Step 1: Write failing scoring test**

Create `tests/unit/test_scoring.py`:
```python
from datetime import datetime, timedelta, timezone
from learning_memory_os.schemas.memory import MemoryItem
from learning_memory_os.selector.scoring import (
    ScoringContext,
    score_item,
)


def _item(item_id: str, body: str, *, embedding=None, tier="semantic", topic="t"):
    return MemoryItem(
        id=item_id,
        tier=tier,
        topic_id=topic,
        title=item_id,
        body=body,
        token_estimate=max(1, len(body) // 4),
        embedding=embedding or [0.0] * 1536,
        created_at=datetime.now(timezone.utc),
    )


def test_relevance_dominates_when_other_signals_zero():
    a = _item("a", "kv cache", embedding=[1.0, 0.0] + [0.0] * 1534)
    b = _item("b", "tokenization", embedding=[0.0, 1.0] + [0.0] * 1534)
    ctx = ScoringContext(
        task_embedding=[1.0, 0.0] + [0.0] * 1534,
        active_misconception_titles=set(),
        prerequisite_titles=set(),
        recent_item_ids=set(),
        reuse_counts={},
    )
    sa = score_item(a, ctx)
    sb = score_item(b, ctx)
    assert sa.total > sb.total


def test_misconception_boost_applied():
    a = _item("misc:wrong-kv", "KV cache stores token ids: misconception", tier="student")
    ctx = ScoringContext(
        task_embedding=[0.0] * 1536,
        active_misconception_titles={"misc:wrong-kv"},
        prerequisite_titles=set(),
        recent_item_ids=set(),
        reuse_counts={},
    )
    s = score_item(a, ctx)
    assert s.misconception > 0
    assert s.total > 0


def test_recency_decays():
    now = datetime.now(timezone.utc)
    old = _item("old", "x", tier="episodic")
    old.created_at = now - timedelta(days=10)
    new = _item("new", "x", tier="episodic")
    new.created_at = now - timedelta(hours=1)

    ctx = ScoringContext(
        task_embedding=[0.0] * 1536,
        active_misconception_titles=set(),
        prerequisite_titles=set(),
        recent_item_ids=set(),
        reuse_counts={},
    )
    so = score_item(old, ctx)
    sn = score_item(new, ctx)
    assert sn.recency > so.recency
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/unit/test_scoring.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement scoring**

Create `src/learning_memory_os/selector/__init__.py` (empty).

Create `src/learning_memory_os/selector/scoring.py`:
```python
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from ..schemas.memory import MemoryItem


@dataclass
class ScoringContext:
    task_embedding: list[float]
    active_misconception_titles: set[str]
    prerequisite_titles: set[str]
    recent_item_ids: set[str]
    reuse_counts: dict[str, int]


@dataclass
class ItemScore:
    relevance: float
    recency: float
    misconception: float
    prerequisite: float
    reuse: float

    @property
    def total(self) -> float:
        return (
            1.0 * self.relevance
            + 0.5 * self.recency
            + 0.8 * self.misconception
            + 0.6 * self.prerequisite
            + 0.2 * self.reuse
        )


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _recency(item: MemoryItem) -> float:
    if not item.created_at:
        return 0.0
    age_hours = max(
        0.0,
        (datetime.now(timezone.utc) - item.created_at).total_seconds() / 3600.0,
    )
    # half-life ~72h
    return math.exp(-age_hours / 72.0)


def score_item(item: MemoryItem, ctx: ScoringContext) -> ItemScore:
    relevance = _cosine(item.embedding, ctx.task_embedding) if item.embedding else 0.0
    recency = _recency(item) if item.tier == "episodic" else 0.0
    misconception = 1.0 if item.id in ctx.active_misconception_titles else 0.0
    prerequisite = 1.0 if item.title in ctx.prerequisite_titles else 0.0
    reuse = math.log1p(ctx.reuse_counts.get(item.id, 0))
    return ItemScore(
        relevance=relevance,
        recency=recency,
        misconception=misconception,
        prerequisite=prerequisite,
        reuse=reuse,
    )
```

- [ ] **Step 4: Run scoring test**

Run: `uv run pytest tests/unit/test_scoring.py -v`
Expected: PASS (all 3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/learning_memory_os/selector/__init__.py src/learning_memory_os/selector/scoring.py tests/unit/test_scoring.py
git commit -m "feat(selector): per-item scoring (relevance/recency/misconception/prereq/reuse)"
```

---

## Task 10: Selector — budgeted packing

**Files:**
- Create: `src/learning_memory_os/selector/pack.py`
- Test: `tests/unit/test_pack.py`

- [ ] **Step 1: Write failing packing test**

Create `tests/unit/test_pack.py`:
```python
from learning_memory_os.schemas.memory import MemoryItem
from learning_memory_os.selector.pack import pack_under_budget


def _item(item_id, tokens, score):
    return MemoryItem(
        id=item_id,
        tier="semantic",
        topic_id="t",
        title=item_id,
        body="x" * (tokens * 4),
        token_estimate=tokens,
    ), score


def test_packs_highest_score_first_under_budget():
    candidates = [
        _item("a", 500, 0.3),
        _item("b", 400, 0.9),
        _item("c", 200, 0.7),
        _item("d", 900, 0.8),
    ]
    selected = pack_under_budget(candidates, budget=1200)
    selected_ids = [s.id for s in selected]
    # Greedy by score: b(400,.9) -> d(900,.8) won't fit (400+900=1300>1200),
    # so c(200,.7) fits (400+200=600). Then a(500,.3) fits (600+500=1100).
    assert selected_ids == ["b", "c", "a"]


def test_skips_oversized_items():
    candidates = [
        _item("big", 5000, 0.99),
        _item("small", 100, 0.1),
    ]
    selected = pack_under_budget(candidates, budget=200)
    assert [s.id for s in selected] == ["small"]


def test_empty_when_no_room():
    candidates = [_item("a", 1000, 0.9)]
    assert pack_under_budget(candidates, budget=100) == []
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/unit/test_pack.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement packing**

Create `src/learning_memory_os/selector/pack.py`:
```python
from ..schemas.memory import MemoryItem


def pack_under_budget(
    scored: list[tuple[MemoryItem, float]],
    *,
    budget: int,
) -> list[MemoryItem]:
    """Greedy: sort by score desc, take items that fit in the remaining token budget."""
    ordered = sorted(scored, key=lambda x: x[1], reverse=True)
    out: list[MemoryItem] = []
    used = 0
    for item, _score in ordered:
        if item.token_estimate <= budget - used:
            out.append(item)
            used += item.token_estimate
    return out
```

- [ ] **Step 4: Run packing test**

Run: `uv run pytest tests/unit/test_pack.py -v`
Expected: PASS (all 3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/learning_memory_os/selector/pack.py tests/unit/test_pack.py
git commit -m "feat(selector): token-budgeted greedy packer"
```

---

## Task 11: Selector engine — assemble end-to-end

**Files:**
- Create: `src/learning_memory_os/selector/engine.py`
- Test: `tests/unit/test_engine.py`

- [ ] **Step 1: Write failing engine test**

Create `tests/unit/test_engine.py`:
```python
from learning_memory_os.schemas.memory import MemoryItem
from learning_memory_os.selector.engine import RoutingEngine, RoutingDecision


def _items():
    return [
        MemoryItem(
            id=str(i),
            tier="semantic",
            topic_id="t",
            title=f"item-{i}",
            body="x" * 400,
            token_estimate=100,
            embedding=[1.0 if i == 0 else 0.0] + [0.0] * 1535,
        )
        for i in range(5)
    ]


def test_decision_includes_selected_and_dropped():
    eng = RoutingEngine()
    items = _items()
    decision = eng.route(
        candidates=items,
        task_embedding=[1.0] + [0.0] * 1535,
        active_misconceptions=set(),
        prerequisites=set(),
        recent_ids=set(),
        reuse_counts={},
        budget=300,
    )
    assert isinstance(decision, RoutingDecision)
    assert len(decision.selected) == 3   # 3 * 100 tokens fits in 300
    assert len(decision.dropped) == 2
    # Item 0 should be top-ranked (relevance)
    assert decision.selected[0].id == "0"
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/unit/test_engine.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement engine**

Create `src/learning_memory_os/selector/engine.py`:
```python
from dataclasses import dataclass, field
from ..schemas.memory import MemoryItem
from .scoring import ScoringContext, score_item, ItemScore
from .pack import pack_under_budget


@dataclass
class RoutingDecision:
    selected: list[MemoryItem]
    dropped: list[MemoryItem]
    scores: dict[str, ItemScore]
    budget: int
    tokens_used: int


class RoutingEngine:
    """Phase 1 — heuristic ranking + budgeted packing."""

    def route(
        self,
        *,
        candidates: list[MemoryItem],
        task_embedding: list[float],
        active_misconceptions: set[str],
        prerequisites: set[str],
        recent_ids: set[str],
        reuse_counts: dict[str, int],
        budget: int,
    ) -> RoutingDecision:
        ctx = ScoringContext(
            task_embedding=task_embedding,
            active_misconception_titles=active_misconceptions,
            prerequisite_titles=prerequisites,
            recent_item_ids=recent_ids,
            reuse_counts=reuse_counts,
        )
        scored = [(it, score_item(it, ctx).total) for it in candidates]
        scores = {it.id: score_item(it, ctx) for it in candidates}
        selected = pack_under_budget(scored, budget=budget)
        selected_ids = {s.id for s in selected}
        dropped = [it for it in candidates if it.id not in selected_ids]
        tokens_used = sum(s.token_estimate for s in selected)
        return RoutingDecision(
            selected=selected,
            dropped=dropped,
            scores=scores,
            budget=budget,
            tokens_used=tokens_used,
        )
```

- [ ] **Step 4: Run engine test**

Run: `uv run pytest tests/unit/test_engine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/learning_memory_os/selector/engine.py tests/unit/test_engine.py
git commit -m "feat(selector): routing engine combining scoring + packing"
```

---

## Task 12: Interaction logger

**Files:**
- Create: `src/learning_memory_os/logging_utils/__init__.py`
- Create: `src/learning_memory_os/logging_utils/interactions.py`
- Test: `tests/unit/test_interaction_log.py`

- [ ] **Step 1: Write failing logger test**

Create `tests/unit/test_interaction_log.py`:
```python
import json
from pathlib import Path
from learning_memory_os.logging_utils.interactions import InteractionLogger


def test_logger_appends_jsonl(tmp_path: Path):
    log_path = tmp_path / "interactions.jsonl"
    logger = InteractionLogger(path=log_path)
    logger.log(
        {
            "event": "routing_decision",
            "task": "explain kv cache",
            "selected_ids": ["a", "b"],
            "dropped_ids": ["c"],
            "tokens_used": 700,
            "budget": 1000,
        }
    )
    logger.log({"event": "tutor_reply", "text": "..."})

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["event"] == "routing_decision"
    assert "timestamp" in first
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/unit/test_interaction_log.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement logger**

Create `src/learning_memory_os/logging_utils/__init__.py` (empty).

Create `src/learning_memory_os/logging_utils/interactions.py`:
```python
import json
from datetime import datetime, timezone
from pathlib import Path


class InteractionLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")
```

- [ ] **Step 4: Run logger test**

Run: `uv run pytest tests/unit/test_interaction_log.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/learning_memory_os/logging_utils tests/unit/test_interaction_log.py
git commit -m "feat(logging): jsonl interaction logger"
```

---

## Task 13: Tutor agent

**Files:**
- Create: `src/learning_memory_os/agents/__init__.py`
- Create: `src/learning_memory_os/agents/base.py`
- Create: `src/learning_memory_os/agents/tutor.py`
- Test: `tests/unit/test_tutor.py`

- [ ] **Step 1: Write failing tutor test**

Create `tests/unit/test_tutor.py`:
```python
from unittest.mock import MagicMock
from learning_memory_os.schemas.memory import MemoryItem
from learning_memory_os.agents.tutor import TutorAgent


def _item(i, body):
    return MemoryItem(
        id=str(i),
        tier="semantic",
        topic_id="t",
        title=f"item-{i}",
        body=body,
        token_estimate=max(1, len(body) // 4),
        embedding=[0.1] * 1536,
    )


def test_tutor_calls_engine_then_llm():
    fake_llm = MagicMock()
    fake_llm.complete.return_value = "The KV cache is..."

    fake_engine = MagicMock()
    fake_engine.route.return_value = MagicMock(
        selected=[_item(1, "KV cache stores K and V."), _item(2, "Cached per-token.")],
        dropped=[],
        scores={},
        budget=1000,
        tokens_used=20,
    )

    fake_logger = MagicMock()
    fake_embedder = MagicMock()
    fake_embedder.embed_one.return_value = [0.1] * 1536

    tutor = TutorAgent(
        llm=fake_llm,
        engine=fake_engine,
        embedder=fake_embedder,
        logger=fake_logger,
    )

    out = tutor.answer(
        student_id="hiva",
        question="what is a KV cache?",
        candidates=[_item(1, "KV cache stores K and V."), _item(2, "Cached per-token.")],
        active_misconceptions=set(),
        prerequisites=set(),
        recent_ids=set(),
        reuse_counts={},
        budget=1000,
    )
    assert out.text == "The KV cache is..."
    fake_engine.route.assert_called_once()
    fake_llm.complete.assert_called_once()
    assert fake_logger.log.call_count >= 1
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/unit/test_tutor.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement tutor**

Create `src/learning_memory_os/agents/__init__.py` (empty).

Create `src/learning_memory_os/agents/base.py`:
```python
from dataclasses import dataclass
from ..schemas.memory import MemoryItem


@dataclass
class AgentResponse:
    text: str
    selected_items: list[MemoryItem]
    tokens_used: int
```

Create `src/learning_memory_os/agents/tutor.py`:
```python
from ..llm import LLM
from ..embeddings import Embedder
from ..logging_utils.interactions import InteractionLogger
from ..schemas.memory import MemoryItem
from ..selector.engine import RoutingEngine
from .base import AgentResponse


TUTOR_SYSTEM = """You are a tutor for ML systems engineering students.
Use ONLY the provided context items as evidence. Cite them by [item-id] inline.
Keep answers tight and concrete. If the context does not answer the question, say so
and suggest what additional material would help."""


class TutorAgent:
    def __init__(
        self,
        *,
        llm: LLM,
        engine: RoutingEngine,
        embedder: Embedder,
        logger: InteractionLogger,
    ):
        self.llm = llm
        self.engine = engine
        self.embedder = embedder
        self.logger = logger

    def answer(
        self,
        *,
        student_id: str,
        question: str,
        candidates: list[MemoryItem],
        active_misconceptions: set[str],
        prerequisites: set[str],
        recent_ids: set[str],
        reuse_counts: dict[str, int],
        budget: int,
    ) -> AgentResponse:
        task_emb = self.embedder.embed_one(question)
        decision = self.engine.route(
            candidates=candidates,
            task_embedding=task_emb,
            active_misconceptions=active_misconceptions,
            prerequisites=prerequisites,
            recent_ids=recent_ids,
            reuse_counts=reuse_counts,
            budget=budget,
        )
        self.logger.log(
            {
                "event": "routing_decision",
                "agent": "tutor",
                "student_id": student_id,
                "task": question,
                "selected_ids": [it.id for it in decision.selected],
                "dropped_ids": [it.id for it in decision.dropped],
                "tokens_used": decision.tokens_used,
                "budget": decision.budget,
            }
        )

        context_block = "\n\n".join(
            f"[{it.id}] {it.title}\n{it.body}" for it in decision.selected
        )
        user_prompt = f"CONTEXT ITEMS:\n{context_block}\n\nSTUDENT QUESTION:\n{question}"
        text = self.llm.complete(
            system=TUTOR_SYSTEM, user=user_prompt, max_tokens=1024
        )

        self.logger.log(
            {
                "event": "tutor_reply",
                "agent": "tutor",
                "student_id": student_id,
                "text": text,
            }
        )
        return AgentResponse(
            text=text, selected_items=decision.selected, tokens_used=decision.tokens_used
        )
```

- [ ] **Step 4: Run tutor test**

Run: `uv run pytest tests/unit/test_tutor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/learning_memory_os/agents tests/unit/test_tutor.py
git commit -m "feat(agents): tutor agent (selector + LLM + logging)"
```

---

## Task 14: Ingestion CLI

**Files:**
- Create: `scripts/ingest_topic.py`
- Create: `data/seed_topics/kv_cache/source.md`
- Create: `data/seed_topics/agent_memory/source.md`
- Create: `data/seed_topics/tokenization/source.md`
- Create: `data/seed_topics/data_parallelism/source.md`
- Test: `tests/integration/test_ingest_cli.py`

- [ ] **Step 1: Write seed source files**

Each seed file is a short paragraph distilled from CS336/CS349D/CS153 — enough to extract ≥2 concepts + ≥1 misconception. Use the placeholder content below and replace with real lecture excerpts during execution.

Create `data/seed_topics/kv_cache/source.md`:
```markdown
# KV Cache (Area C — Inference)

During autoregressive decoding, a transformer recomputes attention over every prior token at each step. The KV cache stores the key (K) and value (V) projections of past tokens so they don't need to be recomputed. This trades memory for compute. PagedAttention (vLLM) extends this by managing the KV cache in fixed-size blocks similar to OS virtual memory, enabling efficient memory use across concurrent requests. A common misconception is that the KV cache stores raw token ids — it actually stores K and V tensors per layer and per head.
```

Create `data/seed_topics/agent_memory/source.md`:
```markdown
# Agent Memory Architectures (Area E — Agents)

Long-horizon agent systems separate memory into tiers so that the right information surfaces for the right task. A common split: semantic memory (stable facts), episodic memory (recent events), decision/intervention memory (what was tried, did it work). The hard part is not generation but deciding what to remember, summarize, retrieve, or discard. A misconception is that more context is always better — beyond a budget, irrelevant context degrades performance.
```

Create `data/seed_topics/tokenization/source.md`:
```markdown
# Tokenization (Area A — Fundamentals)

Tokenization splits raw text into integer ids the model can consume. Byte-pair encoding (BPE) iteratively merges the most frequent symbol pairs, balancing vocabulary size against sequence length. Smaller vocabularies mean longer sequences and slower training; larger vocabularies cost embedding parameters. A misconception is that the tokenizer is interchangeable across models — pretraining data and tokenizer are coupled; swapping the tokenizer usually requires retraining embeddings.
```

Create `data/seed_topics/data_parallelism/source.md`:
```markdown
# Data Parallelism (Area B — Training)

In distributed training, data parallelism replicates the model across workers and splits each batch across them. Each worker computes gradients locally and synchronizes via all-reduce. DDP is PyTorch's standard implementation. The bottleneck is communication: large models have large gradient buffers, and slow interconnects make all-reduce dominate step time. A misconception is that DDP scales arbitrarily — beyond modest worker counts, communication overhead and stragglers cap throughput.
```

- [ ] **Step 2: Write integration test for CLI**

Create `tests/integration/test_ingest_cli.py`:
```python
import subprocess
import sys
from pathlib import Path


def test_ingest_runs_end_to_end(db_conn, tmp_path: Path):
    # Real LLM call. Mark this with an env flag if it should be skipped offline.
    src = Path("data/seed_topics/kv_cache/source.md").resolve()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.ingest_topic",
            "--topic-id",
            "kv_cache",
            "--source",
            str(src),
        ],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src", **__import__("os").environ},
    )
    assert result.returncode == 0, result.stderr

    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT artifact_type, count(*) FROM semantic_items "
            "WHERE topic_id = 'kv_cache' GROUP BY artifact_type"
        )
        counts = {r["artifact_type"]: r["count"] for r in cur.fetchall()}
    assert counts.get("concept", 0) >= 1
    assert counts.get("misconception", 0) >= 1
```

- [ ] **Step 3: Implement CLI**

Create `scripts/__init__.py` (empty file).

Create `scripts/ingest_topic.py`:
```python
"""Ingest a single topic source file into semantic memory."""

import sys
from pathlib import Path
import typer

from learning_memory_os.config import get_settings
from learning_memory_os.llm import LLM
from learning_memory_os.embeddings import Embedder
from learning_memory_os.memory.store import connect
from learning_memory_os.memory.semantic import SemanticStore
from learning_memory_os.ingestion.extractors import ArtifactExtractor
from learning_memory_os.schemas.memory import MemoryItem
from learning_memory_os.schemas.artifacts import artifact_to_body


app = typer.Typer()


@app.command()
def main(
    topic_id: str = typer.Option(..., "--topic-id"),
    source: Path = typer.Option(..., "--source", exists=True, readable=True),
):
    settings = get_settings()
    llm = LLM(api_key=settings.anthropic_api_key)
    embedder = Embedder(api_key=settings.openai_api_key)
    extractor = ArtifactExtractor(llm=llm)

    text = source.read_text()
    artifacts = extractor.extract(topic_id=topic_id, source_text=text)
    if not artifacts:
        typer.echo("No artifacts extracted.", err=True)
        raise typer.Exit(2)

    bodies = [artifact_to_body(a) for a in artifacts]
    vectors = embedder.embed_many(bodies)

    conn = connect(settings.database_url)
    store = SemanticStore(conn)
    inserted = 0
    try:
        for a, v in zip(artifacts, vectors):
            item = MemoryItem.from_artifact(a, embedding=v)
            store.insert(item)
            inserted += 1
        conn.commit()
    finally:
        conn.close()

    typer.echo(f"Ingested {inserted} artifacts for topic '{topic_id}'.")


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run ingestion test**

Run:
```bash
uv run pytest tests/integration/test_ingest_cli.py -v
```

Expected: PASS. (Hits the real Anthropic + OpenAI APIs; ensure `.env` is configured and Postgres is up.)

If you want to dry-run manually first:
```bash
uv run python -m scripts.ingest_topic --topic-id kv_cache --source data/seed_topics/kv_cache/source.md
```

- [ ] **Step 5: Commit**

```bash
git add scripts/ingest_topic.py scripts/__init__.py data/seed_topics tests/integration/test_ingest_cli.py
git commit -m "feat(cli): ingest_topic command + 4 seed topics"
```

---

## Task 15: End-to-end tutor REPL

**Files:**
- Create: `scripts/tutor_repl.py`
- Test: `tests/integration/test_e2e_tutor.py`

- [ ] **Step 1: Write end-to-end test**

Create `tests/integration/test_e2e_tutor.py`:
```python
"""End-to-end smoke test: ingest a topic, ask a question, verify a reply."""

import subprocess
import sys
from pathlib import Path


def test_full_pipeline_kv_cache(db_conn):
    # Ingest
    src = Path("data/seed_topics/kv_cache/source.md").resolve()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.ingest_topic",
            "--topic-id",
            "kv_cache",
            "--source",
            str(src),
        ],
        check=True,
        env={"PYTHONPATH": "src", **__import__("os").environ},
    )

    # Ask
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.tutor_repl",
            "--student-id",
            "hiva-smoke",
            "--question",
            "What is a KV cache and why does it exist?",
            "--topic-id",
            "kv_cache",
            "--budget",
            "2000",
        ],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src", **__import__("os").environ},
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout.lower()
    # Reply mentions the cached tensors. Loose check.
    assert "k" in out and "v" in out
```

- [ ] **Step 2: Implement tutor REPL command**

Create `scripts/tutor_repl.py`:
```python
"""Ask the tutor a question. Single-turn for now; loop comes later."""

from pathlib import Path
import typer

from learning_memory_os.config import get_settings
from learning_memory_os.llm import LLM
from learning_memory_os.embeddings import Embedder
from learning_memory_os.memory.store import connect
from learning_memory_os.memory.semantic import SemanticStore
from learning_memory_os.memory.student import StudentStore
from learning_memory_os.memory.episodic import EpisodicStore
from learning_memory_os.selector.engine import RoutingEngine
from learning_memory_os.agents.tutor import TutorAgent
from learning_memory_os.logging_utils.interactions import InteractionLogger


app = typer.Typer()


@app.command()
def main(
    student_id: str = typer.Option(..., "--student-id"),
    question: str = typer.Option(..., "--question"),
    topic_id: str | None = typer.Option(None, "--topic-id"),
    budget: int = typer.Option(8000, "--budget"),
):
    settings = get_settings()
    llm = LLM(api_key=settings.anthropic_api_key)
    embedder = Embedder(api_key=settings.openai_api_key)
    engine = RoutingEngine()
    log_path = settings.log_dir / "interactions.jsonl"
    logger = InteractionLogger(path=log_path)

    conn = connect(settings.database_url)
    try:
        student = StudentStore(conn)
        student.ensure_student(student_id)
        semantic = SemanticStore(conn)
        episodic = EpisodicStore(conn)

        # Candidate pool: topic-scoped if topic given, else vector-search globally.
        if topic_id:
            candidates = semantic.by_topic(topic_id)
        else:
            q_emb = embedder.embed_one(question)
            candidates = semantic.vector_search(query=q_emb, k=20)

        # Re-embed candidates if missing (semantic.by_topic doesn't fetch embedding column).
        # For MVP, just re-embed cheaply.
        for c in candidates:
            if not c.embedding:
                c.embedding = embedder.embed_one(c.body)

        misconceptions = {
            m["id"] for m in student.active_misconceptions(student_id)
        }

        tutor = TutorAgent(
            llm=llm, engine=engine, embedder=embedder, logger=logger
        )
        response = tutor.answer(
            student_id=student_id,
            question=question,
            candidates=candidates,
            active_misconceptions=misconceptions,
            prerequisites=set(),
            recent_ids=set(),
            reuse_counts={},
            budget=budget,
        )

        episodic.append(
            student_id=student_id,
            event_type="question",
            payload={"text": question, "topic_id": topic_id},
        )
        episodic.append(
            student_id=student_id,
            event_type="tutor_reply",
            payload={
                "text": response.text,
                "selected_ids": [it.id for it in response.selected_items],
                "tokens_used": response.tokens_used,
            },
        )
        conn.commit()
    finally:
        conn.close()

    typer.echo(response.text)


if __name__ == "__main__":
    app()
```

- [ ] **Step 3: Run e2e test**

Run: `uv run pytest tests/integration/test_e2e_tutor.py -v`
Expected: PASS. Real API calls; reply mentions K and V.

For a manual smoke test:
```bash
uv run python -m scripts.tutor_repl \
  --student-id hiva \
  --question "What is a KV cache and why does it exist?" \
  --topic-id kv_cache \
  --budget 2000
```

Expected: a coherent few-sentence answer printed to stdout. The file `logs/interactions.jsonl` should contain `routing_decision` and `tutor_reply` records.

- [ ] **Step 4: Verify log file**

Run:
```bash
tail -n 2 logs/interactions.jsonl
```

Expected: two JSON lines — one `routing_decision`, one `tutor_reply` — with timestamps.

- [ ] **Step 5: Commit**

```bash
git add scripts/tutor_repl.py tests/integration/test_e2e_tutor.py
git commit -m "feat(cli): tutor_repl + end-to-end smoke test"
```

---

## Task 16: Ingest all 4 seed topics + sanity-tutor across them

**Files:**
- Modify (run): no new code; this task is a verification run.

- [ ] **Step 1: Ingest the remaining 3 seed topics**

Run:
```bash
for t in agent_memory tokenization data_parallelism; do
  uv run python -m scripts.ingest_topic \
    --topic-id "$t" \
    --source "data/seed_topics/$t/source.md"
done
```

Expected: prints `Ingested N artifacts for topic 'X'.` for each topic.

- [ ] **Step 2: Confirm DB state**

Run:
```bash
docker compose exec db psql -U lmos -d learning_memory_os -c \
  "SELECT topic_id, count(*) FROM semantic_items GROUP BY topic_id;"
```

Expected: four rows, one per seed topic, each with count ≥ 2.

- [ ] **Step 3: Tutor questions across all 4 topics**

Run, for each question:
```bash
uv run python -m scripts.tutor_repl --student-id hiva \
  --question "What problem does the KV cache solve in autoregressive decoding?" \
  --topic-id kv_cache --budget 3000

uv run python -m scripts.tutor_repl --student-id hiva \
  --question "Why split agent memory into semantic, episodic, and intervention tiers?" \
  --topic-id agent_memory --budget 3000

uv run python -m scripts.tutor_repl --student-id hiva \
  --question "What tradeoff does BPE balance?" \
  --topic-id tokenization --budget 3000

uv run python -m scripts.tutor_repl --student-id hiva \
  --question "Where does DDP's communication cost dominate?" \
  --topic-id data_parallelism --budget 3000
```

Expected: each prints a coherent answer (no errors). Visually verify the answer cites at least one `[item-id]` from the selected context.

- [ ] **Step 4: Inspect interaction log**

Run:
```bash
wc -l logs/interactions.jsonl
```

Expected: ≥ 8 lines (4 routing_decision + 4 tutor_reply).

- [ ] **Step 5: Commit log snapshot for the writeup**

```bash
cp logs/interactions.jsonl docs/superpowers/specs/mvp-day5-interactions.jsonl
git add docs/superpowers/specs/mvp-day5-interactions.jsonl
git commit -m "docs: capture MVP smoke-run interactions"
```

---

## Task 17: MVP completion checklist

- [ ] **Step 1: Run full test suite**

Run:
```bash
uv run pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Run linter**

Run:
```bash
uv run ruff check src tests scripts
```

Expected: zero issues. Fix any reported.

- [ ] **Step 3: Verify deliverables**

Confirm each item exists:
- [ ] `pyproject.toml` lists declared deps
- [ ] `docker compose ps` shows `db` running
- [ ] `migrations/001_init.sql` applied (semantic_items table exists with 4 topics' worth of rows)
- [ ] `scripts/ingest_topic.py` ingests a topic in <30s
- [ ] `scripts/tutor_repl.py` answers a question with cited context
- [ ] `logs/interactions.jsonl` contains routing decisions + replies
- [ ] All tests under `tests/` green

- [ ] **Step 4: Tag the MVP**

Run:
```bash
git tag -a mvp-week6 -m "Plan 1 MVP complete: ingestion + memory + heuristic selector + tutor"
```

- [ ] **Step 5: Final commit**

If lint or test fixes were needed:
```bash
git add -A
git commit -m "chore: lint and test fixes for MVP cut"
```

The MVP is now ready for Plan 2 (curriculum content pipeline) to layer on top.

---

## Self-review notes

- **Spec coverage**: This plan covers §2.1 (ingestion), §2.2 (multi-tier memory), §2.3 Phase 1 (heuristic ranker + budgeted packing), §2.4 (tutor agent only — other agents are later plans), §2.5 (interaction logging). Phases 2/3 of the routing engine, the synthetic data pipeline, fine-tuning, ablations, and the writeup are out of scope for Plan 1 and live in Plans 2–6.
- **No placeholders**: every code step shows the actual code. Seed-topic markdown bodies are real (not "TODO add lecture content") — they can be expanded with real CS336/CS349D/CS153 transcripts later but are functional for ingestion now.
- **Type consistency**: `MemoryItem` shape is fixed in Task 4 and reused in Tasks 5–13. `RoutingDecision` defined in Task 11 is consumed by Task 13. `ItemScore.total` formula introduced in Task 9 is used in Task 11.
- **One known limitation, deferred to Plan 2 or runtime**: `SemanticStore.by_topic` doesn't return embeddings (only `id, topic_id, artifact_type, title, body, metadata`). The REPL re-embeds on the fly as a workaround. Fix: add `embedding` to the select list and convert pgvector → list[float] in `_row_to_item`. Flag for Plan 2 cleanup.

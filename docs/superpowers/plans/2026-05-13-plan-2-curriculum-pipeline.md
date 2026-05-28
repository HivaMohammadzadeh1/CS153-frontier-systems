# Plan 2 — Curriculum Content Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest the full ML-systems-engineer curriculum into semantic memory — all 20 topics from CS336 (training-from-scratch), CS349D (inference infrastructure), and CS153 (frontier systems) — at production quality, resumable and idempotent. Plan 3 (synthetic trajectories + LoRA routers) depends on having a richly populated `semantic_items` table.

**Architecture:** Add a topic-to-source mapping config + a CS336 GitHub fetcher + a bulk ingestion runner with idempotent skip-if-exists + a quality report. The existing Plan 1 ingestion pipeline (Anthropic Claude → Pydantic artifacts → Postgres semantic store) is the unchanged hot path; this plan adds orchestration + sources around it.

**Tech Stack:** Same as Plan 1 (Python 3.11+, uv, Postgres + pgvector, Anthropic/OpenAI, Typer). New deps: `pyyaml` (config), `httpx` (CS336 source fetching).

**Spec reference:** `docs/superpowers/specs/2026-05-12-learning-memory-os-design.md` §3 (curriculum) and §4.1 (corpus stream).

**Prior plan reference:** `docs/superpowers/plans/2026-05-13-plan-1-mvp-system.md` — assumed complete.

---

## Scope decisions baked in

- **CS336 is the primary source** for Plan 2 because it's publicly available as Python lecture files at `https://github.com/stanford-cs336/spring2025-lectures`. The Plan 1 student-zero test already demonstrated that L10 yields ~52 artifacts of good quality.
- **CS349D and CS153 source acquisition is partial.** CS349D's "mini serving engine" project description exists but lecture-level transcripts are not all public. CS153 lectures are PDFs the user has locally. Plan 2 handles whatever's locally available + scaffolds for user-supplied transcripts.
- **No fine-tuning, no synthetic data, no router work.** Those belong to Plan 3.
- **Plan 2 does not seek 100% topic coverage in one shot.** It seeks (a) every CS336 lecture loaded as a separate topic-source, (b) the 20-topic map fully wired with source-file pointers, (c) any missing source files flagged in the quality report so the user knows what to provide.

---

## File Structure

Plan 2 adds:

```
CS153-frontier-systems/
├── config/
│   └── topics.yaml                  # NEW: 20-topic curriculum map
├── scripts/
│   ├── fetch_cs336.py               # NEW: CS336 lecture fetcher
│   ├── ingest_all.py                # NEW: bulk ingestion runner
│   └── quality_report.py            # NEW: post-ingestion sanity report
├── src/learning_memory_os/
│   ├── ingestion/
│   │   ├── lecture_to_markdown.py   # NEW: CS336 .py → .md converter
│   │   └── topic_loader.py          # NEW: read topics.yaml + resolve source files
│   └── memory/
│       └── semantic.py              # MODIFY: add count_by_topic + delete_by_topic
└── data/
    └── seed_topics/                 # MODIFY: add 13 CS336 + 5 CS349D + 3 CS153 subdirs
```

Boundary rationale:
- `topics.yaml` is the single source of truth for curriculum structure — every topic_id, name, area, source files, prerequisites lives there.
- `lecture_to_markdown.py` is pure conversion logic (TDD-friendly, no I/O).
- `topic_loader.py` resolves a topic to its source content; can be unit-tested against a fake filesystem.
- `ingest_all.py` is the only orchestrator that touches the DB — it reuses Plan 1's `ArtifactExtractor` and `SemanticStore`.

---

## Task 1: Topic curriculum config + loader

**Files:**
- Create: `config/topics.yaml`
- Create: `src/learning_memory_os/ingestion/topic_loader.py`
- Test: `tests/unit/test_topic_loader.py`

- [ ] **Step 1: Add pyyaml dependency**

Run:
```bash
uv add pyyaml
```

- [ ] **Step 2: Write the 20-topic curriculum config**

Create `config/topics.yaml`. Use this exact content (the topic_ids and source paths are referenced by later tasks; do not rename without updating subsequent tasks):

```yaml
version: 1
description: "Learning Memory OS curriculum — 20 topics across 5 areas"

areas:
  A: "Model fundamentals (CS336 L1-L4)"
  B: "Training systems (CS336 L5-L8, L11)"
  C: "Inference infrastructure (CS349D + CS336 L10)"
  D: "Data & alignment (CS336 L13-L17)"
  E: "Agent systems & frontier framing (CS153 + project recursion)"

topics:
  # Area A — Model fundamentals
  - id: tokenization
    area: A
    title: "Tokenization (BPE, byte-level)"
    sources:
      - "data/seed_topics/tokenization/source.md"
      - "data/curriculum/cs336_l01_overview.md"
    prerequisites: []

  - id: transformer_architecture
    area: A
    title: "Transformer architecture & hyperparameters"
    sources:
      - "data/curriculum/cs336_l03_architecture.md"
    prerequisites: [tokenization]

  - id: attention_moe
    area: A
    title: "Attention variants & MoE"
    sources:
      - "data/curriculum/cs336_l04_attention_moe.md"
    prerequisites: [transformer_architecture]

  - id: resource_accounting
    area: A
    title: "Resource accounting (FLOPs, memory, arithmetic intensity)"
    sources:
      - "data/curriculum/cs336_l02_resource_accounting.md"
    prerequisites: [transformer_architecture]

  # Area B — Training systems
  - id: gpu_kernels
    area: B
    title: "GPU/TPU hardware & kernels (Triton, FlashAttention2)"
    sources:
      - "data/curriculum/cs336_l05_gpus_tpus.md"
      - "data/curriculum/cs336_l06_kernels.md"
    prerequisites: [resource_accounting]

  - id: data_parallelism
    area: B
    title: "Data parallelism (DDP)"
    sources:
      - "data/seed_topics/data_parallelism/source.md"
      - "data/curriculum/cs336_l07_parallelism.md"
    prerequisites: [resource_accounting]

  - id: sharded_training
    area: B
    title: "Sharded training (FSDP, ZeRO-1/2/3)"
    sources:
      - "data/curriculum/cs336_l08_parallelism.md"
    prerequisites: [data_parallelism]

  - id: model_parallelism
    area: B
    title: "Model parallelism (tensor + pipeline)"
    sources:
      - "data/curriculum/cs336_l08_parallelism.md"
    prerequisites: [sharded_training]

  - id: scaling_laws
    area: B
    title: "Scaling laws"
    sources:
      - "data/curriculum/cs336_l09_scaling.md"
      - "data/curriculum/cs336_l11_scaling.md"
    prerequisites: [resource_accounting]

  # Area C — Inference infrastructure
  - id: kv_cache
    area: C
    title: "KV cache & PagedAttention"
    sources:
      - "data/seed_topics/kv_cache/source.md"
      - "data/seed_topics/cs336_l10_inference/source.md"
    prerequisites: [attention_moe]

  - id: quantization
    area: C
    title: "Quantization (int8/int4, GPTQ/AWQ)"
    sources:
      - "data/seed_topics/cs336_l10_inference/source.md"
      - "data/curriculum/cs349d_quantization.md"
    prerequisites: [resource_accounting]

  - id: speculative_decoding
    area: C
    title: "Speculative decoding"
    sources:
      - "data/seed_topics/cs336_l10_inference/source.md"
      - "data/curriculum/cs349d_speculative.md"
    prerequisites: [kv_cache]

  - id: continuous_batching
    area: C
    title: "Continuous batching & request scheduling"
    sources:
      - "data/seed_topics/cs336_l10_inference/source.md"
      - "data/curriculum/cs349d_batching.md"
    prerequisites: [kv_cache]

  - id: prefill_decode_disagg
    area: C
    title: "Prefill-decode disaggregation & hierarchical caching"
    sources:
      - "data/curriculum/cs349d_disaggregation.md"
    prerequisites: [continuous_batching]

  # Area D — Data & alignment
  - id: pretraining_data
    area: D
    title: "Pretraining data (collection, dedup, filtering)"
    sources:
      - "data/curriculum/cs336_l13_data.md"
      - "data/curriculum/cs336_l14_data.md"
    prerequisites: [tokenization]

  - id: sft_rlhf_dpo
    area: D
    title: "SFT + RLHF / DPO"
    sources:
      - "data/curriculum/cs336_l15_alignment.md"
    prerequisites: [pretraining_data]

  - id: rl_systems
    area: D
    title: "RL systems for reasoning"
    sources:
      - "data/curriculum/cs336_l16_rl.md"
      - "data/curriculum/cs336_l17_rl_systems.md"
    prerequisites: [sft_rlhf_dpo]

  # Area E — Agent systems & frontier framing
  - id: agent_memory
    area: E
    title: "Agent memory architectures"
    sources:
      - "data/seed_topics/agent_memory/source.md"
      - "data/curriculum/cs153_agents_framing.md"
    prerequisites: []

  - id: context_selection
    area: E
    title: "Context selection under budgets"
    sources:
      - "data/curriculum/agents_context_selection.md"
    prerequisites: [agent_memory]

  - id: multi_agent_orchestration
    area: E
    title: "Multi-agent orchestration & long-horizon execution"
    sources:
      - "data/curriculum/agents_orchestration.md"
    prerequisites: [context_selection]
```

- [ ] **Step 3: Write failing test**

Create `tests/unit/test_topic_loader.py`:
```python
from pathlib import Path
import yaml
from learning_memory_os.ingestion.topic_loader import (
    load_topics,
    Topic,
    resolve_sources,
)


def test_load_topics_returns_20_entries(tmp_path: Path):
    cfg = tmp_path / "topics.yaml"
    cfg.write_text(
        """
version: 1
areas: {A: a, B: b, C: c, D: d, E: e}
topics:
  - id: t1
    area: A
    title: Topic 1
    sources: [s1.md]
    prerequisites: []
  - id: t2
    area: A
    title: Topic 2
    sources: [s1.md, s2.md]
    prerequisites: [t1]
"""
    )
    topics = load_topics(cfg)
    assert len(topics) == 2
    assert topics[0].id == "t1"
    assert topics[1].prerequisites == ["t1"]
    assert topics[1].sources == ["s1.md", "s2.md"]


def test_load_topics_real_curriculum_has_20(tmp_path: Path):
    """Smoke test against the committed curriculum config."""
    topics = load_topics(Path("config/topics.yaml"))
    assert len(topics) == 20
    # All five areas represented
    areas = {t.area for t in topics}
    assert areas == {"A", "B", "C", "D", "E"}


def test_resolve_sources_skips_missing(tmp_path: Path):
    (tmp_path / "exists.md").write_text("hello")
    topic = Topic(
        id="t",
        area="A",
        title="t",
        sources=[
            str(tmp_path / "exists.md"),
            str(tmp_path / "missing.md"),
        ],
        prerequisites=[],
    )
    resolved = resolve_sources(topic, base=Path("."))
    # Returns (path, content, exists) tuples; missing files come through with exists=False
    assert resolved[0][2] is True
    assert resolved[0][1] == "hello"
    assert resolved[1][2] is False
    assert resolved[1][1] == ""
```

- [ ] **Step 4: Run to verify failure**

Run: `uv run pytest tests/unit/test_topic_loader.py -v`
Expected: FAIL — module missing.

- [ ] **Step 5: Implement loader**

Create `src/learning_memory_os/ingestion/topic_loader.py`:
```python
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class Topic:
    id: str
    area: str
    title: str
    sources: list[str]
    prerequisites: list[str]


def load_topics(path: Path) -> list[Topic]:
    raw = yaml.safe_load(Path(path).read_text())
    out: list[Topic] = []
    for entry in raw.get("topics", []):
        out.append(
            Topic(
                id=entry["id"],
                area=entry["area"],
                title=entry["title"],
                sources=list(entry.get("sources", [])),
                prerequisites=list(entry.get("prerequisites", [])),
            )
        )
    return out


def resolve_sources(topic: Topic, *, base: Path) -> list[tuple[Path, str, bool]]:
    """Resolve each source path against the project root. Returns (path, content, exists)."""
    out: list[tuple[Path, str, bool]] = []
    for src in topic.sources:
        p = Path(src) if Path(src).is_absolute() else base / src
        if p.exists():
            out.append((p, p.read_text(), True))
        else:
            out.append((p, "", False))
    return out
```

- [ ] **Step 6: Run test**

Run: `uv run pytest tests/unit/test_topic_loader.py -v`
Expected: PASS (all 3 tests). Note: the third test reads the real `config/topics.yaml` and expects 20 entries — it will only pass if the YAML in Step 2 was saved correctly.

- [ ] **Step 7: Commit**

```bash
git add config/topics.yaml src/learning_memory_os/ingestion/topic_loader.py tests/unit/test_topic_loader.py
git commit -m "feat(curriculum): 20-topic config + loader"
```

---

## Task 2: CS336 lecture fetcher + .py-to-markdown converter

**Files:**
- Create: `src/learning_memory_os/ingestion/lecture_to_markdown.py`
- Create: `scripts/fetch_cs336.py`
- Test: `tests/unit/test_lecture_to_markdown.py`

- [ ] **Step 1: Add httpx dependency**

Run:
```bash
uv add httpx
```

- [ ] **Step 2: Write failing converter test**

Create `tests/unit/test_lecture_to_markdown.py`:
```python
from learning_memory_os.ingestion.lecture_to_markdown import convert_lecture_py


SAMPLE_LECTURE = '''
from execute_util import text, link, image


def main():
    text("**Lecture 99: Sample**")
    text("This is the intro paragraph.")
    section_one()


def section_one():
    text("### Section One")
    text("First bullet")
    text("Second bullet")
    image("foo.png", width=300)
    link(title="ref", url="https://example.com")
    text("Closing line for section one")


if __name__ == "__main__":
    main()
'''


def test_extracts_text_calls_in_order():
    md = convert_lecture_py(SAMPLE_LECTURE)
    assert "Lecture 99: Sample" in md
    assert "intro paragraph" in md
    assert "Section One" in md
    assert "First bullet" in md
    assert "Closing line for section one" in md


def test_ignores_image_and_link_calls():
    md = convert_lecture_py(SAMPLE_LECTURE)
    assert "foo.png" not in md
    assert "https://example.com" not in md


def test_stripped_output_is_plain_markdown():
    md = convert_lecture_py(SAMPLE_LECTURE)
    # No leftover Python syntax
    assert "def " not in md
    assert "text(" not in md
    assert "image(" not in md
```

- [ ] **Step 3: Verify failure**

Run: `uv run pytest tests/unit/test_lecture_to_markdown.py -v`
Expected: FAIL — module missing.

- [ ] **Step 4: Implement converter**

Create `src/learning_memory_os/ingestion/lecture_to_markdown.py`:
```python
import ast


def convert_lecture_py(source: str) -> str:
    """Walk the AST of a CS336 lecture .py file and emit the prose inside text(...) calls
    as a markdown document, preserving order. Ignore image(...), link(...), and other calls."""
    tree = ast.parse(source)
    lines: list[str] = []

    class TextCollector(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr

            if func_name == "text" and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    lines.append(arg.value)
            # Recurse to find nested calls
            self.generic_visit(node)

    TextCollector().visit(tree)
    # Join with blank lines so markdown sections render
    return "\n\n".join(lines).strip() + "\n"
```

- [ ] **Step 5: Run converter test**

Run: `uv run pytest tests/unit/test_lecture_to_markdown.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Implement the CS336 fetcher CLI**

Create `scripts/fetch_cs336.py`:
```python
"""Fetch CS336 lecture .py files from GitHub and convert each to markdown under data/curriculum/."""

from pathlib import Path
import httpx
import typer

from learning_memory_os.ingestion.lecture_to_markdown import convert_lecture_py


app = typer.Typer()

REPO_RAW_BASE = "https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main"

# CS336 -> our topic source filenames (matches config/topics.yaml)
LECTURE_MAP = {
    "lecture_01.py": "cs336_l01_overview.md",
    "lecture_02.py": "cs336_l02_resource_accounting.md",
    "lecture_06.py": "cs336_l06_kernels.md",
    "lecture_08.py": "cs336_l08_parallelism.md",
    "lecture_10.py": None,   # Already curated as data/seed_topics/cs336_l10_inference/source.md
    "lecture_12.py": "cs336_l12_evaluation.md",
    "lecture_13.py": "cs336_l13_data.md",
    "lecture_14.py": "cs336_l14_data.md",
    "lecture_17.py": "cs336_l17_rl_systems.md",
}


@app.command()
def main(
    out_dir: Path = typer.Option(Path("data/curriculum"), "--out-dir"),
):
    out_dir.mkdir(parents=True, exist_ok=True)
    fetched = 0
    skipped = 0
    for lecture_file, md_name in LECTURE_MAP.items():
        if md_name is None:
            typer.echo(f"skip {lecture_file} (curated separately)")
            skipped += 1
            continue
        url = f"{REPO_RAW_BASE}/{lecture_file}"
        try:
            resp = httpx.get(url, timeout=30.0)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            typer.echo(f"fetch failed for {lecture_file}: {e}", err=True)
            continue
        md = convert_lecture_py(resp.text)
        target = out_dir / md_name
        target.write_text(md)
        typer.echo(f"wrote {target} ({len(md)} chars)")
        fetched += 1
    typer.echo(f"\nDone. Fetched: {fetched}, skipped: {skipped}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 7: Run the fetcher**

```bash
uv run python -m scripts.fetch_cs336
```

Expected: writes 8 markdown files under `data/curriculum/` (every entry in `LECTURE_MAP` except `lecture_10.py` which is curated). Each should be 5–30KB of plain markdown text.

Spot-check one:
```bash
head -40 data/curriculum/cs336_l02_resource_accounting.md
```

Should show readable prose, no `text(` / `def ` / `image(` syntax.

If any fetch fails (e.g., GitHub returns 404 because a lecture filename changed), capture the error and report — but continue with the rest.

- [ ] **Step 8: Commit**

```bash
git add src/learning_memory_os/ingestion/lecture_to_markdown.py scripts/fetch_cs336.py tests/unit/test_lecture_to_markdown.py
git add data/curriculum/
git commit -m "feat(curriculum): cs336 lecture fetcher + .py→.md converter"
```

---

## Task 3: SemanticStore — count and delete by topic

The bulk ingestion runner needs to skip topics that are already loaded (idempotency) and optionally re-ingest (replace).

**Files:**
- Modify: `src/learning_memory_os/memory/semantic.py`
- Test: `tests/integration/test_semantic_admin.py`

- [ ] **Step 1: Write failing test**

Create `tests/integration/test_semantic_admin.py`:
```python
from learning_memory_os.memory.semantic import SemanticStore
from learning_memory_os.schemas.artifacts import Concept
from learning_memory_os.schemas.memory import MemoryItem


def test_count_by_topic_starts_at_zero(db_conn):
    store = SemanticStore(db_conn)
    assert store.count_by_topic("nonexistent_topic_abc") == 0


def test_count_after_inserts(db_conn):
    store = SemanticStore(db_conn)
    topic = "test_count_isolated"
    for i in range(3):
        c = Concept(
            topic_id=topic,
            title=f"C{i}",
            definition=f"def {i}",
            deep_explanation="",
            prerequisites=[],
        )
        store.insert(MemoryItem.from_artifact(c, embedding=[0.0] * 1536, item_id=f"sem:{topic}:{i}"))
    assert store.count_by_topic(topic) == 3


def test_delete_by_topic_clears_rows(db_conn):
    store = SemanticStore(db_conn)
    topic = "test_delete_isolated"
    for i in range(2):
        c = Concept(
            topic_id=topic,
            title=f"C{i}",
            definition=f"def {i}",
            deep_explanation="",
            prerequisites=[],
        )
        store.insert(MemoryItem.from_artifact(c, embedding=[0.0] * 1536, item_id=f"sem:{topic}:del{i}"))
    assert store.count_by_topic(topic) == 2

    deleted = store.delete_by_topic(topic)
    assert deleted == 2
    assert store.count_by_topic(topic) == 0
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/integration/test_semantic_admin.py -v`
Expected: FAIL — `count_by_topic` and `delete_by_topic` missing.

- [ ] **Step 3: Add the methods**

Append to `src/learning_memory_os/memory/semantic.py` (do NOT remove existing methods):
```python
    def count_by_topic(self, topic_id: str) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) AS n FROM semantic_items WHERE topic_id = %s",
                (topic_id,),
            )
            return int(cur.fetchone()["n"])

    def delete_by_topic(self, topic_id: str) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM semantic_items WHERE topic_id = %s",
                (topic_id,),
            )
            return cur.rowcount
```

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/integration/test_semantic_admin.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/learning_memory_os/memory/semantic.py tests/integration/test_semantic_admin.py
git commit -m "feat(memory): count_by_topic and delete_by_topic admin methods"
```

---

## Task 4: Bulk ingestion runner

**Files:**
- Create: `scripts/ingest_all.py`
- Test: `tests/integration/test_ingest_all.py`

- [ ] **Step 1: Write integration test**

Create `tests/integration/test_ingest_all.py`:
```python
"""Smoke test: bulk ingester loads at least one topic when sources exist."""

import subprocess
import sys


def test_ingest_all_help_runs():
    """The CLI prints its help without errors."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.ingest_all", "--help"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src", **__import__("os").environ},
    )
    assert result.returncode == 0
    assert "ingest" in result.stdout.lower()


def test_ingest_all_dry_run_reports_topics(db_conn):
    """A --dry-run lists the topics it would process and exits 0."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.ingest_all", "--dry-run"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src", **__import__("os").environ},
    )
    assert result.returncode == 0, result.stderr
    # Should mention at least the seed topics we know exist
    assert "kv_cache" in result.stdout
    assert "agent_memory" in result.stdout
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/integration/test_ingest_all.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the runner**

Create `scripts/ingest_all.py`:
```python
"""Bulk-ingest the curriculum defined in config/topics.yaml into semantic memory.

Idempotency: by default, topics with >0 existing artifacts are skipped.
Use --force to delete-and-reingest a topic.
Use --only TOPIC_ID to ingest a single topic.
Use --dry-run to print what would happen without calling any API or DB.
"""

from pathlib import Path
import sys
import typer

from learning_memory_os.config import get_settings
from learning_memory_os.llm import LLM
from learning_memory_os.embeddings import Embedder
from learning_memory_os.memory.store import connect
from learning_memory_os.memory.semantic import SemanticStore
from learning_memory_os.ingestion.extractors import ArtifactExtractor
from learning_memory_os.ingestion.topic_loader import load_topics, resolve_sources, Topic
from learning_memory_os.schemas.memory import MemoryItem
from learning_memory_os.schemas.artifacts import artifact_to_body


app = typer.Typer()


def _ingest_one(
    topic: Topic,
    *,
    extractor: ArtifactExtractor,
    embedder: Embedder,
    store: SemanticStore,
    base: Path,
) -> int:
    resolved = resolve_sources(topic, base=base)
    present = [(p, body) for (p, body, exists) in resolved if exists and body.strip()]
    if not present:
        typer.echo(f"  [skip] {topic.id}: no source files found")
        return 0

    combined = "\n\n".join(
        f"# Source: {p.name}\n\n{body}" for p, body in present
    )
    artifacts = extractor.extract(topic_id=topic.id, source_text=combined)
    if not artifacts:
        typer.echo(f"  [warn] {topic.id}: extractor returned 0 artifacts")
        return 0

    bodies = [artifact_to_body(a) for a in artifacts]
    vectors = embedder.embed_many(bodies)
    inserted = 0
    for a, v in zip(artifacts, vectors):
        item = MemoryItem.from_artifact(a, embedding=v)
        store.insert(item)
        inserted += 1
    return inserted


@app.command()
def main(
    config: Path = typer.Option(Path("config/topics.yaml"), "--config"),
    only: str | None = typer.Option(None, "--only", help="Process only this topic_id"),
    force: bool = typer.Option(False, "--force", help="Delete existing artifacts and re-ingest"),
    dry_run: bool = typer.Option(False, "--dry-run", help="List actions without executing"),
):
    topics = load_topics(config)
    if only:
        topics = [t for t in topics if t.id == only]
        if not topics:
            typer.echo(f"No topic with id={only}", err=True)
            raise typer.Exit(2)

    if dry_run:
        typer.echo("DRY RUN — topics that would be processed:")
        for t in topics:
            typer.echo(f"  {t.area}/{t.id}: {len(t.sources)} source(s)")
            for s in t.sources:
                exists = Path(s).exists()
                marker = "OK" if exists else "MISSING"
                typer.echo(f"    [{marker}] {s}")
        return

    settings = get_settings()
    llm = LLM(api_key=settings.anthropic_api_key)
    embedder = Embedder(api_key=settings.openai_api_key)
    extractor = ArtifactExtractor(llm=llm)
    conn = connect(settings.database_url)
    store = SemanticStore(conn)

    total_inserted = 0
    try:
        for topic in topics:
            existing = store.count_by_topic(topic.id)
            if existing > 0 and not force:
                typer.echo(f"[exists] {topic.id}: {existing} artifacts — use --force to replace")
                continue
            if existing > 0 and force:
                removed = store.delete_by_topic(topic.id)
                typer.echo(f"[clear]  {topic.id}: removed {removed} existing artifacts")

            typer.echo(f"[ingest] {topic.id} (area {topic.area})")
            n = _ingest_one(
                topic,
                extractor=extractor,
                embedder=embedder,
                store=store,
                base=Path("."),
            )
            typer.echo(f"  -> {n} artifacts")
            total_inserted += n
            conn.commit()  # Commit per topic so partial progress survives interruption
    finally:
        conn.close()

    typer.echo(f"\nTotal inserted: {total_inserted}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/integration/test_ingest_all.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/ingest_all.py tests/integration/test_ingest_all.py
git commit -m "feat(curriculum): bulk ingestion runner with idempotency + dry-run"
```

---

## Task 5: Run the bulk ingestion

This task is a runbook, no new code.

- [ ] **Step 1: Verify Docker and .env**

```bash
docker compose ps db
grep -E '^(ANTHROPIC|OPENAI)_API_KEY' .env | head -2
```

Both should show valid (non-placeholder) values. If not, report BLOCKED.

- [ ] **Step 2: Dry-run to see coverage**

```bash
uv run python -m scripts.ingest_all --dry-run
```

Capture output. Note which topics show `MISSING` for any source file — these are topics with no available material. They will be skipped (logged as `[skip]`) but the bulk run will still process every topic with at least one source available.

- [ ] **Step 3: Run the full bulk ingestion**

```bash
uv run python -m scripts.ingest_all 2>&1 | tee data/runtime/ingest_all_$(date +%Y%m%d_%H%M%S).log
```

Expected duration: ~5–15 minutes depending on how many topics have sources. API costs: a few dollars in Anthropic + cents in OpenAI embeddings.

If any single topic fails, the runner should print an error for it and continue with the next. (Subprocess that handles a topic is not isolated — if Python raises, the whole run dies. If you observe that, capture which topic killed the run and report BLOCKED.)

- [ ] **Step 4: Verify DB state**

```bash
docker compose exec db psql -U lmos -d learning_memory_os -c \
  "SELECT topic_id, count(*) FROM semantic_items GROUP BY topic_id ORDER BY topic_id;"
```

Capture the table. Goal: every topic with source content in the YAML should appear with count >= 2.

- [ ] **Step 5: Commit the ingest log**

```bash
git add data/runtime/ingest_all_*.log
```

(Note: `data/runtime/` is currently gitignored per Plan 1's `.gitignore`. If git refuses, place the log at `data/ingest_logs/` instead and `.gitignore` exclude future entries beyond the first one.)

If gitignore blocks, just don't commit the log — record the counts in the next task's quality report instead.

---

## Task 6: Quality report script

**Files:**
- Create: `scripts/quality_report.py`
- Test: none (script-level utility; manual run)

- [ ] **Step 1: Implement the script**

Create `scripts/quality_report.py`:
```python
"""Print a per-topic ingestion quality report:
- artifact count per type
- a random sample of artifact bodies for spot-check
- topics with zero artifacts (i.e., missing source content)
"""

import random
import typer

from learning_memory_os.config import get_settings
from learning_memory_os.ingestion.topic_loader import load_topics
from learning_memory_os.memory.store import connect


app = typer.Typer()


@app.command()
def main(
    samples_per_topic: int = typer.Option(1, "--samples"),
    config: str = typer.Option("config/topics.yaml", "--config"),
):
    from pathlib import Path
    settings = get_settings()
    topics = load_topics(Path(config))
    conn = connect(settings.database_url)
    try:
        for topic in topics:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT artifact_type, count(*) AS n "
                    "FROM semantic_items WHERE topic_id = %s "
                    "GROUP BY artifact_type ORDER BY artifact_type",
                    (topic.id,),
                )
                breakdown = list(cur.fetchall())
            total = sum(r["n"] for r in breakdown)
            typer.echo(f"\n=== {topic.area}/{topic.id} ({topic.title}) — total {total}")
            if not breakdown:
                typer.echo("  (no artifacts; topic missing or extraction failed)")
                continue
            for row in breakdown:
                typer.echo(f"  {row['artifact_type']}: {row['n']}")
            # Sample
            if samples_per_topic > 0:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT artifact_type, title, body FROM semantic_items "
                        "WHERE topic_id = %s ORDER BY random() LIMIT %s",
                        (topic.id, samples_per_topic),
                    )
                    samples = list(cur.fetchall())
                for s in samples:
                    typer.echo(f"  ~ sample [{s['artifact_type']}] {s['title']}")
                    body_preview = (s["body"] or "")[:200].replace("\n", " ")
                    typer.echo(f"    {body_preview}{'...' if len(s['body'] or '') > 200 else ''}")
    finally:
        conn.close()


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Run the report**

```bash
uv run python -m scripts.quality_report --samples 2 | tee data/quality_report.txt
```

Read the output yourself: are extraction outputs coherent? Are any topics suspiciously thin (< 4 artifacts)? Flag concerns.

- [ ] **Step 3: Commit script + report snapshot**

```bash
git add scripts/quality_report.py data/quality_report.txt
git commit -m "feat(curriculum): quality report + first snapshot"
```

---

## Task 7: Smoke-test tutoring across the loaded curriculum

This is a runbook task, no new code. The goal is to confirm the tutor works on at least 3 newly-loaded topics across different Areas, just like Plan 1 Task 16 did for the seed topics.

- [ ] **Step 1: Pick 3 topics from the loaded set**

Choose one each from Areas A, B, D (Areas C and E were validated in Plan 1). Pick whichever topics actually have artifacts per the quality report. Suggestions if available:

- Area A: `transformer_architecture` or `resource_accounting`
- Area B: `gpu_kernels` or `scaling_laws`
- Area D: `pretraining_data` or `sft_rlhf_dpo`

- [ ] **Step 2: Run the tutor on each**

For each chosen topic, ask one substantive question. Example commands (replace topics + questions with the actual choices):

```bash
uv run python -m scripts.tutor_repl --student-id hiva-curriculum-check \
  --question "Why does arithmetic intensity matter on an H100?" \
  --topic-id resource_accounting --budget 3000

uv run python -m scripts.tutor_repl --student-id hiva-curriculum-check \
  --question "When does Triton beat hand-tuned CUDA in practice?" \
  --topic-id gpu_kernels --budget 3000

uv run python -m scripts.tutor_repl --student-id hiva-curriculum-check \
  --question "What changed when deduplication was added to Common Crawl pipelines?" \
  --topic-id pretraining_data --budget 3000
```

- [ ] **Step 3: Note quality**

For each reply, confirm:
- It's coherent
- It uses inline `[a1b2c3d4]` short-id citations (the Plan 1 followup #2 should be working)
- It doesn't hallucinate facts not in the source

If any reply is hallucinated or off-topic, that points to a thin or low-quality ingestion for that topic — note in the quality report.

- [ ] **Step 4: Capture log snapshot**

```bash
cp logs/interactions.jsonl data/curriculum-smoke-interactions.jsonl
git add data/curriculum-smoke-interactions.jsonl
git commit -m "docs: curriculum smoke-test interaction log"
```

---

## Task 8: Tag and merge readiness

- [ ] **Step 1: Full test suite**

```bash
uv run pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Ruff**

```bash
uv run ruff check src tests scripts
```

Expected: clean. Fix any issues and commit `chore: ruff fixes`.

- [ ] **Step 3: Tag**

```bash
git tag -a curriculum-loaded -m "Plan 2 complete: full CS336-anchored curriculum ingested into semantic memory"
git tag -n
```

- [ ] **Step 4: Final state report**

Produce a short markdown summary at `data/plan-2-summary.md`:

```markdown
# Plan 2 — Curriculum Loaded

- Topics defined: 20
- Topics with artifacts: <count from quality report>
- Topics missing source content: <list>
- Total semantic_items: <count>
- Tag: `curriculum-loaded`

## Open gaps (deferred to user-supplied content)
- CS349D lectures: <list of cs349d_*.md files not yet written>
- CS153 lectures: <list of cs153_*.md files not yet written>
- Project-recursion topics (agents_context_selection, agents_orchestration): need authored material
```

Commit:
```bash
git add data/plan-2-summary.md
git commit -m "docs: plan 2 final state summary"
```

---

## Self-review notes

- **Spec coverage**: Plan 2 implements §3 (curriculum) and §4.1 corpus stream of the design spec. It does NOT touch §2.3 Phase 2 (combinatorial selector) or Phase 3 (fine-tuned router) — those are Plan 3+.
- **No placeholders**: every code step has actual code. The YAML config in Task 1 lists 20 fully-specified topics with real paths.
- **Type consistency**: `Topic` dataclass defined in Task 1 is used in Task 4. `SemanticStore.count_by_topic` and `.delete_by_topic` from Task 3 are used by Task 4.
- **Known gap, deferred to user-supplied content**: 5 CS349D and 3 CS153 source files (`data/curriculum/cs349d_*.md`, `data/curriculum/cs153_*.md`) plus 2 project-recursion files (`data/curriculum/agents_*.md`) are referenced in the YAML but not generated by any task. The bulk ingester `[skip]`s these gracefully; the quality report flags them; Task 8's summary documents what's missing for the user to fill in.
- **Idempotency**: `ingest_all` is rerunnable. `--force` allows full re-ingest of a topic. Per-topic commits inside `_ingest_one` mean an interrupted run preserves partial progress.

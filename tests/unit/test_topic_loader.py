from pathlib import Path
from learning_memory_os.ingestion.topic_loader import (
    load_topics,
    Topic,
    resolve_sources,
)


def test_load_topics_returns_n_entries(tmp_path: Path):
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


def test_load_topics_real_curriculum_has_28():
    """Smoke test against the committed curriculum config."""
    topics = load_topics(Path("config/topics.yaml"))
    assert len(topics) == 28
    areas = {t.area for t in topics}
    assert areas == {"A", "B", "C", "D", "E", "F"}


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
    assert resolved[0][2] is True
    assert resolved[0][1] == "hello"
    assert resolved[1][2] is False
    assert resolved[1][1] == ""

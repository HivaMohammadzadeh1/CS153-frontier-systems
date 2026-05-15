from dataclasses import dataclass
from pathlib import Path
import yaml
import psycopg


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


def resolve_prerequisite_titles(
    conn: psycopg.Connection,
    *,
    topic_id: str,
    topics: list[Topic],
) -> set[str]:
    """For a given topic, return the set of concept titles from its prerequisite topics.

    The selector's scoring uses `prerequisite_titles` (set[str]) which is matched against
    `MemoryItem.title`. We look up concept-type artifacts in the prereq topics.
    """
    topic = next((t for t in topics if t.id == topic_id), None)
    if topic is None or not topic.prerequisites:
        return set()

    prereq_ids = list(topic.prerequisites)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT title FROM semantic_items
            WHERE topic_id = ANY(%s) AND artifact_type = 'concept'
            """,
            (prereq_ids,),
        )
        return {r["title"] for r in cur.fetchall()}


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

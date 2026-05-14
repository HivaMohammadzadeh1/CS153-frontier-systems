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

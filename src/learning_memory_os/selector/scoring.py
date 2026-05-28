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
    if item.tier == "xtrace":
        relevance = float(item.metadata.get("xtrace_similarity", 0.0))
    else:
        relevance = _cosine(item.embedding, ctx.task_embedding) if item.embedding else 0.0
    recency = _recency(item) if item.tier in ("episodic", "xtrace") else 0.0
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

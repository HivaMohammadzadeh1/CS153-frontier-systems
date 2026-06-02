import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from ..schemas.memory import MemoryItem


@dataclass
class ScoringContext:
    task_embedding: list[float]
    # Semantic-item ids of the concepts a student's active misconceptions point
    # at (misconceptions.concept_id -> semantic_items.id). Items whose id is in
    # this set directly address a live misconception.
    misconception_concept_ids: set[str] = field(default_factory=set)
    # Topics those flagged concepts belong to — items in the same topic get a
    # partial boost even if they aren't the exact flagged concept.
    misconception_topics: set[str] = field(default_factory=set)
    prerequisite_titles: set[str] = field(default_factory=set)
    recent_item_ids: set[str] = field(default_factory=set)
    reuse_counts: dict[str, int] = field(default_factory=dict)
    # Concept ids that are due/overdue for spaced-repetition review (Track B).
    due_concept_ids: set[str] = field(default_factory=set)


@dataclass
class ItemScore:
    relevance: float
    recency: float
    misconception: float
    prerequisite: float
    reuse: float
    review_due: float = 0.0

    @property
    def total(self) -> float:
        return (
            1.0 * self.relevance
            + 0.5 * self.recency
            + 0.8 * self.misconception
            + 0.6 * self.prerequisite
            + 0.2 * self.reuse
            + 0.3 * self.review_due
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
    if item.id in ctx.misconception_concept_ids:
        misconception = 1.0
    elif item.topic_id and item.topic_id in ctx.misconception_topics:
        misconception = 0.4
    else:
        misconception = 0.0
    prerequisite = 1.0 if item.title in ctx.prerequisite_titles else 0.0
    reuse = math.log1p(ctx.reuse_counts.get(item.id, 0))
    review_due = 1.0 if item.id in ctx.due_concept_ids else 0.0
    return ItemScore(
        relevance=relevance,
        recency=recency,
        misconception=misconception,
        prerequisite=prerequisite,
        reuse=reuse,
        review_due=review_due,
    )

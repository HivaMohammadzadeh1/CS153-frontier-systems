from dataclasses import dataclass
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
        misconception_concept_ids: set[str] | None = None,
        misconception_topics: set[str] | None = None,
        prerequisites: set[str],
        recent_ids: set[str],
        reuse_counts: dict[str, int],
        due_concept_ids: set[str] | None = None,
        budget: int,
        diversity: float = 0.7,
    ) -> RoutingDecision:
        ctx = ScoringContext(
            task_embedding=task_embedding,
            misconception_concept_ids=misconception_concept_ids or set(),
            misconception_topics=misconception_topics or set(),
            prerequisite_titles=prerequisites,
            recent_item_ids=recent_ids,
            reuse_counts=reuse_counts,
            due_concept_ids=due_concept_ids or set(),
        )
        scored = [(it, score_item(it, ctx).total) for it in candidates]
        scores = {it.id: score_item(it, ctx) for it in candidates}
        selected = pack_under_budget(scored, budget=budget, diversity=diversity)
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

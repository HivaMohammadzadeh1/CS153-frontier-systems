import math
from ..schemas.memory import MemoryItem


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def pack_under_budget(
    scored: list[tuple[MemoryItem, float]],
    *,
    budget: int,
    diversity: float = 0.0,
) -> list[MemoryItem]:
    """Pick items under a token budget.

    diversity (lambda) in [0, 1] turns on Maximal Marginal Relevance: each pick
    maximizes ``(1 - diversity) * score - diversity * max_cosine_to_selected``,
    so near-duplicate chunks don't crowd out broader coverage. With
    ``diversity == 0`` (or when items lack embeddings) this is the original
    greedy-by-score behavior.
    """
    if diversity <= 0.0:
        return _greedy(scored, budget)

    remaining = list(scored)
    out: list[MemoryItem] = []
    used = 0
    while remaining:
        best_idx = None
        best_mmr = None
        for i, (item, score) in enumerate(remaining):
            if item.token_estimate > budget - used:
                continue
            penalty = max(
                (_cosine(item.embedding, sel.embedding) for sel in out),
                default=0.0,
            )
            mmr = (1.0 - diversity) * score - diversity * penalty
            if best_mmr is None or mmr > best_mmr:
                best_mmr, best_idx = mmr, i
        if best_idx is None:
            break  # nothing left fits
        item, _ = remaining.pop(best_idx)
        out.append(item)
        used += item.token_estimate
    return out


def _greedy(scored: list[tuple[MemoryItem, float]], budget: int) -> list[MemoryItem]:
    ordered = sorted(scored, key=lambda x: x[1], reverse=True)
    out: list[MemoryItem] = []
    used = 0
    for item, _score in ordered:
        if item.token_estimate <= budget - used:
            out.append(item)
            used += item.token_estimate
    return out

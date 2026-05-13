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

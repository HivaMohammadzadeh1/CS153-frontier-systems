from learning_memory_os.schemas.memory import MemoryItem
from learning_memory_os.selector.pack import pack_under_budget


def _item(item_id, tokens, score, emb):
    return MemoryItem(
        id=item_id, tier="semantic", topic_id="t", title=item_id,
        body="x" * (tokens * 4), token_estimate=tokens, embedding=emb,
    ), score


def test_mmr_prefers_diverse_item_over_near_duplicate():
    # a and b are near-identical; c is orthogonal. Budget fits only two items.
    a = _item("a", 100, 0.90, [1.0, 0.0])
    b = _item("b", 100, 0.88, [0.99, 0.01])   # near-duplicate of a
    c = _item("c", 100, 0.70, [0.0, 1.0])      # diverse, lower score
    selected = [s.id for s in pack_under_budget([a, b, c], budget=200, diversity=0.7)]
    assert "a" in selected            # top item always taken
    assert "c" in selected            # diversity pulls in the orthogonal item
    assert "b" not in selected        # the near-duplicate is suppressed


def test_diversity_zero_is_plain_greedy():
    a = _item("a", 100, 0.90, [1.0, 0.0])
    b = _item("b", 100, 0.88, [0.99, 0.01])
    c = _item("c", 100, 0.70, [0.0, 1.0])
    selected = [s.id for s in pack_under_budget([a, b, c], budget=200, diversity=0.0)]
    assert selected == ["a", "b"]     # pure score order, ignores redundancy

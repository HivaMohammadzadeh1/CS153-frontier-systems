from learning_memory_os.schemas.memory import MemoryItem
from learning_memory_os.selector.pack import pack_under_budget


def _item(item_id, tokens, score):
    return MemoryItem(
        id=item_id,
        tier="semantic",
        topic_id="t",
        title=item_id,
        body="x" * (tokens * 4),
        token_estimate=tokens,
    ), score


def test_packs_highest_score_first_under_budget():
    candidates = [
        _item("a", 500, 0.3),
        _item("b", 400, 0.9),
        _item("c", 200, 0.7),
        _item("d", 900, 0.8),
    ]
    selected = pack_under_budget(candidates, budget=1200)
    selected_ids = [s.id for s in selected]
    # Greedy by score: b(400,.9) -> d(900,.8) won't fit (400+900=1300>1200),
    # so c(200,.7) fits (400+200=600). Then a(500,.3) fits (600+500=1100).
    assert selected_ids == ["b", "c", "a"]


def test_skips_oversized_items():
    candidates = [
        _item("big", 5000, 0.99),
        _item("small", 100, 0.1),
    ]
    selected = pack_under_budget(candidates, budget=200)
    assert [s.id for s in selected] == ["small"]


def test_empty_when_no_room():
    candidates = [_item("a", 1000, 0.9)]
    assert pack_under_budget(candidates, budget=100) == []

from learning_memory_os.schemas.memory import MemoryItem
from learning_memory_os.selector.engine import RoutingEngine, RoutingDecision


def _items():
    return [
        MemoryItem(
            id=str(i),
            tier="semantic",
            topic_id="t",
            title=f"item-{i}",
            body="x" * 400,
            token_estimate=100,
            embedding=[1.0 if i == 0 else 0.0] + [0.0] * 1535,
        )
        for i in range(5)
    ]


def test_decision_includes_selected_and_dropped():
    eng = RoutingEngine()
    items = _items()
    decision = eng.route(
        candidates=items,
        task_embedding=[1.0] + [0.0] * 1535,
        active_misconceptions=set(),
        prerequisites=set(),
        recent_ids=set(),
        reuse_counts={},
        budget=300,
    )
    assert isinstance(decision, RoutingDecision)
    assert len(decision.selected) == 3   # 3 * 100 tokens fits in 300
    assert len(decision.dropped) == 2
    # Item 0 should be top-ranked (relevance)
    assert decision.selected[0].id == "0"

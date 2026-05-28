"""Selector engine treats XTrace recall hits as first-class candidates."""

from learning_memory_os.memory.xtrace import XTraceMemoryItem, xtrace_to_memory_item
from learning_memory_os.schemas.memory import MemoryItem
from learning_memory_os.selector.engine import RoutingEngine


def _semantic_item(*, id: str, title: str, body: str, embedding: list[float], tokens: int = 50) -> MemoryItem:
    return MemoryItem(
        id=id,
        tier="semantic",
        title=title,
        body=body,
        token_estimate=tokens,
        embedding=embedding,
    )


def test_xtrace_hit_converts_to_memory_item_with_xtrace_tier():
    hit = XTraceMemoryItem(
        id="mem_42",
        type="fact",
        text="The student previously asked about KV cache eviction policies.",
        score=0.83,
        user_id="alice",
        conv_id="conv_x",
        metadata={},
    )
    item = xtrace_to_memory_item(hit)
    assert isinstance(item, MemoryItem)
    assert item.tier == "xtrace"
    assert item.id == "xtrace:mem_42"
    assert "KV cache eviction" in item.body
    # XTrace similarity is preserved on the item for scoring.
    assert item.metadata["xtrace_similarity"] == 0.83


def test_xtrace_item_competes_for_budget_alongside_semantic_items():
    engine = RoutingEngine()

    # Two semantic candidates with semi-meaningful embeddings.
    sem_a = _semantic_item(
        id="sem_a",
        title="KV cache 101",
        body="KV caches store attention K/V tensors.",
        embedding=[1.0, 0.0],
        tokens=40,
    )
    sem_b = _semantic_item(
        id="sem_b",
        title="Quantization 101",
        body="Quantization reduces numerical precision.",
        embedding=[0.0, 1.0],
        tokens=40,
    )

    hit = XTraceMemoryItem(
        id="mem_99",
        type="fact",
        text="Student is implementing a paged KV cache.",
        score=0.95,
        user_id="alice",
        conv_id="conv_x",
        metadata={},
    )
    xtrace_item = xtrace_to_memory_item(hit)
    # Token estimate should be reasonable from the body length.
    assert xtrace_item.token_estimate > 0

    decision = engine.route(
        candidates=[sem_a, sem_b, xtrace_item],
        task_embedding=[1.0, 0.0],
        active_misconceptions=set(),
        prerequisites=set(),
        recent_ids=set(),
        reuse_counts={},
        budget=1000,  # enough to fit everything; we're testing presence + scoring
    )

    selected_ids = {it.id for it in decision.selected}
    assert "xtrace:mem_99" in selected_ids
    # The XTrace item should be scored (entry in scores dict).
    assert "xtrace:mem_99" in decision.scores
    # Its relevance score should come from the XTrace similarity, not from
    # cosine against the embedding (which is empty for XTrace items).
    assert decision.scores["xtrace:mem_99"].relevance == 0.95


def test_xtrace_item_dropped_when_budget_too_tight():
    engine = RoutingEngine()

    # Make the semantic item easy to fit; XTrace item has very high token cost.
    sem = _semantic_item(
        id="sem_a",
        title="Topic A",
        body="A",
        embedding=[1.0, 0.0],
        tokens=10,
    )
    hit = XTraceMemoryItem(
        id="mem_big",
        type="episode",
        text="x" * 4000,  # forces a big token estimate
        score=0.5,
        user_id="alice",
        conv_id="conv_x",
        metadata={},
    )
    big = xtrace_to_memory_item(hit)
    assert big.token_estimate > 100

    decision = engine.route(
        candidates=[sem, big],
        task_embedding=[1.0, 0.0],
        active_misconceptions=set(),
        prerequisites=set(),
        recent_ids=set(),
        reuse_counts={},
        budget=50,
    )

    selected_ids = {it.id for it in decision.selected}
    dropped_ids = {it.id for it in decision.dropped}
    assert "sem_a" in selected_ids
    assert "xtrace:mem_big" in dropped_ids

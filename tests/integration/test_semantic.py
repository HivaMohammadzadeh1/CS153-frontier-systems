from learning_memory_os.memory.semantic import SemanticStore
from learning_memory_os.schemas.artifacts import Concept
from learning_memory_os.schemas.memory import MemoryItem


def test_insert_and_retrieve_by_topic(db_conn):
    store = SemanticStore(db_conn)
    # Use a test-scoped topic_id so real ingested data doesn't interfere
    test_topic = "test_insert_retrieve_isolated"
    c = Concept(
        topic_id=test_topic,
        title="KV cache",
        definition="def",
        deep_explanation="more",
        prerequisites=[],
    )
    item = MemoryItem.from_artifact(c, embedding=[0.0] * 1536)
    store.insert(item)

    results = store.by_topic(test_topic)
    assert len(results) == 1
    assert results[0].title == "KV cache"


def test_by_topic_returns_embeddings(db_conn):
    """Regression: by_topic must return the embedding column, not an empty list."""
    from learning_memory_os.memory.semantic import SemanticStore
    from learning_memory_os.schemas.artifacts import Concept
    from learning_memory_os.schemas.memory import MemoryItem

    store = SemanticStore(db_conn)
    c = Concept(
        topic_id="test_by_topic_emb",
        title="Embedded",
        definition="def",
        deep_explanation="more",
        prerequisites=[],
    )
    vec = [0.0, 0.1, 0.2] + [0.0] * 1533  # length 1536
    store.insert(MemoryItem.from_artifact(c, embedding=vec))
    rows = store.by_topic("test_by_topic_emb")
    assert len(rows) == 1
    assert len(rows[0].embedding) == 1536
    # Round-trip to ~4 decimals (pgvector text format has limited precision)
    assert abs(rows[0].embedding[1] - 0.1) < 1e-4


def test_vector_search_returns_closest(db_conn):
    store = SemanticStore(db_conn)
    # Three vectors pointing in distinct directions
    vecs = [
        [1.0, 0.0, 0.0] + [0.0] * 1533,   # item 0
        [0.0, 1.0, 0.0] + [0.0] * 1533,   # item 1
        [0.0, 0.0, 1.0] + [0.0] * 1533,   # item 2
    ]
    items = [
        MemoryItem(
            id=f"x:{i}",
            tier="semantic",
            topic_id="t",
            title=f"item {i}",
            body=f"body {i}",
            token_estimate=10,
            embedding=vecs[i],
        )
        for i in range(3)
    ]
    for it in items:
        store.insert(it)

    # Query close to item 2 (third basis vector).
    # Use k=20 to ensure our test items surface despite the IVFFLAT approximate
    # index probing only a subset of lists across the production data.
    hits = store.vector_search(query=[0.0, 0.0, 1.0] + [0.0] * 1533, k=20)
    assert len(hits) >= 1
    assert hits[0].title == "item 2"

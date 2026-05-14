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


def test_vector_search_returns_closest(db_conn):
    store = SemanticStore(db_conn)
    items = [
        MemoryItem(
            id=f"x:{i}",
            tier="semantic",
            topic_id="t",
            title=f"item {i}",
            body=f"body {i}",
            token_estimate=10,
            embedding=[float(i)] + [0.0] * 1535,
        )
        for i in range(3)
    ]
    for it in items:
        store.insert(it)

    # query vector close to item 2
    hits = store.vector_search(query=[2.0] + [0.0] * 1535, k=2)
    assert len(hits) == 2
    assert hits[0].title == "item 2"

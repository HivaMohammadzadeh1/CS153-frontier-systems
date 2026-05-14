from learning_memory_os.memory.semantic import SemanticStore
from learning_memory_os.schemas.artifacts import Concept
from learning_memory_os.schemas.memory import MemoryItem


def test_count_by_topic_starts_at_zero(db_conn):
    store = SemanticStore(db_conn)
    assert store.count_by_topic("nonexistent_topic_abc") == 0


def test_count_after_inserts(db_conn):
    store = SemanticStore(db_conn)
    topic = "test_count_isolated"
    for i in range(3):
        c = Concept(
            topic_id=topic,
            title=f"C{i}",
            definition=f"def {i}",
            deep_explanation="",
            prerequisites=[],
        )
        store.insert(MemoryItem.from_artifact(c, embedding=[0.0] * 1536, item_id=f"sem:{topic}:{i}"))
    assert store.count_by_topic(topic) == 3


def test_delete_by_topic_clears_rows(db_conn):
    store = SemanticStore(db_conn)
    topic = "test_delete_isolated"
    for i in range(2):
        c = Concept(
            topic_id=topic,
            title=f"C{i}",
            definition=f"def {i}",
            deep_explanation="",
            prerequisites=[],
        )
        store.insert(MemoryItem.from_artifact(c, embedding=[0.0] * 1536, item_id=f"sem:{topic}:del{i}"))
    assert store.count_by_topic(topic) == 2

    deleted = store.delete_by_topic(topic)
    assert deleted == 2
    assert store.count_by_topic(topic) == 0

from learning_memory_os.schemas.artifacts import (
    Concept,
    Misconception,
    ArtifactType,
)
from learning_memory_os.schemas.memory import (
    MemoryItem,
    MasteryEntry,
)


def test_concept_round_trip():
    c = Concept(
        topic_id="kv_cache",
        title="KV cache",
        definition="A cache of past attention K and V tensors.",
        deep_explanation="Long form explanation.",
        prerequisites=[],
    )
    assert c.artifact_type == ArtifactType.CONCEPT
    assert c.model_dump()["topic_id"] == "kv_cache"


def test_misconception_has_correction():
    m = Misconception(
        topic_id="kv_cache",
        statement="KV cache stores raw token ids.",
        correction="It stores K and V tensors of past tokens.",
    )
    assert m.artifact_type == ArtifactType.MISCONCEPTION


def test_memory_item_from_artifact():
    c = Concept(
        topic_id="kv_cache",
        title="KV cache",
        definition="A cache.",
        deep_explanation="More.",
        prerequisites=[],
    )
    item = MemoryItem.from_artifact(c, embedding=[0.0] * 1536)
    assert item.tier == "semantic"
    assert item.title == "KV cache"
    assert len(item.embedding) == 1536


def test_mastery_entry_bounds():
    m = MasteryEntry(student_id="hiva", concept_id="abc", score=0.7, confidence=0.5)
    assert 0.0 <= m.score <= 1.0

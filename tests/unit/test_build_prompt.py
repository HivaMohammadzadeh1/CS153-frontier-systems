"""TutorAgent.build_prompt — shared by /api/chat and /api/chat/stream."""

from learning_memory_os.agents.tutor import TUTOR_SYSTEM, TutorAgent
from learning_memory_os.schemas.memory import MemoryItem


def _item(item_id, title, body):
    return MemoryItem(id=item_id, tier="semantic", topic_id="t", title=title, body=body, token_estimate=10)


def test_build_prompt_includes_context_and_profile():
    items = [_item("abc123", "KV Cache", "stores keys/values"),
             _item("def456", "PagedAttention", "paged blocks")]
    system, user = TutorAgent.build_prompt(
        "How does the KV cache work?", items,
        weak_concepts=["Attention"], strong_concepts=["Tokenization"],
        active_misconception_texts=["cache stores tokens"], due_concepts=["RoPE"],
    )
    assert system == TUTOR_SYSTEM
    assert "[abc123] KV Cache" in user and "stores keys/values" in user
    assert "STUDENT PROFILE" in user
    assert "Mastery is LOW for: Attention" in user
    assert "Mastery is HIGH for: Tokenization" in user
    assert "cache stores tokens" in user
    assert "Due for review" in user and "RoPE" in user
    assert "STUDENT QUESTION:\nHow does the KV cache work?" in user


def test_build_prompt_omits_profile_block_when_empty():
    _system, user = TutorAgent.build_prompt("q", [_item("a", "A", "b")])
    assert "STUDENT PROFILE" not in user
    assert "CONTEXT ITEMS:" in user

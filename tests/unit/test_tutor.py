from unittest.mock import MagicMock, call
from learning_memory_os.schemas.memory import MemoryItem
from learning_memory_os.agents.tutor import TutorAgent


def _item(i, body):
    return MemoryItem(
        id=str(i),
        tier="semantic",
        topic_id="t",
        title=f"item-{i}",
        body=body,
        token_estimate=max(1, len(body) // 4),
        embedding=[0.1] * 1536,
    )


def test_tutor_calls_engine_then_llm():
    fake_llm = MagicMock()
    fake_llm.complete.return_value = "The KV cache is..."

    fake_engine = MagicMock()
    fake_engine.route.return_value = MagicMock(
        selected=[_item(1, "KV cache stores K and V."), _item(2, "Cached per-token.")],
        dropped=[],
        scores={},
        budget=1000,
        tokens_used=20,
    )

    fake_logger = MagicMock()
    fake_embedder = MagicMock()
    fake_embedder.embed_one.return_value = [0.1] * 1536

    tutor = TutorAgent(
        llm=fake_llm,
        engine=fake_engine,
        embedder=fake_embedder,
        logger=fake_logger,
    )

    out = tutor.answer(
        student_id="hiva",
        question="what is a KV cache?",
        candidates=[_item(1, "KV cache stores K and V."), _item(2, "Cached per-token.")],
        active_misconceptions=set(),
        prerequisites=set(),
        recent_ids=set(),
        reuse_counts={},
        budget=1000,
    )
    assert out.text == "The KV cache is..."
    fake_engine.route.assert_called_once()
    fake_llm.complete.assert_called_once()
    assert fake_logger.log.call_count >= 1


def test_tutor_uses_student_profile():
    """When weak_concepts is provided, the user prompt must include STUDENT PROFILE."""
    fake_llm = MagicMock()
    fake_llm.complete.return_value = "Here is the explanation."

    fake_engine = MagicMock()
    fake_engine.route.return_value = MagicMock(
        selected=[_item(1, "Concept body.")],
        dropped=[],
        scores={},
        budget=1000,
        tokens_used=10,
    )

    fake_logger = MagicMock()
    fake_embedder = MagicMock()
    fake_embedder.embed_one.return_value = [0.1] * 1536

    tutor = TutorAgent(
        llm=fake_llm,
        engine=fake_engine,
        embedder=fake_embedder,
        logger=fake_logger,
    )

    tutor.answer(
        student_id="test-student",
        question="What is tensor parallelism?",
        candidates=[_item(1, "Concept body.")],
        active_misconceptions=set(),
        prerequisites=set(),
        recent_ids=set(),
        reuse_counts={},
        budget=1000,
        weak_concepts=["attention mechanism", "softmax scaling"],
        strong_concepts=["matrix multiplication"],
        active_misconception_texts=["Thinks batch size affects model parallelism"],
    )

    # Verify the user prompt passed to llm.complete contains STUDENT PROFILE
    assert fake_llm.complete.call_count == 1
    _, kwargs = fake_llm.complete.call_args
    user_prompt = kwargs.get("user", "")
    assert "STUDENT PROFILE" in user_prompt
    assert "attention mechanism" in user_prompt
    assert "matrix multiplication" in user_prompt
    assert "Thinks batch size" in user_prompt


def test_tutor_no_profile_when_none():
    """When no profile kwargs provided, STUDENT PROFILE block is absent."""
    fake_llm = MagicMock()
    fake_llm.complete.return_value = "Plain answer."

    fake_engine = MagicMock()
    fake_engine.route.return_value = MagicMock(
        selected=[_item(1, "Body.")],
        dropped=[],
        scores={},
        budget=1000,
        tokens_used=5,
    )

    fake_logger = MagicMock()
    fake_embedder = MagicMock()
    fake_embedder.embed_one.return_value = [0.1] * 1536

    tutor = TutorAgent(
        llm=fake_llm,
        engine=fake_engine,
        embedder=fake_embedder,
        logger=fake_logger,
    )

    tutor.answer(
        student_id="anon",
        question="What is gradient checkpointing?",
        candidates=[_item(1, "Body.")],
        active_misconceptions=set(),
        prerequisites=set(),
        recent_ids=set(),
        reuse_counts={},
        budget=1000,
        # no profile kwargs — should default to None
    )

    _, kwargs = fake_llm.complete.call_args
    user_prompt = kwargs.get("user", "")
    assert "STUDENT PROFILE" not in user_prompt

"""Regression tests for the misconception + review-due scoring signals.

Before the fix, the misconception boost compared a semantic-item id against a
set of *misconception-row ids*, so it never fired. These pin the intended
behavior: exact concept match, topic fallback, and the spaced-repetition signal.
"""

from learning_memory_os.schemas.memory import MemoryItem
from learning_memory_os.selector.scoring import ScoringContext, score_item


def _item(item_id, topic="kv_cache"):
    return MemoryItem(
        id=item_id, tier="semantic", topic_id=topic, title=item_id,
        body="b", token_estimate=1, embedding=[0.0] * 4,
    )


def test_exact_concept_match_full_boost():
    ctx = ScoringContext(task_embedding=[0.0] * 4, misconception_concept_ids={"concept-123"})
    assert score_item(_item("concept-123"), ctx).misconception == 1.0


def test_same_topic_partial_boost():
    ctx = ScoringContext(
        task_embedding=[0.0] * 4,
        misconception_concept_ids={"concept-123"},
        misconception_topics={"kv_cache"},
    )
    assert score_item(_item("other", topic="kv_cache"), ctx).misconception == 0.4


def test_unrelated_no_boost():
    ctx = ScoringContext(
        task_embedding=[0.0] * 4,
        misconception_concept_ids={"concept-123"},
        misconception_topics={"kv_cache"},
    )
    assert score_item(_item("x", topic="tokenization"), ctx).misconception == 0.0


def test_review_due_signal():
    ctx = ScoringContext(task_embedding=[0.0] * 4, due_concept_ids={"concept-9"})
    s = score_item(_item("concept-9"), ctx)
    assert s.review_due == 1.0
    assert s.total > 0

from datetime import datetime, timedelta, timezone
from learning_memory_os.schemas.memory import MemoryItem
from learning_memory_os.selector.scoring import (
    ScoringContext,
    score_item,
)


def _item(item_id: str, body: str, *, embedding=None, tier="semantic", topic="t"):
    return MemoryItem(
        id=item_id,
        tier=tier,
        topic_id=topic,
        title=item_id,
        body=body,
        token_estimate=max(1, len(body) // 4),
        embedding=embedding or [0.0] * 1536,
        created_at=datetime.now(timezone.utc),
    )


def test_relevance_dominates_when_other_signals_zero():
    a = _item("a", "kv cache", embedding=[1.0, 0.0] + [0.0] * 1534)
    b = _item("b", "tokenization", embedding=[0.0, 1.0] + [0.0] * 1534)
    ctx = ScoringContext(
        task_embedding=[1.0, 0.0] + [0.0] * 1534,
        misconception_concept_ids=set(),
        prerequisite_titles=set(),
        recent_item_ids=set(),
        reuse_counts={},
    )
    sa = score_item(a, ctx)
    sb = score_item(b, ctx)
    assert sa.total > sb.total


def test_misconception_boost_applied():
    a = _item("misc:wrong-kv", "KV cache stores token ids: misconception", tier="student")
    ctx = ScoringContext(
        task_embedding=[0.0] * 1536,
        misconception_concept_ids={"misc:wrong-kv"},
        prerequisite_titles=set(),
        recent_item_ids=set(),
        reuse_counts={},
    )
    s = score_item(a, ctx)
    assert s.misconception > 0
    assert s.total > 0


def test_recency_decays():
    now = datetime.now(timezone.utc)
    old = _item("old", "x", tier="episodic")
    old.created_at = now - timedelta(days=10)
    new = _item("new", "x", tier="episodic")
    new.created_at = now - timedelta(hours=1)

    ctx = ScoringContext(
        task_embedding=[0.0] * 1536,
        misconception_concept_ids=set(),
        prerequisite_titles=set(),
        recent_item_ids=set(),
        reuse_counts={},
    )
    so = score_item(old, ctx)
    sn = score_item(new, ctx)
    assert sn.recency > so.recency

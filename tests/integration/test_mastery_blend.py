"""Confidence-weighted mastery blending + SM-2 review scheduling (Track A3 / B2)."""

from learning_memory_os.memory.student import StudentStore


def _concept(db_conn, topic="t"):
    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO semantic_items (topic_id, artifact_type, title, body) "
            "VALUES (%s, 'concept', 'C', 'b') RETURNING id::text",
            (topic,),
        )
        return cur.fetchone()["id"]


def test_blend_does_not_let_one_noisy_quiz_overwrite(db_conn, fresh_student_id):
    store = StudentStore(db_conn)
    cid = _concept(db_conn)
    store.update_mastery(fresh_student_id, cid, score=0.9, confidence=0.8)
    store.update_mastery(fresh_student_id, cid, score=0.1, confidence=0.2)  # noisy low
    e = store.mastery_for(fresh_student_id)[0]
    # (0.8*0.9 + 0.2*0.1) / (0.8+0.2) = 0.74 — stays near the confident prior.
    assert 0.6 < e.score < 0.85
    assert e.confidence > 0.8  # confidence accrues, never wiped


def test_due_for_review_filters_by_schedule(db_conn, fresh_student_id):
    store = StudentStore(db_conn)
    cid = _concept(db_conn)
    store.update_mastery(fresh_student_id, cid, score=0.9, confidence=0.8)
    assert cid not in store.due_for_review(fresh_student_id)  # scheduled in future
    with db_conn.cursor() as cur:
        cur.execute(
            "UPDATE mastery SET next_review_at = now() - interval '1 day' "
            "WHERE student_id = %s AND concept_id::text = %s",
            (fresh_student_id, cid),
        )
    assert cid in store.due_for_review(fresh_student_id)


def test_passing_advances_then_failing_resets_schedule(db_conn, fresh_student_id):
    store = StudentStore(db_conn)
    cid = _concept(db_conn)
    store.update_mastery(fresh_student_id, cid, score=0.8, confidence=0.7)  # insert
    store.update_mastery(fresh_student_id, cid, score=0.8, confidence=0.7)  # pass -> reps up
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT reps FROM mastery WHERE student_id=%s AND concept_id::text=%s",
            (fresh_student_id, cid),
        )
        assert cur.fetchone()["reps"] >= 1
    store.update_mastery(fresh_student_id, cid, score=0.2, confidence=0.7)  # fail -> reset
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT reps, interval_days FROM mastery WHERE student_id=%s AND concept_id::text=%s",
            (fresh_student_id, cid),
        )
        row = cur.fetchone()
        assert row["reps"] == 0
        assert row["interval_days"] == 1


def test_misconception_carries_topic(db_conn, fresh_student_id):
    store = StudentStore(db_conn)
    store.record_misconception(
        fresh_student_id,
        concept_id=None,
        description="KV cache stores token ids",
        topic_id="kv_cache",
    )
    active = store.active_misconceptions(fresh_student_id)
    assert active[0]["topic_id"] == "kv_cache"

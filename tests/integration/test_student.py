from learning_memory_os.memory.student import StudentStore


def test_set_and_get_mastery(db_conn, fresh_student_id):
    store = StudentStore(db_conn)
    # Need a concept first
    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO semantic_items (topic_id, artifact_type, title, body) "
            "VALUES ('t', 'concept', 'C', 'b') RETURNING id::text"
        )
        concept_id = cur.fetchone()["id"]

    store.update_mastery(fresh_student_id, concept_id, score=0.7, confidence=0.6)
    entries = store.mastery_for(fresh_student_id)
    assert len(entries) == 1
    assert entries[0].score == 0.7


def test_record_misconception(db_conn, fresh_student_id):
    store = StudentStore(db_conn)
    mid = store.record_misconception(
        fresh_student_id,
        concept_id=None,
        description="KV cache stores token ids",
        evidence="quiz answer",
    )
    active = store.active_misconceptions(fresh_student_id)
    assert len(active) == 1
    assert active[0]["description"] == "KV cache stores token ids"

    store.resolve_misconception(mid)
    assert store.active_misconceptions(fresh_student_id) == []

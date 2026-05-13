from learning_memory_os.memory.intervention import InterventionStore


def test_record_and_list(db_conn, fresh_student_id):
    store = InterventionStore(db_conn)
    store.record(
        student_id=fresh_student_id,
        misconception_id=None,
        strategy="worked_example",
        outcome="helped",
        notes="student answered correctly after",
    )
    records = store.for_student(fresh_student_id)
    assert len(records) == 1
    assert records[0]["strategy"] == "worked_example"
    assert records[0]["outcome"] == "helped"

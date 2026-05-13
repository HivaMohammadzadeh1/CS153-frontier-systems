from learning_memory_os.memory.episodic import EpisodicStore


def test_append_and_recent(db_conn, fresh_student_id):
    store = EpisodicStore(db_conn)
    store.append(
        student_id=fresh_student_id,
        event_type="question",
        payload={"text": "what is KV cache?"},
        embedding=[0.1] * 1536,
    )
    store.append(
        student_id=fresh_student_id,
        event_type="tutor_reply",
        payload={"text": "it caches K and V tensors..."},
        embedding=[0.2] * 1536,
    )
    recent = store.recent(fresh_student_id, limit=5)
    assert len(recent) == 2
    # Most recent first
    assert recent[0].event_type == "tutor_reply"

"""LearnerProfile builder — the per-user adaptation snapshot."""

from learning_memory_os.agents.profile import build_profile
from learning_memory_os.memory.student import StudentStore


def _concept(db_conn, title, topic="t"):
    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO semantic_items (topic_id, artifact_type, title, body) "
            "VALUES (%s, 'concept', %s, 'b') RETURNING id::text",
            (topic, title),
        )
        return cur.fetchone()["id"]


def test_profile_classifies_strength_and_weakness(db_conn, fresh_student_id):
    store = StudentStore(db_conn)
    strong = _concept(db_conn, "Strong Concept")
    weak = _concept(db_conn, "Weak Concept")
    store.update_mastery(fresh_student_id, strong, score=0.85, confidence=0.6)
    store.update_mastery(fresh_student_id, weak, score=0.2, confidence=0.6)

    p = build_profile(db_conn, fresh_student_id)
    assert "Strong Concept" in p.strengths
    assert "Weak Concept" in p.weaknesses
    assert 0.0 < p.overall_mastery < 1.0


def test_profile_includes_misconceptions(db_conn, fresh_student_id):
    store = StudentStore(db_conn)
    store.record_misconception(
        fresh_student_id, concept_id=None,
        description="KV cache stores token ids", topic_id="kv_cache",
    )
    p = build_profile(db_conn, fresh_student_id)
    assert any("KV cache" in m for m in p.misconceptions)
    # prompt block renders the profile for the tutor
    assert "STUDENT PROFILE" in p.prompt_block()

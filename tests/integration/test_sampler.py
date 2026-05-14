from learning_memory_os.trajectories.sampler import (
    sample_candidate_pool,
    sample_student_state,
)


def test_sample_candidate_pool_returns_items(db_conn):
    pool = sample_candidate_pool(
        db_conn, target_topic="kv_cache", pool_size=10
    )
    assert 1 <= len(pool) <= 10
    assert all(p.id and p.title and p.body_excerpt for p in pool)
    # All ids should be 8-char hex (short form)
    assert all(len(p.id) == 8 for p in pool)


def test_sample_student_state_returns_realistic_shape(db_conn):
    state = sample_student_state(
        db_conn, student_id="synthetic-1", target_concepts=["kv_cache", "quantization"]
    )
    assert state.student_id == "synthetic-1"
    for v in state.mastery.values():
        assert 0.0 <= v <= 1.0

from learning_memory_os.trajectories.schemas import (
    StudentState,
    Trajectory,
    PoolItem,
    TaskType,
)


def test_pool_item_minimal():
    p = PoolItem(id="abc12345", title="KV cache", body_excerpt="A cache of K and V.", token_estimate=50)
    assert p.id == "abc12345"


def test_trajectory_round_trip():
    state = StudentState(
        student_id="s1",
        mastery={"kv_cache": 0.3, "tokenization": 0.8},
        active_misconceptions=["KV cache stores token ids"],
        recent_episodic_ids=["ev1", "ev2"],
    )
    t = Trajectory(
        id="traj-0001",
        student_state=state,
        task_type=TaskType.EXPLAIN,
        task_text="What is the KV cache?",
        budget=2000,
        candidate_pool=[
            PoolItem(id="aaaa1111", title="A", body_excerpt="x", token_estimate=100),
            PoolItem(id="bbbb2222", title="B", body_excerpt="y", token_estimate=100),
        ],
        oracle_selection=["aaaa1111"],
    )
    serialized = t.model_dump()
    assert serialized["id"] == "traj-0001"
    assert serialized["task_type"] == "explain"
    assert serialized["oracle_selection"] == ["aaaa1111"]

from learning_memory_os.trajectories.schemas import (
    StudentState, PoolItem, TaskType, Trajectory,
)
from learning_memory_os.trajectories.serializer import trajectory_to_training_pair


def test_trajectory_to_training_pair_produces_input_and_target():
    t = Trajectory(
        id="t1",
        student_state=StudentState(student_id="s", mastery={}, active_misconceptions=[], recent_episodic_ids=[]),
        task_type=TaskType.EXPLAIN,
        task_text="explain A",
        budget=300,
        candidate_pool=[
            PoolItem(id="aaaa1111", title="A", body_excerpt="x", token_estimate=100),
            PoolItem(id="bbbb2222", title="B", body_excerpt="y", token_estimate=100),
        ],
        oracle_selection=["aaaa1111"],
    )
    pair = trajectory_to_training_pair(t)
    assert "aaaa1111" in pair["input"]
    assert pair["target"] == "aaaa1111"


def test_trajectory_to_training_pair_handles_empty_selection():
    t = Trajectory(
        id="t2",
        student_state=StudentState(student_id="s", mastery={}, active_misconceptions=[], recent_episodic_ids=[]),
        task_type=TaskType.QUIZ,
        task_text="quiz",
        budget=300,
        candidate_pool=[PoolItem(id="aaaa1111", title="A", body_excerpt="x", token_estimate=100)],
        oracle_selection=[],
    )
    pair = trajectory_to_training_pair(t)
    assert pair["target"] == ""

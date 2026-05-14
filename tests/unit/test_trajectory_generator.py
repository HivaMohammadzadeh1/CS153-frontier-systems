from unittest.mock import MagicMock
from learning_memory_os.trajectories.schemas import (
    StudentState,
    PoolItem,
    TaskType,
    Trajectory,
)
from learning_memory_os.trajectories.generator import build_trajectory


def test_build_trajectory_calls_oracle_and_packs_result():
    fake_llm = MagicMock()
    fake_llm.complete_json.return_value = {
        "selected_ids": ["aaaa1111", "cccc3333"],
        "rationale": "These two items directly explain the KV cache.",
    }

    state = StudentState(student_id="s1", mastery={}, active_misconceptions=[], recent_episodic_ids=[])
    pool = [
        PoolItem(id="aaaa1111", title="A", body_excerpt="kv cache stores K and V", token_estimate=100),
        PoolItem(id="bbbb2222", title="B", body_excerpt="unrelated topic", token_estimate=100),
        PoolItem(id="cccc3333", title="C", body_excerpt="why kv cache exists", token_estimate=100),
    ]

    t = build_trajectory(
        traj_id="traj-1",
        student_state=state,
        task_type=TaskType.EXPLAIN,
        task_text="What is a KV cache?",
        budget=300,
        candidate_pool=pool,
        oracle_llm=fake_llm,
    )
    assert isinstance(t, Trajectory)
    assert t.oracle_selection == ["aaaa1111", "cccc3333"]
    fake_llm.complete_json.assert_called_once()


def test_build_trajectory_filters_hallucinated_ids():
    fake_llm = MagicMock()
    fake_llm.complete_json.return_value = {
        "selected_ids": ["aaaa1111", "zzz_not_in_pool"],
        "rationale": "...",
    }
    state = StudentState(student_id="s", mastery={}, active_misconceptions=[], recent_episodic_ids=[])
    pool = [PoolItem(id="aaaa1111", title="A", body_excerpt="x", token_estimate=100)]
    t = build_trajectory(
        traj_id="t",
        student_state=state,
        task_type=TaskType.EXPLAIN,
        task_text="explain A",
        budget=200,
        candidate_pool=pool,
        oracle_llm=fake_llm,
    )
    assert t.oracle_selection == ["aaaa1111"]

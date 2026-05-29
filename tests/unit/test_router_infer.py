from unittest.mock import MagicMock

from learning_memory_os.trajectories.schemas import StudentState, PoolItem, TaskType
from learning_memory_os.router.frontier_api import FrontierAPIRouter


def _pool():
    return [
        PoolItem(id="aaaa1111", title="A", body_excerpt="x", token_estimate=100),
        PoolItem(id="bbbb2222", title="B", body_excerpt="y", token_estimate=100),
    ]


def test_frontier_api_router_parses_llm_ids():
    fake_llm = MagicMock()
    fake_llm.complete.return_value = "aaaa1111,bbbb2222"
    r = FrontierAPIRouter(fake_llm)
    state = StudentState(student_id="s", mastery={}, active_misconceptions=[], recent_episodic_ids=[])
    out = r.route(
        student_state=state,
        task_type=TaskType.EXPLAIN,
        task_text="explain A",
        budget=300,
        candidate_pool=_pool(),
    )
    assert out == ["aaaa1111", "bbbb2222"]
    fake_llm.complete.assert_called_once()


def test_frontier_api_router_handles_empty_selection():
    fake_llm = MagicMock()
    fake_llm.complete.return_value = "none"
    r = FrontierAPIRouter(fake_llm)
    state = StudentState(student_id="s", mastery={}, active_misconceptions=[], recent_episodic_ids=[])
    out = r.route(
        student_state=state,
        task_type=TaskType.QUIZ,
        task_text="quiz",
        budget=300,
        candidate_pool=_pool(),
    )
    assert out == []

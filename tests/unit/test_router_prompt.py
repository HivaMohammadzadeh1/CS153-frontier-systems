from learning_memory_os.trajectories.schemas import StudentState, PoolItem, TaskType
from learning_memory_os.router.prompt import (
    format_router_input,
    parse_router_output,
)


def test_format_router_input_includes_all_sections():
    state = StudentState(student_id="s", mastery={"a": 0.2}, active_misconceptions=["x"], recent_episodic_ids=[])
    pool = [
        PoolItem(id="aaaa1111", title="A", body_excerpt="x", token_estimate=50),
        PoolItem(id="bbbb2222", title="B", body_excerpt="y", token_estimate=80),
    ]
    text = format_router_input(
        student_state=state,
        task_type=TaskType.EXPLAIN,
        task_text="explain A",
        budget=500,
        candidate_pool=pool,
    )
    assert "STUDENT" in text
    assert "TASK" in text
    assert "POOL" in text
    assert "aaaa1111" in text
    assert "bbbb2222" in text
    assert "500" in text


def test_parse_router_output_extracts_ids():
    assert parse_router_output("aaaa1111,bbbb2222") == ["aaaa1111", "bbbb2222"]
    assert parse_router_output("  aaaa1111, bbbb2222  \n") == ["aaaa1111", "bbbb2222"]
    assert parse_router_output("[aaaa1111, bbbb2222]") == ["aaaa1111", "bbbb2222"]
    assert parse_router_output("") == []
    assert parse_router_output("none") == []

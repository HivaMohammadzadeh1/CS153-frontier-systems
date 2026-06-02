"""TraceStore: capture, reward, export to the fine-tune Trajectory schema."""

from learning_memory_os.memory.trace import TraceStore
from learning_memory_os.trajectories.schemas import Trajectory


def _pool():
    return [
        {"id": "a", "title": "A", "body_excerpt": "x", "token_estimate": 10},
        {"id": "b", "title": "B", "body_excerpt": "y", "token_estimate": 20},
    ]


def test_record_and_export_round_trip(db_conn, fresh_student_id):
    ts = TraceStore(db_conn)
    ts.record_turn(
        student_id=fresh_student_id,
        task_text="What is the KV cache?",
        budget=3000,
        student_state={"mastery": {"c1": 0.5}, "active_misconceptions": ["foo"], "recent_episodic_ids": ["e1"]},
        candidate_pool=_pool(),
        selected_ids=["a"],
        dropped_ids=["b"],
        scores={"a": 1.2, "b": 0.3},
        reply="an answer",
        model="claude",
    )
    assert ts.count(fresh_student_id) == 1
    trajs = ts.export_trajectories(fresh_student_id)
    assert len(trajs) == 1 and isinstance(trajs[0], Trajectory)
    t = trajs[0]
    assert t.task_text == "What is the KV cache?"
    assert t.oracle_selection == ["a"]
    assert [p.id for p in t.candidate_pool] == ["a", "b"]
    assert t.student_state.mastery == {"c1": 0.5}


def test_attach_reward_targets_latest_and_min_reward_filters(db_conn, fresh_student_id):
    ts = TraceStore(db_conn)
    ts.record_turn(student_id=fresh_student_id, task_text="q1", budget=1000,
                   student_state={}, candidate_pool=[], selected_ids=[])
    ts.attach_reward(fresh_student_id, 0.9)   # labels q1 (latest at this point)
    ts.record_turn(student_id=fresh_student_id, task_text="q2", budget=1000,
                   student_state={}, candidate_pool=[], selected_ids=[])
    ts.attach_reward(fresh_student_id, 0.1)   # labels q2
    high = ts.export_trajectories(fresh_student_id, min_reward=0.5)
    assert [t.task_text for t in high] == ["q1"]


def test_export_all_users_for_finetuning(db_conn):
    """export_trajectories(None) aggregates every user's turns for fine-tuning."""
    import uuid
    ts = TraceStore(db_conn)
    a, b = f"u-{uuid.uuid4().hex[:6]}", f"u-{uuid.uuid4().hex[:6]}"
    for sid in (a, b):
        with db_conn.cursor() as cur:
            cur.execute("INSERT INTO students (id) VALUES (%s) ON CONFLICT DO NOTHING", (sid,))
        ts.record_turn(student_id=sid, task_text=f"q-{sid}", budget=1000,
                       student_state={}, candidate_pool=[], selected_ids=[])
    all_tasks = {t.task_text for t in ts.export_trajectories(None)}
    assert f"q-{a}" in all_tasks and f"q-{b}" in all_tasks


def test_delete_for_student(db_conn, fresh_student_id):
    ts = TraceStore(db_conn)
    ts.record_turn(student_id=fresh_student_id, task_text="q", budget=1000,
                   student_state={}, candidate_pool=[], selected_ids=[])
    assert ts.count(fresh_student_id) == 1
    assert ts.delete_for_student(fresh_student_id) == 1
    assert ts.count(fresh_student_id) == 0

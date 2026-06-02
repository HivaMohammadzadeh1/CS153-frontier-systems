"""API tests for the per-user profile + trace-capture/export endpoints."""

import json
import os
import uuid

import psycopg
from psycopg.rows import dict_row
from fastapi.testclient import TestClient

from learning_memory_os.api import app
from learning_memory_os.memory.student import StudentStore
from learning_memory_os.memory.trace import TraceStore

DB_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://lmos:lmos_dev@localhost:5433/learning_memory_os",
)


def _authed():
    """A TestClient with a session cookie; returns (client, username)."""
    username = f"tracetest-{uuid.uuid4().hex[:8]}"
    c = TestClient(app)
    r = c.post("/api/auth/signup", json={
        "username": username, "email": f"{username}@example.com", "password": "pw123456"})
    assert r.status_code == 200, r.text
    return c, username


def _seed_trace(student_id: str, reply: str = "ans"):
    conn = psycopg.connect(DB_URL, row_factory=dict_row)
    try:
        StudentStore(conn).ensure_student(student_id)
        TraceStore(conn).record_turn(
            student_id=student_id, task_text="What is X?", budget=3000,
            student_state={"mastery": {}, "active_misconceptions": [], "recent_episodic_ids": []},
            candidate_pool=[{"id": "a", "title": "A", "body_excerpt": "x", "token_estimate": 10}],
            selected_ids=["a"], reply=reply, model="claude",
        )
        conn.commit()
    finally:
        conn.close()


def test_profile_endpoint_shape():
    c, u = _authed()
    r = c.get(f"/api/student/{u}/profile")
    assert r.status_code == 200
    d = r.json()
    for k in ("student_id", "overall_mastery", "strengths", "weaknesses",
              "misconceptions", "due_for_review"):
        assert k in d


def test_capture_summary_export_feedback_and_delete():
    c, sid = _authed()          # username == student_id; signup created the student row
    _seed_trace(sid)

    s = c.get(f"/api/student/{sid}/traces/summary").json()
    assert s["count"] == 1 and len(s["recent"]) == 1

    # router export round-trips to the Trajectory shape
    line = c.get(f"/api/student/{sid}/traces/export").text.strip().splitlines()[0]
    obj = json.loads(line)
    assert obj["task_text"] == "What is X?" and obj["oracle_selection"] == ["a"]

    # tutor export keeps the reply
    tline = c.get(f"/api/student/{sid}/traces/export?format=tutor").text.strip().splitlines()[0]
    assert json.loads(tline)["reply"] == "ans"

    # 👍 feedback attaches a reward to the captured turn
    fb = c.post("/api/feedback", json={
        "student_id": sid, "message_idx": 0, "rating": 1, "selected_item_ids": ["a"]})
    assert fb.status_code == 200
    assert c.get(f"/api/student/{sid}/traces/summary").json()["recent"][0]["reward"] == 1.0

    # consent control: delete clears everything
    assert c.delete(f"/api/student/{sid}/traces").json()["deleted"] == 1
    assert c.get(f"/api/student/{sid}/traces/summary").json()["count"] == 0

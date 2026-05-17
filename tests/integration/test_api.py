"""Smoke tests for the FastAPI endpoints. Doesn't require a running server — uses TestClient."""

from fastapi.testclient import TestClient
from learning_memory_os.api import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_list_topics():
    r = client.get("/api/topics")
    assert r.status_code == 200
    topics = r.json()
    assert len(topics) == 38
    assert all("id" in t and "title" in t and "area" in t for t in topics)


def test_student_state_creates_student():
    r = client.get("/api/student/api-test-user/state")
    assert r.status_code == 200
    payload = r.json()
    assert "mastery" in payload
    assert "misconceptions" in payload


def test_student_messages_returns_list():
    r = client.get("/api/student/restore-test/messages")
    assert r.status_code == 200
    payload = r.json()
    assert "messages" in payload
    assert isinstance(payload["messages"], list)


def test_student_progress_returns_shape():
    r = client.get("/api/student/progress-test-user/progress")
    assert r.status_code == 200
    data = r.json()
    assert "topics" in data and "misconceptions" in data


def test_feedback_endpoint_accepts_thumbs():
    r = client.post("/api/feedback", json={
        "student_id": "feedback-test",
        "message_idx": 0,
        "rating": 1,
        "selected_item_ids": ["aaaa1111"],
    })
    assert r.status_code == 200
    assert r.json() == {"ok": True}

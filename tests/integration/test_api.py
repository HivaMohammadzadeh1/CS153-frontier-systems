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


def test_create_and_list_conversations():
    # Create
    r = client.post("/api/conversations", json={"student_id": "conv-test"})
    assert r.status_code == 200
    payload = r.json()
    assert "id" in payload
    cid = payload["id"]

    # List
    r = client.get("/api/student/conv-test/conversations")
    assert r.status_code == 200
    convs = r.json()["conversations"]
    assert any(c["id"] == cid for c in convs)


def test_conversation_messages_endpoint():
    # Create a conversation
    r = client.post("/api/conversations", json={"student_id": "conv-msg-test", "title": "test"})
    cid = r.json()["id"]
    # Fetch its (empty) messages
    r = client.get(f"/api/conversations/{cid}/messages")
    assert r.status_code == 200
    data = r.json()
    assert data["conversation_id"] == cid
    assert data["messages"] == []


def test_delete_conversation():
    r = client.post("/api/conversations", json={"student_id": "conv-del-test"})
    cid = r.json()["id"]
    r = client.delete(f"/api/conversations/{cid}")
    assert r.status_code == 200
    # Verify it's gone
    r = client.get("/api/student/conv-del-test/conversations")
    convs = r.json()["conversations"]
    assert not any(c["id"] == cid for c in convs)

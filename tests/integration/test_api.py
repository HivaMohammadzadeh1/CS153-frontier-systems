"""Smoke tests for the FastAPI endpoints. Doesn't require a running server — uses TestClient."""

import uuid

from fastapi.testclient import TestClient

from learning_memory_os.api import app

client = TestClient(app)  # unauthenticated, for public routes


def _authed():
    """A TestClient with a fresh session cookie; returns (client, username)."""
    username = f"apitest-{uuid.uuid4().hex[:8]}"
    c = TestClient(app)
    r = c.post("/api/auth/signup", json={
        "username": username, "email": f"{username}@example.com", "password": "pw123456"})
    assert r.status_code == 200, r.text
    assert r.json()["username"] == username
    return c, username


# ── public routes ──────────────────────────────────────────────────────────
def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_list_topics():
    r = client.get("/api/topics")
    assert r.status_code == 200
    topics = r.json()
    assert len(topics) == 48


# ── auth gate ────────────────────────────────────────────────────────────────
def test_data_routes_require_auth():
    assert client.get("/api/student/whoever/state").status_code == 401
    assert client.get("/api/student/whoever/progress").status_code == 401


def test_ownership_enforced():
    c, _username = _authed()
    # logged in, but asking for someone else's data
    assert c.get("/api/student/some-other-user/state").status_code == 403


# ── authenticated data routes (scoped to the caller's own username) ──────────
def test_student_state_creates_student():
    c, u = _authed()
    r = c.get(f"/api/student/{u}/state")
    assert r.status_code == 200
    assert "mastery" in r.json() and "misconceptions" in r.json()


def test_student_messages_returns_list():
    c, u = _authed()
    r = c.get(f"/api/student/{u}/messages")
    assert r.status_code == 200
    assert isinstance(r.json()["messages"], list)


def test_student_progress_returns_shape():
    c, u = _authed()
    r = c.get(f"/api/student/{u}/progress")
    assert r.status_code == 200
    assert "topics" in r.json() and "misconceptions" in r.json()


def test_feedback_endpoint_accepts_thumbs():
    c, _u = _authed()
    r = c.post("/api/feedback", json={
        "student_id": "ignored-overridden-by-session",
        "message_idx": 0, "rating": 1, "selected_item_ids": ["aaaa1111"]})
    assert r.status_code == 200 and r.json() == {"ok": True}


def test_create_list_and_delete_conversations():
    c, u = _authed()
    cid = c.post("/api/conversations", json={"student_id": u}).json()["id"]
    convs = c.get(f"/api/student/{u}/conversations").json()["conversations"]
    assert any(cv["id"] == cid for cv in convs)

    msgs = c.get(f"/api/conversations/{cid}/messages").json()
    assert msgs["conversation_id"] == cid and msgs["messages"] == []

    assert c.delete(f"/api/conversations/{cid}").status_code == 200
    convs2 = c.get(f"/api/student/{u}/conversations").json()["conversations"]
    assert not any(cv["id"] == cid for cv in convs2)


def test_conversation_cross_user_forbidden():
    c1, u1 = _authed()
    cid = c1.post("/api/conversations", json={"student_id": u1}).json()["id"]
    c2, _u2 = _authed()  # a different logged-in user
    assert c2.get(f"/api/conversations/{cid}/messages").status_code == 403
    assert c2.delete(f"/api/conversations/{cid}").status_code == 403


def test_data_persists_across_logout_login():
    """A user's data is saved server-side and pulled back up on re-login."""
    c, u = _authed()
    cid = c.post("/api/conversations", json={"student_id": u}).json()["id"]

    assert c.post("/api/auth/logout").status_code == 200
    assert c.get(f"/api/student/{u}/conversations").status_code == 401  # session gone

    assert c.post("/api/auth/login", json={"login": u, "password": "pw123456"}).status_code == 200
    convs = c.get(f"/api/student/{u}/conversations").json()["conversations"]
    assert any(cv["id"] == cid for cv in convs)  # data restored

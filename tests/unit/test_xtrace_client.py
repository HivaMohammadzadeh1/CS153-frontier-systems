"""Tests for the XTrace REST client (mem.xtrace.ai)."""

import httpx
import pytest

from learning_memory_os.memory.xtrace import (
    XTraceClient,
    XTraceMemoryItem,
)


def _client_with_transport(handler) -> XTraceClient:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport, base_url="https://api.production.xtrace.ai")
    return XTraceClient(api_key="xtk_test", org_id="org_test", http=http)


def test_ingest_fact_sends_post_to_v1_memories_with_auth_headers():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = request.read().decode()
        return httpx.Response(
            202,
            json={
                "object": "ingest_job",
                "id": "job_abc",
                "status": "pending",
                "created_at": "2026-05-28T00:00:00Z",
                "updated_at": None,
                "result": None,
                "error": None,
            },
        )

    client = _client_with_transport(handler)
    client.ingest_fact(student_id="alice", text="I am studying KV caches")

    assert captured["method"] == "POST"
    assert captured["url"].endswith("/v1/memories")
    assert captured["headers"]["x-api-key"] == "xtk_test"
    assert captured["headers"]["x-org-id"] == "org_test"

    import json

    body = json.loads(captured["body"])
    assert body["user_id"] == "alice"
    assert "conv_id" in body
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"] == "I am studying KV caches"


def test_ingest_fact_uses_provided_conv_id_when_given():
    """The Streamlit app passes a stable per-chat conv_id so XTrace groups all
    of a chat's turns into one Episode automatically (no End-Session button)."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(202, json={"object": "ingest_job", "id": "j", "status": "pending", "created_at": "2026-05-28T00:00:00Z", "updated_at": None, "result": None, "error": None})

    client = _client_with_transport(handler)
    client.ingest_fact(student_id="alice", text="hi", conv_id="chat_42")

    assert captured["body"]["conv_id"] == "chat_42"


def test_ingest_episode_marks_role_as_assistant_summary():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.read().decode()
        return httpx.Response(202, json={"object": "ingest_job", "id": "job_x", "status": "pending", "created_at": "2026-05-28T00:00:00Z", "updated_at": None, "result": None, "error": None})

    client = _client_with_transport(handler)
    client.ingest_episode(student_id="alice", summary="Session covered attention and KV caching")

    import json

    body = json.loads(captured["body"])
    assert body["user_id"] == "alice"
    assert body["messages"][0]["content"] == "Session covered attention and KV caching"
    # episodes use a distinct conv_id prefix so we can filter for them later
    assert body["conv_id"].startswith("episode_")


def test_recall_returns_typed_memory_items():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url).endswith("/v1/memories/search")
        import json

        body = json.loads(request.read().decode())
        assert body["query"] == "what does the student know about caching?"
        assert body["filters"] == {"user_id": "alice"}
        assert body["limit"] == 5
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "id": "mem_1",
                        "type": "fact",
                        "text": "Student is implementing a KV cache",
                        "score": 0.91,
                        "user_id": "alice",
                        "conv_id": "conv_1",
                        "metadata": {},
                        "details": {},
                    },
                    {
                        "id": "mem_2",
                        "type": "episode",
                        "text": "Last session covered attention",
                        "score": 0.78,
                        "user_id": "alice",
                        "conv_id": "episode_1",
                        "metadata": {},
                        "details": {},
                    },
                ],
                "has_more": False,
                "next_cursor": None,
            },
        )

    client = _client_with_transport(handler)
    hits = client.recall(student_id="alice", query="what does the student know about caching?", k=5)

    assert len(hits) == 2
    assert all(isinstance(h, XTraceMemoryItem) for h in hits)
    assert hits[0].id == "mem_1"
    assert hits[0].kind == "fact"
    assert hits[0].similarity == pytest.approx(0.91)
    assert hits[1].kind == "episode"


def test_recall_returns_empty_list_on_5xx_and_does_not_raise():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "service unavailable"})

    client = _client_with_transport(handler)
    hits = client.recall(student_id="alice", query="anything", k=5)
    assert hits == []


def test_ingest_fact_swallows_5xx_and_does_not_raise():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    client = _client_with_transport(handler)
    # must not raise
    client.ingest_fact(student_id="alice", text="anything")


def test_list_memories_uses_search_endpoint_filtered_by_user_id():
    """We use POST /v1/memories/search internally because GET /v1/memories returned
    empty for our org even with no filters (server-side default filter we couldn't
    bypass). Search returns the same memories and is the documented recall path.
    """
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        import json
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "id": "mem_a",
                        "type": "fact",
                        "text": "Student is implementing a paged KV cache",
                        "score": 0.5,
                        "user_id": "alice",
                        "conv_id": "conv_1",
                        "metadata": {},
                        "details": {},
                    },
                    {
                        "id": "mem_b",
                        "type": "episode",
                        "text": "Session on attention and KV caching",
                        "score": 0.4,
                        "user_id": "alice",
                        "conv_id": "episode_1",
                        "metadata": {},
                        "details": {},
                    },
                ],
                "has_more": False,
                "next_cursor": None,
            },
        )

    client = _client_with_transport(handler)
    items = client.list_memories(student_id="alice", limit=100)

    assert captured["method"] == "POST"
    assert str(captured["url"]).endswith("/v1/memories/search")
    assert captured["body"]["filters"] == {"user_id": "alice"}
    assert captured["body"]["limit"] == 100

    assert len(items) == 2
    assert items[0].id == "mem_a"
    assert items[0].kind == "fact"
    assert items[1].kind == "episode"


def test_list_memories_returns_empty_on_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    client = _client_with_transport(handler)
    assert client.list_memories(student_id="alice", limit=50) == []


def test_circuit_breaker_short_circuits_after_failure():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, json={"error": "boom"})

    client = _client_with_transport(handler)
    # first call trips the breaker
    client.recall(student_id="alice", query="q", k=5)
    # subsequent calls within the cooldown window must not hit the wire
    client.recall(student_id="alice", query="q", k=5)
    client.recall(student_id="alice", query="q", k=5)
    assert calls["n"] == 1

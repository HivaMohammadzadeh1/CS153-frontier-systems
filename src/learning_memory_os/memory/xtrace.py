"""XTrace Memory Manager REST client.

A thin wrapper over the mem.xtrace.ai REST API (no Python SDK exists yet).
All recall errors return [] and log a warning; all ingest errors log and return.
A simple circuit breaker prevents repeated HTTP attempts when the service is
unhealthy.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Literal

import httpx
from pydantic import BaseModel, Field

from ..schemas.memory import MemoryItem

logger = logging.getLogger(__name__)

_BREAKER_COOLDOWN_SECONDS = 60.0


class XTraceMemoryItem(BaseModel):
    id: str
    kind: Literal["fact", "artifact", "episode"] = Field(alias="type")
    text: str
    similarity: float = Field(alias="score")
    user_id: str | None = None
    conv_id: str | None = None
    metadata: dict | None = None

    model_config = {"populate_by_name": True}


class XTraceClient:
    """REST client for XTrace Memory Manager."""

    def __init__(
        self,
        *,
        api_key: str,
        org_id: str,
        base_url: str = "https://api.production.xtrace.ai",
        http: httpx.Client | None = None,
        timeout: float = 10.0,
    ):
        self._http = http or httpx.Client(base_url=base_url, timeout=timeout)
        self._headers = {
            "x-api-key": api_key,
            "x-org-id": org_id,
            "content-type": "application/json",
        }
        self._breaker_open_until: float = 0.0

    def _breaker_open(self) -> bool:
        return time.monotonic() < self._breaker_open_until

    def _trip_breaker(self) -> None:
        self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECONDS

    def _ingest(self, *, user_id: str, conv_id: str, content: str) -> None:
        if self._breaker_open():
            return
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": content,
                    "date": datetime.now(tz=timezone.utc).isoformat(),
                    "dia_id": f"turn_{uuid.uuid4().hex[:12]}",
                }
            ],
            "user_id": user_id,
            "conv_id": conv_id,
        }
        try:
            resp = self._http.post("/v1/memories", headers=self._headers, json=body)
        except httpx.HTTPError as exc:
            logger.warning("xtrace ingest network error: %s", exc)
            self._trip_breaker()
            return
        if resp.status_code >= 500:
            logger.warning("xtrace ingest server error %s", resp.status_code)
            self._trip_breaker()
            return
        if resp.status_code >= 400:
            logger.warning("xtrace ingest client error %s: %s", resp.status_code, resp.text[:200])

    def ingest_fact(
        self, student_id: str, text: str, *, conv_id: str | None = None
    ) -> None:
        """Send one user turn to XTrace for fact extraction.

        If ``conv_id`` is provided (e.g. a stable per-chat session id), XTrace
        groups all turns under it into one Episode automatically. If omitted,
        falls back to a date-based id.
        """
        cid = conv_id or f"conv_{datetime.now(tz=timezone.utc).strftime('%Y%m%d')}"
        self._ingest(user_id=student_id, conv_id=cid, content=text)

    def ingest_episode(self, student_id: str, summary: str) -> None:
        """Send an end-of-session summary to XTrace as a distinct conversation.

        The ``episode_`` conv_id prefix lets us filter for session summaries
        separately from per-turn facts on recall if we ever want to.
        """
        conv_id = f"episode_{uuid.uuid4().hex[:12]}"
        self._ingest(user_id=student_id, conv_id=conv_id, content=summary)

    def list_memories(
        self, student_id: str, *, limit: int = 50
    ) -> list[XTraceMemoryItem]:
        """List all stored memories for this student.

        Uses the search endpoint internally — GET /v1/memories returned empty
        even with no filters for our org, but POST /v1/memories/search returns
        the same memories. We treat similarity-ranked search results as the
        "list" since for our small per-student index ranking-by-relevance to a
        generic query is acceptable.

        Returns [] on any error.
        """
        if self._breaker_open():
            return []
        body = {
            "query": "memory",  # generic query; XTrace requires minLength: 1
            "filters": {"user_id": student_id},
            "limit": max(1, min(int(limit), 100)),  # server-side cap is 100
        }
        try:
            resp = self._http.post(
                "/v1/memories/search", headers=self._headers, json=body
            )
        except httpx.HTTPError as exc:
            logger.warning("xtrace list network error: %s", exc)
            self._trip_breaker()
            return []
        if resp.status_code >= 500:
            logger.warning("xtrace list server error %s", resp.status_code)
            self._trip_breaker()
            return []
        if resp.status_code >= 400:
            logger.warning("xtrace list client error %s: %s", resp.status_code, resp.text[:200])
            return []
        payload = resp.json()
        out: list[XTraceMemoryItem] = []
        for item in payload.get("data", []):
            if item.get("score") is None:
                item = {**item, "score": 0.0}
            out.append(XTraceMemoryItem.model_validate(item))
        return out

    def recall(self, student_id: str, query: str, *, k: int = 5) -> list[XTraceMemoryItem]:
        """Search this student's memories. Returns [] on any error."""
        if self._breaker_open():
            return []
        body = {"query": query, "filters": {"user_id": student_id}, "limit": k}
        try:
            resp = self._http.post(
                "/v1/memories/search", headers=self._headers, json=body
            )
        except httpx.HTTPError as exc:
            logger.warning("xtrace recall network error: %s", exc)
            self._trip_breaker()
            return []
        if resp.status_code >= 500:
            logger.warning("xtrace recall server error %s", resp.status_code)
            self._trip_breaker()
            return []
        if resp.status_code >= 400:
            logger.warning("xtrace recall client error %s: %s", resp.status_code, resp.text[:200])
            return []
        payload = resp.json()
        return [XTraceMemoryItem.model_validate(item) for item in payload.get("data", [])]


def xtrace_to_memory_item(hit: XTraceMemoryItem) -> MemoryItem:
    """Convert an XTrace recall hit into a selector-compatible MemoryItem.

    The XTrace similarity score is preserved in metadata so the scorer can use
    it as the relevance signal in lieu of a cosine against an embedding.
    """
    return MemoryItem(
        id=f"xtrace:{hit.id}",
        tier="xtrace",
        title=hit.kind,
        body=hit.text,
        token_estimate=max(1, len(hit.text) // 4),
        metadata={"xtrace_similarity": float(hit.similarity), "xtrace_kind": hit.kind},
    )

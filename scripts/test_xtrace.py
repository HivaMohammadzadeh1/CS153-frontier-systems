"""Smoke test for the XTrace 5th memory tier.

Round-trips a fact ingest + a recall against the live mem.xtrace.ai service.
Requires XTRACE_API_KEY and XTRACE_ORG_ID in .env.

Usage:
    uv run python scripts/test_xtrace.py
"""

from __future__ import annotations

import sys
import time

from learning_memory_os.config import get_settings
from learning_memory_os.memory.xtrace import XTraceClient


def main() -> int:
    s = get_settings()
    if not s.xtrace_api_key or not s.xtrace_org_id:
        print("FAIL: XTRACE_API_KEY or XTRACE_ORG_ID missing from .env")
        return 1

    print(f"using base_url = {s.xtrace_base_url}")
    print(f"org_id         = {s.xtrace_org_id[:8]}...")

    client = XTraceClient(
        api_key=s.xtrace_api_key,
        org_id=s.xtrace_org_id,
        base_url=s.xtrace_base_url,
    )

    student = "smoke-test-user"
    seed = "I am implementing a paged KV cache and confused about page-table mapping to attention heads."

    print("\n[1/3] ingesting a fact...")
    client.ingest_fact(student_id=student, text=seed)
    print("    sent. (server processes async — waiting a few seconds before recall.)")

    time.sleep(5)

    print("\n[2/3] recalling memories for a related query...")
    hits = client.recall(
        student_id=student,
        query="What should I think about for cache invalidation?",
        k=5,
    )

    print(f"\n[3/3] recall returned {len(hits)} memory item(s):")
    if not hits:
        print("    (empty — either ingestion is still processing, or credentials are wrong.)")
        print("    re-run this script in 10s; if still empty, check the Streamlit logs for")
        print("    `xtrace recall server error` lines that indicate an auth or org-id failure.")
        return 2

    for i, h in enumerate(hits, 1):
        text = h.text[:120].replace("\n", " ")
        print(f"  {i}. [{h.kind}] sim={h.similarity:.3f}  {text}{'...' if len(h.text) > 120 else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

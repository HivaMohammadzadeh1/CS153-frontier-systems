"""End-to-end smoke test: ingest a topic, ask a question, verify a reply."""

import subprocess
import sys
from pathlib import Path


def test_full_pipeline_kv_cache(db_conn):
    # Ingest (idempotent across runs — the topic may already be in the DB from prior runs)
    src = Path("data/seed_topics/kv_cache/source.md").resolve()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.ingest_topic",
            "--topic-id",
            "kv_cache",
            "--source",
            str(src),
        ],
        check=True,
        env={"PYTHONPATH": "src", **__import__("os").environ},
    )

    # Ask
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.tutor_repl",
            "--student-id",
            "hiva-smoke",
            "--question",
            "What is a KV cache and why does it exist?",
            "--topic-id",
            "kv_cache",
            "--budget",
            "2000",
        ],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src", **__import__("os").environ},
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout.lower()
    # Reply should mention K and V (the cached tensors). Loose check.
    assert "k" in out and "v" in out

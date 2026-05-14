"""Smoke test: bulk ingester CLI surface."""

import subprocess
import sys


def test_ingest_all_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.ingest_all", "--help"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src", **__import__("os").environ},
    )
    assert result.returncode == 0
    assert "ingest" in result.stdout.lower()


def test_ingest_all_dry_run_reports_topics(db_conn):
    result = subprocess.run(
        [sys.executable, "-m", "scripts.ingest_all", "--dry-run"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src", **__import__("os").environ},
    )
    assert result.returncode == 0, result.stderr
    assert "kv_cache" in result.stdout
    assert "agent_memory" in result.stdout

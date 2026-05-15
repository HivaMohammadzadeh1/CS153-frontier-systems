"""Smoke test the multi-turn session CLI surface (help)."""

import subprocess, sys


def test_tutor_session_help():
    r = subprocess.run(
        [sys.executable, "-m", "scripts.tutor_session", "--help"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src", **__import__("os").environ},
    )
    assert r.returncode == 0
    assert "student-id" in r.stdout.lower()

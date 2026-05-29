import os
import subprocess
import sys


def test_eval_routers_help():
    r = subprocess.run(
        [sys.executable, "-m", "scripts.eval_routers", "--help"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src", **os.environ},
    )
    assert r.returncode == 0
    out = r.stdout.lower()
    assert "test" in out or "eval" in out or "limit" in out

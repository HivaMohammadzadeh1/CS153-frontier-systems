import subprocess, sys


def test_generate_trajectories_help():
    r = subprocess.run(
        [sys.executable, "-m", "scripts.generate_trajectories", "--help"],
        capture_output=True, text=True,
        env={"PYTHONPATH": "src", **__import__("os").environ},
    )
    assert r.returncode == 0
    assert "trajectories" in r.stdout.lower() or "generate" in r.stdout.lower() or "target" in r.stdout.lower()

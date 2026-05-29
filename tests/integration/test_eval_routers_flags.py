"""The eval CLI must run with neither the frontier baseline nor adapters,
so the two halves (GPU adapter job / login-pod frontier baseline) can be split."""

import json
import os
import subprocess
import sys

from learning_memory_os.trajectories.schemas import (
    StudentState,
    PoolItem,
    TaskType,
    Trajectory,
)


def _write_one_trajectory(path):
    t = Trajectory(
        id="t1",
        student_state=StudentState(
            student_id="s", mastery={}, active_misconceptions=[], recent_episodic_ids=[]
        ),
        task_type=TaskType.EXPLAIN,
        task_text="explain A",
        budget=300,
        candidate_pool=[PoolItem(id="aaaa1111", title="A", body_excerpt="x", token_estimate=100)],
        oracle_selection=["aaaa1111"],
    )
    path.write_text(t.model_dump_json() + "\n")


def test_eval_routers_no_frontier_no_adapters_writes_empty(tmp_path):
    test_file = tmp_path / "val.jsonl"
    out_file = tmp_path / "results.json"
    _write_one_trajectory(test_file)

    r = subprocess.run(
        [
            sys.executable, "-m", "scripts.eval_routers",
            "--no-frontier", "--no-adapters",
            "--test", str(test_file),
            "--out", str(out_file),
            "--limit", "1",
        ],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src", **os.environ},
    )
    assert r.returncode == 0, r.stderr
    assert json.loads(out_file.read_text()) == []

"""Build the Loop-2 router training set: synthetic trajectories + REAL,
reward-weighted captured traces -> data/trajectories/main.jsonl (the file
scripts/finetune_router.py defaults to).

This is the bridge that moves the router off *pure* oracle distillation toward
learning from what actually helped students (continuous-improvement Loop 2):

  1. (optional) label captured turns by realized mastery gain
  2. export them as Trajectories, reward-weighted (good turns upsampled, bad dropped)
  3. mix with the synthetic set and write JSONL

Then retrain on the cluster with TRAJ pointing at the output, e.g.:
  TRAJ=$REPO/data/trajectories/main.jsonl bash cluster/submit_sweep.sh
"""
from pathlib import Path
from typing import Optional

import typer

from learning_memory_os.config import get_settings
from learning_memory_os.memory.store import connect
from learning_memory_os.memory.trace import TraceStore
from learning_memory_os.trajectories.schemas import Trajectory

app = typer.Typer()


@app.command()
def main(
    synthetic: Path = typer.Option(Path("data/trajectories/val.jsonl"), "--synthetic"),
    student: Optional[str] = typer.Option(None, "--student", help="limit real traces to one student; default all"),
    min_reward: Optional[float] = typer.Option(None, "--min-reward", help="drop turns below this reward"),
    weight_by_reward: bool = typer.Option(True, "--weight-by-reward/--no-weight"),
    label_from_mastery: bool = typer.Option(True, "--label-from-mastery/--no-label"),
    out: Path = typer.Option(Path("data/trajectories/main.jsonl"), "--out"),
):
    conn = connect(get_settings().database_url)
    try:
        store = TraceStore(conn)
        if label_from_mastery and student:
            labeled = store.label_rewards_from_mastery(student)
            conn.commit()
            typer.echo(f"[build] labeled {labeled} turns from realized mastery gain")
        real = store.export_trajectories(student, min_reward=min_reward, weight_by_reward=weight_by_reward)
    finally:
        conn.close()

    synth: list[Trajectory] = []
    if synthetic.exists():
        for line in synthetic.read_text().splitlines():
            if line.strip():
                synth.append(Trajectory.model_validate_json(line))

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for t in [*synth, *real]:
            f.write(t.model_dump_json() + "\n")

    typer.echo(
        f"[build] wrote {len(synth)} synthetic + {len(real)} real (reward-weighted) "
        f"= {len(synth) + len(real)} trajectories -> {out}"
    )


if __name__ == "__main__":
    app()

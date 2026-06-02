"""Export captured per-user learning traces to fine-tune-ready Trajectory JSONL.

The output is the same schema scripts/finetune_router.py consumes, so real user
data can be mixed with (or replace) the synthetic trajectory set.

    uv run python -m scripts.export_traces --out data/trajectories/real.jsonl
    uv run python -m scripts.export_traces --student Hiva --min-reward 0.6
"""

import json
from pathlib import Path

import typer

from learning_memory_os.config import get_settings
from learning_memory_os.memory.store import connect
from learning_memory_os.memory.trace import TraceStore

app = typer.Typer()


@app.command()
def main(
    out: Path = typer.Option(Path("data/trajectories/real.jsonl"), "--out"),
    student_id: str = typer.Option(None, "--student", help="Limit to one student id"),
    min_reward: float = typer.Option(
        None, "--min-reward", help="Only export turns whose reward >= this"
    ),
    fmt: str = typer.Option(
        "router", "--format", help="router (selection training) or tutor (reply+reward)"
    ),
):
    settings = get_settings()
    conn = connect(settings.database_url)
    try:
        store = TraceStore(conn)
        if fmt == "tutor":
            rows = store.export_records(student_id, min_reward=min_reward)
        else:
            rows = [t.model_dump() for t in store.export_trajectories(student_id, min_reward=min_reward)]
    finally:
        conn.close()

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    typer.echo(f"Exported {len(rows)} {fmt} records to {out}")


if __name__ == "__main__":
    app()

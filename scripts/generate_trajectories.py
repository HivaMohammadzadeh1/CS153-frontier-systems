"""Generate N synthetic tutoring trajectories and write to JSONL.

Sampling (DB) runs sequentially; oracle LLM calls run in a ThreadPoolExecutor.
"""

import json
import random
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import typer

from learning_memory_os.config import get_settings
from learning_memory_os.llm import LLM
from learning_memory_os.memory.store import connect
from learning_memory_os.trajectories.sampler import sample_candidate_pool, sample_student_state
from learning_memory_os.trajectories.generator import build_trajectory
from learning_memory_os.trajectories.schemas import TaskType
from learning_memory_os.ingestion.topic_loader import load_topics


app = typer.Typer()


TASK_TEMPLATES = {
    TaskType.EXPLAIN: [
        "Explain the core idea of {title}.",
        "Why does {title} exist? What problem does it solve?",
        "Walk me through how {title} works step by step.",
        "What is the most common misconception about {title}?",
    ],
    TaskType.QUIZ: [
        "Generate a 3-question quiz on {title}.",
        "Quiz me on the tradeoffs in {title}.",
    ],
    TaskType.REVIEW: [
        "Summarize what I should remember about {title} for review.",
        "Give me a one-page review sheet on {title}.",
    ],
    TaskType.LAB: [
        "Suggest a hands-on lab to deepen mastery of {title}.",
    ],
}


def _sample_batch(conn, populated_topics, batch_size, base_idx):
    """Pre-sample inputs for `batch_size` trajectories. DB-bound, fast (~10ms/trajectory)."""
    batch = []
    for i in range(batch_size):
        topic, _n = random.choice(populated_topics)
        task_type = random.choice(list(TaskType))
        template = random.choice(TASK_TEMPLATES[task_type])
        task_text = template.format(title=topic.title)
        pool = sample_candidate_pool(conn, target_topic=topic.id, pool_size=15)
        if not pool:
            continue
        state = sample_student_state(
            conn,
            student_id=f"synthetic-{uuid.uuid4().hex[:8]}",
            target_concepts=[topic.id],
        )
        batch.append({
            "idx": base_idx + i,
            "topic": topic,
            "task_type": task_type,
            "task_text": task_text,
            "pool": pool,
            "state": state,
            "budget": random.choice([2000, 3000, 4000, 6000]),
        })
    return batch


def _build_one(item, llm):
    traj = build_trajectory(
        traj_id=f"traj-{item['idx']:06d}",
        student_state=item["state"],
        task_type=item["task_type"],
        task_text=item["task_text"],
        budget=item["budget"],
        candidate_pool=item["pool"],
        oracle_llm=llm,
    )
    return traj


@app.command()
def main(
    target: int = typer.Option(5000, "--target"),
    out: Path = typer.Option(Path("data/trajectories/main.jsonl"), "--out"),
    config: Path = typer.Option(Path("config/topics.yaml"), "--config"),
    oracle_model: str = typer.Option("claude-sonnet-4-6", "--oracle-model"),
    seed: int = typer.Option(42, "--seed"),
    workers: int = typer.Option(16, "--workers", help="Parallel oracle calls"),
    batch_size: int = typer.Option(64, "--batch-size", help="Pre-sample batch size"),
):
    random.seed(seed)
    settings = get_settings()
    topics = load_topics(config)

    out.parent.mkdir(parents=True, exist_ok=True)

    llm = LLM(api_key=settings.anthropic_api_key, model=oracle_model)
    conn = connect(settings.database_url)

    populated_topics = []
    with conn.cursor() as cur:
        for t in topics:
            cur.execute("SELECT count(*) AS n FROM semantic_items WHERE topic_id = %s", (t.id,))
            n = cur.fetchone()["n"]
            if n > 0:
                populated_topics.append((t, n))
    if not populated_topics:
        typer.echo("No populated topics in DB. Run Plan 2 first.", err=True)
        raise typer.Exit(2)

    typer.echo(f"Generating {target} trajectories across {len(populated_topics)} populated topics with {workers} workers...")

    written = 0
    failed = 0
    try:
        with out.open("w") as f:
            base_idx = 0
            while written < target:
                remaining = target - written
                this_batch = min(batch_size, remaining)
                batch = _sample_batch(conn, populated_topics, this_batch, base_idx)
                if not batch:
                    break
                base_idx += this_batch

                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futures = {ex.submit(_build_one, item, llm): item for item in batch}
                    for fut in as_completed(futures):
                        item = futures[fut]
                        try:
                            traj = fut.result()
                            f.write(json.dumps(traj.model_dump(), default=str) + "\n")
                            f.flush()
                            written += 1
                            if written % 50 == 0:
                                typer.echo(f"  wrote {written}/{target}")
                        except Exception as e:
                            failed += 1
                            if failed <= 5:
                                typer.echo(f"  [warn] traj failed: {e}", err=True)
    finally:
        conn.close()

    typer.echo(f"\nDone. {written} trajectories written to {out}. {failed} failures.")


if __name__ == "__main__":
    app()

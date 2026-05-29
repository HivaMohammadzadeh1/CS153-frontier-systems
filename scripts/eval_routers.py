"""Evaluate routers on a held-out trajectory split.

Writes <out> with one row per router: {router_id, precision, recall, jaccard, n, ms_per_call}.
Includes a frontier-API baseline (network + ANTHROPIC_API_KEY, no GPU) and every
fine-tuned adapter found under data/router_checkpoints/<size>/adapter (needs GPU).

Pass --no-frontier to skip the API baseline (e.g. on a GPU job with no network).
"""

import json
import time
from pathlib import Path

import typer
import yaml

from learning_memory_os.config import get_settings
from learning_memory_os.llm import LLM
from learning_memory_os.trajectories.schemas import Trajectory
from learning_memory_os.eval.router_eval import evaluate

app = typer.Typer()


def _load_test_trajectories(path: Path, limit: int) -> list[Trajectory]:
    items: list[Trajectory] = []
    with path.open() as f:
        for line in f:
            items.append(Trajectory.model_validate_json(line))
            if len(items) >= limit:
                break
    return items


@app.command()
def main(
    test_file: Path = typer.Option(Path("data/trajectories/val.jsonl"), "--test"),
    test_limit: int = typer.Option(500, "--limit"),
    out: Path = typer.Option(Path("data/eval/router_results.json"), "--out"),
    checkpoints: Path = typer.Option(Path("data/router_checkpoints"), "--checkpoints"),
    sizes_config: Path = typer.Option(Path("config/router_sizes.yaml"), "--sizes-config"),
    frontier: bool = typer.Option(True, "--frontier/--no-frontier"),
    adapters: bool = typer.Option(True, "--adapters/--no-adapters"),
    frontier_model: str = typer.Option("claude-sonnet-4-6", "--frontier-model"),
):
    out.parent.mkdir(parents=True, exist_ok=True)
    test = _load_test_trajectories(test_file, test_limit)
    typer.echo(f"Loaded {len(test)} test trajectories from {test_file}.")
    gold = [t.oracle_selection for t in test]

    results: list[dict] = []

    if frontier:
        from learning_memory_os.router.frontier_api import FrontierAPIRouter

        settings = get_settings()
        fr = FrontierAPIRouter(LLM(api_key=settings.anthropic_api_key, model=frontier_model))
        preds: list[list[str]] = []
        t0 = time.time()
        for t in test:
            preds.append(
                fr.route(
                    student_state=t.student_state,
                    task_type=t.task_type,
                    task_text=t.task_text,
                    budget=t.budget,
                    candidate_pool=t.candidate_pool,
                )
            )
        elapsed = time.time() - t0
        m = evaluate(preds, gold)
        results.append(
            {
                "router_id": f"frontier_api:{frontier_model}",
                "precision": m.precision,
                "recall": m.recall,
                "jaccard": m.jaccard,
                "n": m.n,
                "ms_per_call": elapsed / max(1, m.n) * 1000,
            }
        )
        typer.echo(f"  frontier_api:{frontier_model}  jaccard={m.jaccard:.3f}")

    sizes = yaml.safe_load(sizes_config.read_text())["sizes"] if adapters else []
    for s in sizes:
        adapter = checkpoints / s["id"] / "adapter"
        if not adapter.exists():
            typer.echo(f"  [skip] {s['id']}: no adapter at {adapter}")
            continue
        from learning_memory_os.router.infer import FineTunedRouter

        try:
            r = FineTunedRouter(adapter_dir=adapter, base_model=s["hf_model"])
        except Exception as e:  # noqa: BLE001 — report and continue the sweep
            typer.echo(f"  [skip] {s['id']}: load failed: {e}")
            continue
        preds = []
        t0 = time.time()
        for t in test:
            preds.append(
                r.route(
                    student_state=t.student_state,
                    task_type=t.task_type,
                    task_text=t.task_text,
                    budget=t.budget,
                    candidate_pool=t.candidate_pool,
                )
            )
        elapsed = time.time() - t0
        m = evaluate(preds, gold)
        results.append(
            {
                "router_id": s["id"],
                "precision": m.precision,
                "recall": m.recall,
                "jaccard": m.jaccard,
                "n": m.n,
                "ms_per_call": elapsed / max(1, m.n) * 1000,
            }
        )
        typer.echo(f"  {s['id']}  jaccard={m.jaccard:.3f}  ms/call={results[-1]['ms_per_call']:.1f}")

    out.write_text(json.dumps(results, indent=2))
    typer.echo(f"Wrote {len(results)} rows to {out}")


if __name__ == "__main__":
    app()

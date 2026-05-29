"""Render the router accuracy-vs-cost Pareto plot from eval results."""

from pathlib import Path

import typer

from learning_memory_os.eval.pareto import plot_pareto

app = typer.Typer()


@app.command()
def main(
    results: Path = typer.Option(Path("data/eval/router_results.json"), "--results"),
    out: Path = typer.Option(Path("data/eval/pareto.png"), "--out"),
):
    plot_pareto(results, out)
    typer.echo(f"Wrote {out}")


if __name__ == "__main__":
    app()

"""Plot the router accuracy-vs-cost frontier from an eval results JSON."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed (tests, login pod, cluster)
import matplotlib.pyplot as plt  # noqa: E402


def plot_pareto(results_json: Path, out_png: Path) -> None:
    data = json.loads(Path(results_json).read_text())
    xs = [d["ms_per_call"] for d in data]
    ys = [d["jaccard"] for d in data]
    labels = [d["router_id"] for d in data]

    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, s=80)
    for x, y, label in zip(xs, ys, labels):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    plt.xscale("log")
    plt.xlabel("Latency (ms / call, log scale)")
    plt.ylabel("Selection Jaccard vs oracle")
    plt.title("Learning Memory OS — Router Accuracy vs Cost")
    plt.grid(True, alpha=0.3)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

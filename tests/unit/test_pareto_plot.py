import json

from learning_memory_os.eval.pareto import plot_pareto


def test_plot_pareto_writes_png(tmp_path):
    results = [
        {"router_id": "qwen2_5_0_5b", "jaccard": 0.55, "ms_per_call": 40.0},
        {"router_id": "frontier_api_sonnet", "jaccard": 0.82, "ms_per_call": 900.0},
    ]
    results_json = tmp_path / "router_results.json"
    results_json.write_text(json.dumps(results))
    out_png = tmp_path / "pareto.png"

    plot_pareto(results_json, out_png)

    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_plot_pareto_creates_parent_dir(tmp_path):
    results_json = tmp_path / "r.json"
    results_json.write_text(json.dumps([{"router_id": "x", "jaccard": 0.1, "ms_per_call": 10.0}]))
    out_png = tmp_path / "nested" / "deep" / "pareto.png"

    plot_pareto(results_json, out_png)

    assert out_png.exists()

from pathlib import Path
import yaml


def test_router_sizes_lists_4_sizes():
    cfg = yaml.safe_load(Path("config/router_sizes.yaml").read_text())
    assert len(cfg["sizes"]) == 4
    ids = [s["id"] for s in cfg["sizes"]]
    assert ids == ["qwen2_5_0_5b", "qwen2_5_1_5b", "qwen2_5_3b", "qwen2_5_7b"]

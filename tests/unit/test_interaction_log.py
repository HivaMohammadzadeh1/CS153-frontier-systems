import json
from pathlib import Path
from learning_memory_os.logging_utils.interactions import InteractionLogger


def test_logger_appends_jsonl(tmp_path: Path):
    log_path = tmp_path / "interactions.jsonl"
    logger = InteractionLogger(path=log_path)
    logger.log(
        {
            "event": "routing_decision",
            "task": "explain kv cache",
            "selected_ids": ["a", "b"],
            "dropped_ids": ["c"],
            "tokens_used": 700,
            "budget": 1000,
        }
    )
    logger.log({"event": "tutor_reply", "text": "..."})

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["event"] == "routing_decision"
    assert "timestamp" in first

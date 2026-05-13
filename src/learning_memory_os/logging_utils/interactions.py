import json
from datetime import datetime, timezone
from pathlib import Path


class InteractionLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")

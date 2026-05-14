import subprocess
import sys
from pathlib import Path


def test_ingest_runs_end_to_end(db_conn, tmp_path: Path):
    # Real LLM call. Requires .env with valid keys.
    src = Path("data/seed_topics/kv_cache/source.md").resolve()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.ingest_topic",
            "--topic-id",
            "kv_cache",
            "--source",
            str(src),
        ],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src", **__import__("os").environ},
    )
    assert result.returncode == 0, result.stderr

    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT artifact_type, count(*) FROM semantic_items "
            "WHERE topic_id = 'kv_cache' GROUP BY artifact_type"
        )
        counts = {r["artifact_type"]: r["count"] for r in cur.fetchall()}
    assert counts.get("concept", 0) >= 1
    assert counts.get("misconception", 0) >= 1

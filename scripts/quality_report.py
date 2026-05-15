"""Per-topic curriculum quality report v2.

Adds:
- Near-duplicate concept detection (cosine > 0.95)
- Short-body flag (< 100 chars)
- Low-misconception ratio flag (< 5%)
- Thin coverage flag (< 4 artifacts)

Use --samples N to also show N random body excerpts per topic (default 0).
"""

import math
from pathlib import Path
import typer

from learning_memory_os.config import get_settings
from learning_memory_os.ingestion.topic_loader import load_topics
from learning_memory_os.memory.store import connect
from learning_memory_os.memory.semantic import _parse_vec   # type: ignore


app = typer.Typer()


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _find_near_dups(rows: list[dict], threshold: float = 0.95) -> list[tuple[str, str, float]]:
    """Pairwise cosine over concept embeddings. Returns list of (title_a, title_b, score)."""
    parsed = [(r["title"], _parse_vec(r["embedding"])) for r in rows]
    out: list[tuple[str, str, float]] = []
    for i in range(len(parsed)):
        for j in range(i + 1, len(parsed)):
            ta, va = parsed[i]
            tb, vb = parsed[j]
            if not va or not vb:
                continue
            c = _cosine(va, vb)
            if c >= threshold:
                out.append((ta, tb, c))
    return out


@app.command()
def main(
    samples_per_topic: int = typer.Option(0, "--samples"),
    config: str = typer.Option("config/topics.yaml", "--config"),
    dup_threshold: float = typer.Option(0.95, "--dup-threshold"),
):
    settings = get_settings()
    topics = load_topics(Path(config))
    conn = connect(settings.database_url)
    flags: dict[str, list[str]] = {}

    try:
        for topic in topics:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT artifact_type, count(*) AS n FROM semantic_items "
                    "WHERE topic_id = %s GROUP BY artifact_type ORDER BY artifact_type",
                    (topic.id,),
                )
                breakdown = list(cur.fetchall())
            total = sum(r["n"] for r in breakdown)
            typer.echo(f"\n=== {topic.area}/{topic.id} ({topic.title}) — total {total}")
            if not breakdown:
                typer.echo("  (no artifacts; missing source content)")
                flags.setdefault(topic.id, []).append("empty")
                continue

            for row in breakdown:
                typer.echo(f"  {row['artifact_type']}: {row['n']}")

            # Flag: thin coverage
            if total < 4:
                flags.setdefault(topic.id, []).append(f"thin (only {total} artifacts)")

            # Flag: low misconception ratio
            misc = next((r["n"] for r in breakdown if r["artifact_type"] == "misconception"), 0)
            ratio = misc / total if total else 0
            if total >= 8 and ratio < 0.05:
                flags.setdefault(topic.id, []).append(
                    f"low misconceptions ({misc}/{total} = {ratio:.0%})"
                )

            # Flag: short bodies
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) AS n FROM semantic_items "
                    "WHERE topic_id = %s AND length(body) < 100",
                    (topic.id,),
                )
                short = cur.fetchone()["n"]
            if short > 0:
                flags.setdefault(topic.id, []).append(f"{short} short bodies (<100 chars)")

            # Flag: near-duplicate concepts
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT title, embedding::text AS embedding FROM semantic_items "
                    "WHERE topic_id = %s AND artifact_type = 'concept' "
                    "AND embedding IS NOT NULL",
                    (topic.id,),
                )
                concept_rows = list(cur.fetchall())
            dups = _find_near_dups(concept_rows, threshold=dup_threshold)
            if dups:
                flags.setdefault(topic.id, []).append(f"{len(dups)} near-dup concept pairs")
                for ta, tb, score in dups[:3]:
                    typer.echo(f"    NEAR-DUP ({score:.3f}): {ta!r} ≈ {tb!r}")

            # Samples
            if samples_per_topic > 0:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT artifact_type, title, body FROM semantic_items "
                        "WHERE topic_id = %s ORDER BY random() LIMIT %s",
                        (topic.id, samples_per_topic),
                    )
                    samples = list(cur.fetchall())
                for s in samples:
                    typer.echo(f"  ~ sample [{s['artifact_type']}] {s['title']}")
                    body_preview = (s["body"] or "")[:200].replace("\n", " ")
                    typer.echo(f"    {body_preview}{'...' if len(s['body'] or '') > 200 else ''}")
    finally:
        conn.close()

    typer.echo("\n=== FLAGS ===")
    if not flags:
        typer.echo("  (none)")
    else:
        for tid, items in flags.items():
            for f in items:
                typer.echo(f"  {tid}: {f}")


if __name__ == "__main__":
    app()

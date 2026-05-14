"""Print a per-topic ingestion quality report:
- artifact count per type
- a random sample of artifact bodies for spot-check
- topics with zero artifacts (missing source content)
"""

from pathlib import Path
import typer

from learning_memory_os.config import get_settings
from learning_memory_os.ingestion.topic_loader import load_topics
from learning_memory_os.memory.store import connect


app = typer.Typer()


@app.command()
def main(
    samples_per_topic: int = typer.Option(1, "--samples"),
    config: str = typer.Option("config/topics.yaml", "--config"),
):
    settings = get_settings()
    topics = load_topics(Path(config))
    conn = connect(settings.database_url)
    try:
        for topic in topics:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT artifact_type, count(*) AS n "
                    "FROM semantic_items WHERE topic_id = %s "
                    "GROUP BY artifact_type ORDER BY artifact_type",
                    (topic.id,),
                )
                breakdown = list(cur.fetchall())
            total = sum(r["n"] for r in breakdown)
            typer.echo(f"\n=== {topic.area}/{topic.id} ({topic.title}) — total {total}")
            if not breakdown:
                typer.echo("  (no artifacts; topic missing or extraction failed)")
                continue
            for row in breakdown:
                typer.echo(f"  {row['artifact_type']}: {row['n']}")
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


if __name__ == "__main__":
    app()

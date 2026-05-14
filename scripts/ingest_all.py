"""Bulk-ingest the curriculum defined in config/topics.yaml into semantic memory.

Idempotency: by default, topics with >0 existing artifacts are skipped.
Use --force to delete-and-reingest a topic.
Use --only TOPIC_ID to ingest a single topic.
Use --dry-run to print what would happen without calling any API or DB.
"""

from pathlib import Path
import typer

from learning_memory_os.config import get_settings
from learning_memory_os.llm import LLM
from learning_memory_os.embeddings import Embedder
from learning_memory_os.memory.store import connect
from learning_memory_os.memory.semantic import SemanticStore
from learning_memory_os.ingestion.extractors import ArtifactExtractor
from learning_memory_os.ingestion.topic_loader import load_topics, resolve_sources, Topic
from learning_memory_os.schemas.memory import MemoryItem
from learning_memory_os.schemas.artifacts import artifact_to_body


app = typer.Typer()


def _ingest_one(
    topic: Topic,
    *,
    extractor: ArtifactExtractor,
    embedder: Embedder,
    store: SemanticStore,
    base: Path,
) -> int:
    resolved = resolve_sources(topic, base=base)
    present = [(p, body) for (p, body, exists) in resolved if exists and body.strip()]
    if not present:
        typer.echo(f"  [skip] {topic.id}: no source files found")
        return 0

    combined = "\n\n".join(
        f"# Source: {p.name}\n\n{body}" for p, body in present
    )
    artifacts = extractor.extract(topic_id=topic.id, source_text=combined)
    if not artifacts:
        typer.echo(f"  [warn] {topic.id}: extractor returned 0 artifacts")
        return 0

    bodies = [artifact_to_body(a) for a in artifacts]
    vectors = embedder.embed_many(bodies)
    inserted = 0
    for a, v in zip(artifacts, vectors):
        item = MemoryItem.from_artifact(a, embedding=v)
        store.insert(item)
        inserted += 1
    return inserted


@app.command()
def main(
    config: Path = typer.Option(Path("config/topics.yaml"), "--config"),
    only: str | None = typer.Option(None, "--only", help="Process only this topic_id"),
    force: bool = typer.Option(False, "--force", help="Delete existing artifacts and re-ingest"),
    dry_run: bool = typer.Option(False, "--dry-run", help="List actions without executing"),
):
    """Ingest the full curriculum into semantic memory."""
    topics = load_topics(config)
    if only:
        topics = [t for t in topics if t.id == only]
        if not topics:
            typer.echo(f"No topic with id={only}", err=True)
            raise typer.Exit(2)

    if dry_run:
        typer.echo("DRY RUN — topics that would be processed:")
        for t in topics:
            typer.echo(f"  {t.area}/{t.id}: {len(t.sources)} source(s)")
            for s in t.sources:
                exists = Path(s).exists()
                marker = "OK" if exists else "MISSING"
                typer.echo(f"    [{marker}] {s}")
        return

    settings = get_settings()
    llm = LLM(api_key=settings.anthropic_api_key)
    embedder = Embedder(api_key=settings.openai_api_key)
    extractor = ArtifactExtractor(llm=llm)
    conn = connect(settings.database_url)
    store = SemanticStore(conn)

    total_inserted = 0
    try:
        for topic in topics:
            existing = store.count_by_topic(topic.id)
            if existing > 0 and not force:
                typer.echo(f"[exists] {topic.id}: {existing} artifacts — use --force to replace")
                continue
            if existing > 0 and force:
                removed = store.delete_by_topic(topic.id)
                typer.echo(f"[clear]  {topic.id}: removed {removed} existing artifacts")

            typer.echo(f"[ingest] {topic.id} (area {topic.area})")
            try:
                n = _ingest_one(
                    topic,
                    extractor=extractor,
                    embedder=embedder,
                    store=store,
                    base=Path("."),
                )
                typer.echo(f"  -> {n} artifacts")
                total_inserted += n
                conn.commit()
            except Exception as e:
                typer.echo(f"  [error] {topic.id}: {e}", err=True)
                conn.rollback()
                continue
    finally:
        conn.close()

    typer.echo(f"\nTotal inserted: {total_inserted}")


if __name__ == "__main__":
    app()

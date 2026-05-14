"""Ingest a single topic source file into semantic memory."""

from pathlib import Path
import typer

from learning_memory_os.config import get_settings
from learning_memory_os.llm import LLM
from learning_memory_os.embeddings import Embedder
from learning_memory_os.memory.store import connect
from learning_memory_os.memory.semantic import SemanticStore
from learning_memory_os.ingestion.extractors import ArtifactExtractor
from learning_memory_os.schemas.memory import MemoryItem
from learning_memory_os.schemas.artifacts import artifact_to_body


app = typer.Typer()


@app.command()
def main(
    topic_id: str = typer.Option(..., "--topic-id"),
    source: Path = typer.Option(..., "--source", exists=True, readable=True),
):
    settings = get_settings()
    llm = LLM(api_key=settings.anthropic_api_key)
    embedder = Embedder(api_key=settings.openai_api_key)
    extractor = ArtifactExtractor(llm=llm)

    text = source.read_text()
    artifacts = extractor.extract(topic_id=topic_id, source_text=text)
    if not artifacts:
        typer.echo("No artifacts extracted.", err=True)
        raise typer.Exit(2)

    bodies = [artifact_to_body(a) for a in artifacts]
    vectors = embedder.embed_many(bodies)

    conn = connect(settings.database_url)
    store = SemanticStore(conn)
    inserted = 0
    try:
        for a, v in zip(artifacts, vectors):
            item = MemoryItem.from_artifact(a, embedding=v)
            store.insert(item)
            inserted += 1
        conn.commit()
    finally:
        conn.close()

    typer.echo(f"Ingested {inserted} artifacts for topic '{topic_id}'.")


if __name__ == "__main__":
    app()

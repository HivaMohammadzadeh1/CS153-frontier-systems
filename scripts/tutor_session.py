"""Interactive multi-turn tutor session.

Usage:
  uv run python -m scripts.tutor_session --student-id NAME [--topic-id TOPIC] [--budget N]

Special commands at the prompt:
  :quit / :q          exit
  :topic TOPIC_ID     switch focus topic
  :topic              clear focus topic (use global vector search)
  :mastery            show current mastery for the student
  :misconceptions     list active misconceptions
  :help               show this list
"""

from collections import Counter
from pathlib import Path
import typer

from learning_memory_os.config import get_settings
from learning_memory_os.llm import LLM
from learning_memory_os.embeddings import Embedder
from learning_memory_os.memory.store import connect
from learning_memory_os.memory.semantic import SemanticStore
from learning_memory_os.memory.student import StudentStore
from learning_memory_os.memory.episodic import EpisodicStore
from learning_memory_os.selector.engine import RoutingEngine
from learning_memory_os.agents.tutor import TutorAgent
from learning_memory_os.logging_utils.interactions import InteractionLogger
from learning_memory_os.ingestion.topic_loader import (
    load_topics, resolve_prerequisite_titles,
)


app = typer.Typer()


def _candidates(semantic, embedder, question, topic_id):
    if topic_id:
        cands = semantic.by_topic(topic_id)
    else:
        q_emb = embedder.embed_one(question)
        cands = semantic.vector_search(query=q_emb, k=20)
    return cands


def _print_help():
    typer.echo(":quit / :q          exit")
    typer.echo(":topic TOPIC_ID     switch focus topic")
    typer.echo(":topic              clear focus topic")
    typer.echo(":mastery            show current mastery")
    typer.echo(":misconceptions     list active misconceptions")
    typer.echo(":help               show this list")


@app.command()
def main(
    student_id: str = typer.Option(..., "--student-id"),
    topic_id: str | None = typer.Option(None, "--topic-id"),
    budget: int = typer.Option(8000, "--budget"),
):
    settings = get_settings()
    llm = LLM(api_key=settings.anthropic_api_key)
    embedder = Embedder(api_key=settings.openai_api_key)
    engine = RoutingEngine()
    log_path = settings.log_dir / "interactions.jsonl"
    logger = InteractionLogger(path=log_path)

    conn = connect(settings.database_url)
    student_store = StudentStore(conn)
    semantic = SemanticStore(conn)
    episodic = EpisodicStore(conn)
    student_store.ensure_student(student_id)
    topics_cfg = load_topics(Path("config/topics.yaml"))

    tutor = TutorAgent(llm=llm, engine=engine, embedder=embedder, logger=logger)

    reuse_counts: Counter[str] = Counter()
    typer.echo(f"Tutor session for {student_id}. Focus topic: {topic_id or '(none)'}. :help for commands.")
    try:
        while True:
            try:
                question = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                typer.echo("\nexit")
                break
            if not question:
                continue
            if question in (":quit", ":q"):
                break
            if question == ":help":
                _print_help()
                continue
            if question == ":mastery":
                for m in student_store.mastery_for(student_id):
                    typer.echo(f"  {m.concept_id} score={m.score:.2f} conf={m.confidence:.2f}")
                continue
            if question == ":misconceptions":
                for m in student_store.active_misconceptions(student_id):
                    typer.echo(f"  [{m['id'][:8]}] {m['description']}")
                continue
            if question.startswith(":topic"):
                parts = question.split(maxsplit=1)
                topic_id = parts[1].strip() if len(parts) == 2 else None
                typer.echo(f"focus topic -> {topic_id or '(none)'}")
                continue

            candidates = _candidates(semantic, embedder, question, topic_id)
            misconceptions = {
                m["id"] for m in student_store.active_misconceptions(student_id)
            }
            prereq_titles = (
                resolve_prerequisite_titles(conn, topic_id=topic_id, topics=topics_cfg)
                if topic_id else set()
            )
            recent = episodic.recent(student_id, limit=10)
            recent_ids = {e.id for e in recent if e.id}

            response = tutor.answer(
                student_id=student_id,
                question=question,
                candidates=candidates,
                active_misconceptions=misconceptions,
                prerequisites=prereq_titles,
                recent_ids=recent_ids,
                reuse_counts=dict(reuse_counts),
                budget=budget,
            )

            for it in response.selected_items:
                reuse_counts[it.id] += 1

            episodic.append(
                student_id=student_id,
                event_type="question",
                payload={"text": question, "topic_id": topic_id},
            )
            episodic.append(
                student_id=student_id,
                event_type="tutor_reply",
                payload={
                    "text": response.text,
                    "selected_ids": [it.id for it in response.selected_items],
                    "tokens_used": response.tokens_used,
                },
            )
            conn.commit()

            typer.echo("\n" + "─" * 70)
            typer.echo(response.text)
            typer.echo("─" * 70)
            typer.echo(
                f"context: {len(response.selected_items)} items, "
                f"{response.tokens_used}/{budget} tokens"
            )
    finally:
        conn.close()


if __name__ == "__main__":
    app()

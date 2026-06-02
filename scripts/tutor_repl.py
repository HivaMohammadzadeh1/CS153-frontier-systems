"""Ask the tutor a question. Single-turn for now; loop comes later."""

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


app = typer.Typer()


@app.command()
def main(
    student_id: str = typer.Option(..., "--student-id"),
    question: str = typer.Option(..., "--question"),
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
    try:
        student = StudentStore(conn)
        student.ensure_student(student_id)
        semantic = SemanticStore(conn)
        episodic = EpisodicStore(conn)

        # Candidate pool: topic-scoped if topic given, else vector-search globally.
        if topic_id:
            candidates = semantic.by_topic(topic_id)
        else:
            q_emb = embedder.embed_one(question)
            candidates = semantic.vector_search(query=q_emb, k=20)

        active_misc = student.active_misconceptions(student_id)
        misc_concept_ids = {m["concept_id"] for m in active_misc if m.get("concept_id")}
        misc_topics: set[str] = set()
        if misc_concept_ids:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT topic_id FROM semantic_items WHERE id::text = ANY(%s)",
                    (list(misc_concept_ids),),
                )
                misc_topics = {r["topic_id"] for r in cur.fetchall()}
        due_concept_ids = set(student.due_for_review(student_id))

        prereq_titles: set[str] = set()
        if topic_id:
            from pathlib import Path
            from learning_memory_os.ingestion.topic_loader import (
                load_topics,
                resolve_prerequisite_titles,
            )
            topics_cfg = load_topics(Path("config/topics.yaml"))
            prereq_titles = resolve_prerequisite_titles(
                conn, topic_id=topic_id, topics=topics_cfg
            )

        tutor = TutorAgent(
            llm=llm, engine=engine, embedder=embedder, logger=logger
        )
        response = tutor.answer(
            student_id=student_id,
            question=question,
            candidates=candidates,
            misconception_concept_ids=misc_concept_ids,
            misconception_topics=misc_topics,
            prerequisites=prereq_titles,
            recent_ids=set(),
            reuse_counts={},
            due_concept_ids=due_concept_ids,
            budget=budget,
        )

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
    finally:
        conn.close()

    typer.echo(response.text)


if __name__ == "__main__":
    app()

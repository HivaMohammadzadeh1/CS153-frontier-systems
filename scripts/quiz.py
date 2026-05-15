"""Pre/post quiz CLI.

Usage:
  uv run python -m scripts.quiz pre --student-id NAME --topic-id TOPIC [-n 5]
  uv run python -m scripts.quiz post --student-id NAME --topic-id TOPIC [-n 5]
  uv run python -m scripts.quiz gain --student-id NAME --topic-id TOPIC

`pre` generates a baseline quiz BEFORE the student studies the topic.
`post` generates a parallel quiz AFTER (uses different question pool to avoid memorization).
`gain` reads the JSONL log and prints pre vs post averages.
"""

import json
import random
from pathlib import Path
from collections import defaultdict
import typer

from learning_memory_os.config import get_settings
from learning_memory_os.llm import LLM
from learning_memory_os.memory.store import connect
from learning_memory_os.memory.student import StudentStore
from learning_memory_os.eval.quiz import (
    QuizQuestion, score_answer, append_quiz_log, average_score, QuizScore,
)


app = typer.Typer()


GENERATE_SYSTEM = """Generate a short open-ended quiz on the given ML systems engineering topic.
The quiz should test deep understanding, not lookup. Output STRICT JSON:
{"questions": [{"question": "...", "rubric": "<what a correct answer must contain>"}, ...]}
"""


def _generate_quiz(llm: LLM, topic_id: str, topic_excerpt: str, n: int, phase: str, seed: int) -> list[QuizQuestion]:
    user = (
        f"TOPIC: {topic_id}\n\n"
        f"TOPIC MATERIAL EXCERPT:\n{topic_excerpt[:6000]}\n\n"
        f"PHASE: {phase}\n"
        f"Generate {n} substantive, non-trivial questions. Avoid asking purely definitional questions.\n"
        f"Seed for diversity: {seed}"
    )
    data = llm.complete_json(system=GENERATE_SYSTEM, user=user, max_tokens=2000)
    out: list[QuizQuestion] = []
    for q in data.get("questions", [])[:n]:
        out.append(QuizQuestion(question=q["question"], rubric=q["rubric"]))
    return out


def _fetch_topic_excerpt(conn, topic_id: str, max_bodies: int = 6) -> str:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT body FROM semantic_items WHERE topic_id = %s "
            "AND artifact_type IN ('concept', 'example', 'paper_claim') "
            "ORDER BY random() LIMIT %s",
            (topic_id, max_bodies),
        )
        return "\n\n".join(r["body"] for r in cur.fetchall())


def _run_quiz_session(
    *,
    phase: str,
    student_id: str,
    topic_id: str,
    n: int,
    seed: int,
):
    settings = get_settings()
    llm = LLM(api_key=settings.anthropic_api_key)
    log_path = settings.log_dir / "quiz.jsonl"

    conn = connect(settings.database_url)
    try:
        student = StudentStore(conn)
        student.ensure_student(student_id)
        excerpt = _fetch_topic_excerpt(conn, topic_id)
        if not excerpt:
            typer.echo(f"No source material for topic '{topic_id}'.", err=True)
            raise typer.Exit(2)

        questions = _generate_quiz(llm, topic_id, excerpt, n, phase, seed)
        if not questions:
            typer.echo("Quiz generation produced 0 questions.", err=True)
            raise typer.Exit(2)

        answers: list[str] = []
        for i, q in enumerate(questions, start=1):
            typer.echo(f"\n--- Q{i}/{len(questions)} ({phase}) ---")
            typer.echo(f"{q.question}")
            typer.echo(f"(rubric: {q.rubric})")
            try:
                ans = input("Your answer (blank to skip):\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                ans = ""
            answers.append(ans)

        scores = [score_answer(question=q, student_answer=a, judge_llm=llm)
                  for q, a in zip(questions, answers)]
        avg = average_score(scores)
        typer.echo(f"\nQuiz average ({phase}): {avg:.2f}")
        for i, s in enumerate(scores, start=1):
            typer.echo(f"  Q{i}: {s.score:.2f}  ({s.rationale})")

        append_quiz_log(
            log_path,
            student_id=student_id,
            topic_id=topic_id,
            phase=phase,
            questions=questions,
            answers=answers,
            scores=scores,
        )
        typer.echo(f"\nLogged to {log_path}")
        conn.commit()
    finally:
        conn.close()


@app.command()
def pre(
    student_id: str = typer.Option(..., "--student-id"),
    topic_id: str = typer.Option(..., "--topic-id"),
    n: int = typer.Option(5, "-n", "--num-questions"),
    seed: int = typer.Option(7, "--seed"),
):
    """Run a PRE quiz (before studying)."""
    _run_quiz_session(phase="pre", student_id=student_id, topic_id=topic_id, n=n, seed=seed)


@app.command()
def post(
    student_id: str = typer.Option(..., "--student-id"),
    topic_id: str = typer.Option(..., "--topic-id"),
    n: int = typer.Option(5, "-n", "--num-questions"),
    seed: int = typer.Option(42, "--seed"),
):
    """Run a POST quiz (after studying)."""
    _run_quiz_session(phase="post", student_id=student_id, topic_id=topic_id, n=n, seed=seed)


@app.command()
def gain(
    student_id: str = typer.Option(..., "--student-id"),
    topic_id: str = typer.Option(..., "--topic-id"),
):
    """Compute pre vs post averages for a student/topic from the quiz log."""
    settings = get_settings()
    log_path = settings.log_dir / "quiz.jsonl"
    if not log_path.exists():
        typer.echo("No quiz log yet.", err=True)
        raise typer.Exit(2)
    by_phase: dict[str, list[float]] = defaultdict(list)
    with log_path.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec["student_id"] != student_id or rec["topic_id"] != topic_id:
                continue
            by_phase[rec["phase"]].append(rec["average"])
    pre_avg = sum(by_phase.get("pre", [0.0])) / max(1, len(by_phase.get("pre", [])))
    post_avg = sum(by_phase.get("post", [0.0])) / max(1, len(by_phase.get("post", [])))
    typer.echo(f"student: {student_id}  topic: {topic_id}")
    typer.echo(f"  pre  sessions: {len(by_phase.get('pre', []))}  avg: {pre_avg:.2f}")
    typer.echo(f"  post sessions: {len(by_phase.get('post', []))}  avg: {post_avg:.2f}")
    typer.echo(f"  GAIN: {post_avg - pre_avg:+.2f}")


if __name__ == "__main__":
    app()

"""Pre/post quiz harness for measuring learning gain on a topic."""

import json
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from ..llm import LLM


@dataclass
class QuizQuestion:
    question: str
    rubric: str          # what a correct answer should contain
    concept_id: str | None = None


@dataclass
class QuizScore:
    score: float        # 0.0 - 1.0
    rationale: str


JUDGE_SYSTEM = """You are an expert grader for ML systems engineering quiz answers.
Given a question, a rubric, and the student's answer, output STRICT JSON:
{
  "score": <float 0.0 to 1.0>,
  "rationale": "<one-sentence reason>"
}

Score 1.0 = correct and complete. Score 0.5 = partially correct. Score 0.0 = wrong or absent.
No commentary outside JSON."""

QUIZ_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "description": "Score from 0.0 (wrong) to 1.0 (correct and complete)."},
        "rationale": {"type": "string", "description": "One-sentence reason for the score."},
    },
    "required": ["score", "rationale"],
}


def score_answer(
    *,
    question: QuizQuestion,
    student_answer: str,
    judge_llm: LLM,
) -> QuizScore:
    if not student_answer.strip():
        return QuizScore(score=0.0, rationale="empty answer")
    user = (
        f"QUESTION: {question.question}\n\n"
        f"RUBRIC: {question.rubric}\n\n"
        f"STUDENT ANSWER:\n{student_answer}"
    )
    data = judge_llm.complete_with_schema(
        system=JUDGE_SYSTEM,
        user=user,
        schema=QUIZ_SCORE_SCHEMA,
        tool_name="submit_score",
        tool_description="Submit the score and rationale.",
        max_tokens=512,
    )
    raw_score = data.get("score", 0.0)
    try:
        s = float(raw_score)
    except (TypeError, ValueError):
        s = 0.0
    s = max(0.0, min(1.0, s))
    rationale = str(data.get("rationale", ""))[:500]
    return QuizScore(score=s, rationale=rationale)


def average_score(scores: list[QuizScore]) -> float:
    if not scores:
        return 0.0
    return sum(s.score for s in scores) / len(scores)


def append_quiz_log(
    log_path: Path,
    *,
    student_id: str,
    topic_id: str,
    phase: str,   # "pre" or "post"
    questions: list[QuizQuestion],
    answers: list[str],
    scores: list[QuizScore],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "student_id": student_id,
            "topic_id": topic_id,
            "phase": phase,
            "items": [
                {
                    "question": q.question,
                    "rubric": q.rubric,
                    "answer": a,
                    "score": s.score,
                    "rationale": s.rationale,
                }
                for q, a, s in zip(questions, answers, scores)
            ],
            "average": average_score(scores),
        }) + "\n")

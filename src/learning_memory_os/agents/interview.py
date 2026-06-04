"""Design-interview agent: generate ML-systems design questions and judge answers.

Thin orchestration over the LLM provider + the prompts/schema in
``interview_prompts``. The judge returns the structured Evaluation dict that the
API persists and uses to update the student skill model.
"""
from .interview_prompts import (
    INTERVIEWER_SYSTEM, JUDGE_SYSTEM, EVALUATION_SCHEMA, CATEGORIES,
    DEBUG_INCIDENT_SYSTEM, DEBUG_JUDGE_SYSTEM, DEBUG_EVAL_SCHEMA, DEBUG_CATEGORIES,
    weighted_overall,
)


def _normalize(ev: dict, categories: list[str]) -> dict:
    ev.setdefault("category_scores", {})
    for c in categories:
        ev["category_scores"].setdefault(c, ev.get("overall_score", 0))
    for k in ("strengths", "weaknesses", "misconceptions"):
        ev.setdefault(k, [])
    ev.setdefault("overall_score", 0)
    ev.setdefault("next_topic", "")
    ev.setdefault("recommended_exercise_type", "mock_interview")
    ev.setdefault("improved_answer", "")
    return ev


class InterviewAgent:
    def __init__(self, llm):
        self.llm = llm

    def generate_question(self, *, topic_title: str, level: str = "intermediate",
                          goal: str | None = None) -> str:
        goal_line = f"\nCandidate goal: {goal}." if goal else ""
        user = (
            f"Topic to probe: {topic_title}\n"
            f"Candidate level: {level}.{goal_line}\n\n"
            "Write the interview question now."
        )
        return self.llm.complete(system=INTERVIEWER_SYSTEM, user=user, max_tokens=400).strip()

    def evaluate(self, *, question: str, answer: str, topic_title: str,
                 level: str = "intermediate", profile_summary: str | None = None,
                 role: str = "ml_infra") -> dict:
        ctx = f"\nWhat we know about this student: {profile_summary}" if profile_summary else ""
        user = (
            f"TOPIC: {topic_title}\nLEVEL: {level}{ctx}\n\n"
            f"INTERVIEW QUESTION:\n{question}\n\n"
            f"CANDIDATE ANSWER:\n{answer}\n\n"
            "Grade strictly against every rubric category and return the structured evaluation."
        )
        ev = self.llm.complete_with_schema(
            system=JUDGE_SYSTEM, user=user, schema=EVALUATION_SCHEMA,
            tool_name="submit_evaluation",
            tool_description="Submit the structured ML-systems interview evaluation.",
            max_tokens=2048,
        )
        ev = _normalize(ev, CATEGORIES)
        # Override the LLM's holistic number with the staff-interviewer weighted sum
        # (fixes over-weighting communication).
        ev["overall_score"] = weighted_overall(ev["category_scores"], role)
        return ev


class DebuggingAgent:
    """Production Debugging Mode: generate realistic incidents (with simulated
    logs/metrics/configs) and grade the student's debugging *process*."""

    def __init__(self, llm):
        self.llm = llm

    def generate_incident(self, *, topic_title: str, level: str = "intermediate") -> str:
        user = f"Topic: {topic_title}\nLevel: {level}.\n\nWrite the production incident now."
        return self.llm.complete(system=DEBUG_INCIDENT_SYSTEM, user=user, max_tokens=700).strip()

    def evaluate(self, *, incident: str, diagnosis: str, topic_title: str,
                 level: str = "intermediate", profile_summary: str | None = None) -> dict:
        ctx = f"\nWhat we know about this student: {profile_summary}" if profile_summary else ""
        user = (
            f"TOPIC: {topic_title}\nLEVEL: {level}{ctx}\n\n"
            f"INCIDENT (symptom + evidence shown to the candidate):\n{incident}\n\n"
            f"CANDIDATE DIAGNOSIS / DEBUGGING:\n{diagnosis}\n\n"
            "Grade the debugging process strictly and return the structured evaluation."
        )
        ev = self.llm.complete_with_schema(
            system=DEBUG_JUDGE_SYSTEM, user=user, schema=DEBUG_EVAL_SCHEMA,
            tool_name="submit_evaluation",
            tool_description="Submit the structured debugging-process evaluation.",
            max_tokens=2048,
        )
        return _normalize(ev, DEBUG_CATEGORIES)

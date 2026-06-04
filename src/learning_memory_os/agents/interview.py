"""Design-interview agent: generate ML-systems design questions and judge answers.

Thin orchestration over the LLM provider + the prompts/schema in
``interview_prompts``. The judge returns the structured Evaluation dict that the
API persists and uses to update the student skill model.
"""
from .interview_prompts import (
    INTERVIEWER_SYSTEM, INTERVIEW_FOLLOWUP_SYSTEM, JUDGE_SYSTEM, EVALUATION_SCHEMA, CATEGORIES,
    DEBUG_INCIDENT_SYSTEM, DEBUG_JUDGE_SYSTEM, DEBUG_EVAL_SCHEMA, DEBUG_CATEGORIES,
    FORWARD_SCENARIO_SYSTEM, FORWARD_JUDGE_SYSTEM, FORWARD_EVAL_SCHEMA, FORWARD_CATEGORIES,
    weighted_overall,
)


def _format_transcript(transcript: list[dict]) -> str:
    """Render a multi-turn interview transcript as Q/A pairs for the judge."""
    out = []
    for i, t in enumerate(transcript or [], 1):
        q = (t.get("q") or "").strip()
        a = (t.get("a") or "").strip()
        label = "QUESTION" if i == 1 else f"FOLLOW-UP {i - 1}"
        out.append(f"{label}:\n{q}\n\nCANDIDATE ANSWER:\n{a}")
    return "\n\n———\n\n".join(out)


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

    def followup(self, *, topic_title: str, level: str, transcript: list[dict]) -> str:
        """Generate the next interviewer probe, targeting the weakest part of the
        candidate's latest answer. `transcript` is a list of {"q","a"} turns."""
        convo = _format_transcript(transcript)
        user = (
            f"TOPIC: {topic_title}\nLEVEL: {level}\n\n"
            f"INTERVIEW SO FAR:\n{convo}\n\n"
            "Write your next follow-up question now."
        )
        return self.llm.complete(system=INTERVIEW_FOLLOWUP_SYSTEM, user=user, max_tokens=300).strip()

    def evaluate(self, *, question: str = "", answer: str = "", topic_title: str,
                 level: str = "intermediate", profile_summary: str | None = None,
                 role: str = "ml_infra", transcript: list[dict] | None = None) -> dict:
        ctx = f"\nWhat we know about this student: {profile_summary}" if profile_summary else ""
        if transcript:
            body = (
                f"This was a MULTI-TURN interview ({len(transcript)} exchanges). Grade the "
                f"candidate's reasoning across the WHOLE conversation, including how they "
                f"responded to follow-up probes:\n\n{_format_transcript(transcript)}"
            )
        else:
            body = f"INTERVIEW QUESTION:\n{question}\n\nCANDIDATE ANSWER:\n{answer}"
        user = (
            f"TOPIC: {topic_title}\nLEVEL: {level}{ctx}\n\n"
            f"{body}\n\n"
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


class ForwardDeployedAgent:
    """Forward-deployed engineer mode: a vague customer problem ("our agent feels
    slow") graded on the 7 sub-skills that separate forward-deployed work from pure
    backend engineering — framing, metric selection, localization, hypothesis
    iteration, tradeoffs, cost/business awareness, and customer communication."""

    def __init__(self, llm):
        self.llm = llm

    def generate_scenario(self, *, topic_title: str, level: str = "intermediate") -> str:
        user = f"Topic to ground it in: {topic_title}\nLevel: {level}.\n\nWrite the customer scenario now."
        return self.llm.complete(system=FORWARD_SCENARIO_SYSTEM, user=user, max_tokens=600).strip()

    def evaluate(self, *, scenario: str, response: str, topic_title: str,
                 level: str = "intermediate", profile_summary: str | None = None) -> dict:
        ctx = f"\nWhat we know about this student: {profile_summary}" if profile_summary else ""
        user = (
            f"TOPIC: {topic_title}\nLEVEL: {level}{ctx}\n\n"
            f"CUSTOMER SCENARIO (what the candidate was given):\n{scenario}\n\n"
            f"CANDIDATE'S HANDLING (questions asked, diagnosis, fix, customer explanation):\n{response}\n\n"
            "Grade the 7 forward-deployed sub-skills strictly and return the structured evaluation."
        )
        ev = self.llm.complete_with_schema(
            system=FORWARD_JUDGE_SYSTEM, user=user, schema=FORWARD_EVAL_SCHEMA,
            tool_name="submit_evaluation",
            tool_description="Submit the structured forward-deployed evaluation.",
            max_tokens=2048,
        )
        return _normalize(ev, FORWARD_CATEGORIES)

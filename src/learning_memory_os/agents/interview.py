"""Design-interview agent: generate ML-systems design questions and judge answers.

Thin orchestration over the LLM provider + the prompts/schema in
``interview_prompts``. The judge returns the structured Evaluation dict that the
API persists and uses to update the student skill model.
"""
from .interview_prompts import (
    INTERVIEWER_SYSTEM, INTERVIEW_FOLLOWUP_SYSTEM, JUDGE_SYSTEM, EVALUATION_SCHEMA, CATEGORIES,
    DEBUG_INCIDENT_SYSTEM, DEBUG_JUDGE_SYSTEM, DEBUG_EVAL_SCHEMA, DEBUG_CATEGORIES,
    FORWARD_SCENARIO_SYSTEM, FORWARD_JUDGE_SYSTEM, FORWARD_EVAL_SCHEMA, FORWARD_CATEGORIES,
    FORWARD_CUSTOMER_SYSTEM,
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


def _require_eval(ev: dict) -> dict:
    """Guard against a degenerate judge response (model returned without calling the
    tool). Persisting such an eval would write a bogus 0-score interview into the
    readiness history and mastery, so we reject it and let the caller retry."""
    if not ev or not isinstance(ev, dict) or not ev.get("category_scores"):
        raise ValueError("judge returned no structured evaluation")
    return ev


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
                          goal: str | None = None, known: str | None = None) -> str:
        goal_line = f"\nCandidate goal: {goal}." if goal else ""
        # Condition on what the student already knows / gets wrong so the question
        # targets their actual gaps rather than asking something generic.
        known_line = (f"\nWhat this candidate already knows / struggles with on this topic: {known}. "
                      "Aim the question at their weak edge — don't re-test what they've mastered.") if known else ""
        user = (
            f"Topic to probe: {topic_title}\n"
            f"Candidate level: {level}.{goal_line}{known_line}\n\n"
            "Write the interview question now."
        )
        return self.llm.complete(system=INTERVIEWER_SYSTEM, user=user, max_tokens=400).strip()

    # A real interview tests RANGE, not just depth — so each follow-up targets a new
    # axis rather than always drilling the weakest answer (expert feedback).
    FOLLOWUP_AXES = {
        2: "Drill into the WEAKEST or vaguest technical claim in their last answer — make them defend or correct it.",
        3: "Force a QUANTITATIVE calculation: make them estimate something concrete (KV-cache bytes/token, memory footprint, tokens/s, p99 budget) given the scenario.",
        4: "Inject a PRODUCTION DEBUGGING twist: a symptom appears in this system (e.g. p99 regression, OOM under burst, throughput drop) — ask them to diagnose it.",
        5: "Ask a TRADEOFF / STAKEHOLDER question: have them justify a cost/latency/quality tradeoff and explain it to a non-expert.",
    }

    def followup(self, *, topic_title: str, level: str, transcript: list[dict], turn: int | None = None) -> str:
        """Generate the next interviewer probe. `turn` is the upcoming turn number;
        turns 2-5 each target a distinct axis so the interview covers range, not just
        depth. `transcript` is a list of {"q","a"} turns."""
        convo = _format_transcript(transcript)
        t = turn or (len(transcript) + 1)
        axis = self.FOLLOWUP_AXES.get(t) or self.FOLLOWUP_AXES[2]
        user = (
            f"TOPIC: {topic_title}\nLEVEL: {level}\n\n"
            f"INTERVIEW SO FAR:\n{convo}\n\n"
            f"This is follow-up turn {t}. FOCUS: {axis}\n\n"
            "Write your next follow-up question now (one question, on that focus)."
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
            max_tokens=4096,
        )
        ev = _normalize(_require_eval(ev), CATEGORIES)
        # A FULL multi-turn interview is expected to cover breadth, so we grade it on
        # the strict staff-interviewer weighted sum (communication can't carry it). A
        # SINGLE answer is judged on what it reasonably covered — trust the judge's
        # band-calibrated holistic score so a strong-but-partial answer isn't tanked
        # by dimensions the question never asked about.
        if transcript and len(transcript) > 1:
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
            max_tokens=4096,
        )
        return _normalize(_require_eval(ev), DEBUG_CATEGORIES)


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

    def respond(self, *, scenario: str, transcript: list[dict], message: str,
                topic_title: str) -> str:
        """Customer/system reply in an INTERACTIVE forward-deployed session — reveals
        metrics only when the engineer asks the right diagnostic question. `transcript`
        is prior [{"student","customer"}] exchanges; `message` is the engineer's latest."""
        convo = "\n".join(
            "ENGINEER: {}\nCUSTOMER/SYSTEM: {}".format(t.get("student", ""), t.get("customer", ""))
            for t in (transcript or [])
        )
        convo_block = f"CONVERSATION SO FAR:\n{convo}\n\n" if convo else ""
        user = (
            f"TOPIC: {topic_title}\n\nORIGINAL SCENARIO:\n{scenario}\n\n"
            f"{convo_block}"
            f"ENGINEER'S LATEST MESSAGE:\n{message}\n\n"
            "Reply now as the customer/system."
        )
        return self.llm.complete(system=FORWARD_CUSTOMER_SYSTEM, user=user, max_tokens=400).strip()

    def evaluate(self, *, scenario: str, response: str = "", topic_title: str,
                 level: str = "intermediate", profile_summary: str | None = None,
                 transcript: list[dict] | None = None) -> dict:
        ctx = f"\nWhat we know about this student: {profile_summary}" if profile_summary else ""
        if transcript:
            convo = "\n".join(
                f"ENGINEER: {t.get('student','')}\nCUSTOMER/SYSTEM: {t.get('customer','')}"
                for t in transcript
            )
            handling = (
                "This was an INTERACTIVE session. Grade how the engineer DROVE it — did they "
                "ask for the right metrics before guessing, localize from the evidence, and "
                "land a fix + customer explanation?\n\n"
                f"CONVERSATION:\n{convo}" + (f"\n\nFINAL DIAGNOSIS:\n{response}" if response else "")
            )
        else:
            handling = f"CANDIDATE'S HANDLING (questions asked, diagnosis, fix, customer explanation):\n{response}"
        user = (
            f"TOPIC: {topic_title}\nLEVEL: {level}{ctx}\n\n"
            f"CUSTOMER SCENARIO (what the candidate was given):\n{scenario}\n\n"
            f"{handling}\n\n"
            "Grade the 7 forward-deployed sub-skills strictly and return the structured evaluation."
        )
        ev = self.llm.complete_with_schema(
            system=FORWARD_JUDGE_SYSTEM, user=user, schema=FORWARD_EVAL_SCHEMA,
            tool_name="submit_evaluation",
            tool_description="Submit the structured forward-deployed evaluation.",
            max_tokens=4096,
        )
        return _normalize(_require_eval(ev), FORWARD_CATEGORIES)

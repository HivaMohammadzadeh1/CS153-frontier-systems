"""Learning-style inference — Loop 1 of the continuous-improvement design.

Infers HOW a student learns from signals we already log (questions, quizzes,
timing, mastery, misconceptions), so the tutor can adapt its teaching. Kept cheap
(2 light queries) and honest (returns "new learner" until there's enough history).

`_classify` is a pure function of aggregated signals so it's unit-testable without
a database; `compute_style` does the SQL then delegates to it.
"""
from dataclasses import dataclass, field
import statistics

import psycopg


@dataclass
class LearningStyle:
    summary: str = ""                       # one line for the tutor prompt / "Your AI"
    dimensions: dict = field(default_factory=dict)   # name -> {value, label}

    def to_dict(self) -> dict:
        return {"summary": self.summary, "dimensions": self.dimensions}


def _classify(
    *, n_questions: int, n_topics: int, n_quizzes: int,
    avg_quiz: float | None, overall_mastery: float, median_gap_sec: float | None,
    n_misconceptions: int,
) -> LearningStyle:
    if n_questions < 3:
        return LearningStyle(summary="New learner — not enough history to personalize teaching style yet.")

    dims: dict = {}
    parts: list[str] = []

    # Level: blended mastery + quiz performance -> scaffolding vs. rigor.
    level = overall_mastery or 0.0
    if avg_quiz is not None:
        level = 0.6 * level + 0.4 * avg_quiz
    lvl_label = "building foundations" if level < 0.4 else "intermediate" if level < 0.7 else "advanced"
    dims["level"] = {"value": round(level, 2), "label": lvl_label}
    parts.append({
        "building foundations": "still building foundations — keep it concrete and scaffolded",
        "intermediate": "intermediate — can handle applied reasoning",
        "advanced": "advanced — push depth, edge cases, and tradeoffs",
    }[lvl_label])

    # Depth vs. breadth: questions per distinct topic.
    qpt = n_questions / max(1, n_topics)
    if qpt >= 3:
        depth_label = "depth-first"; parts.append("learns depth-first (digs deep into one topic)")
    elif qpt <= 1.4:
        depth_label = "breadth-first"; parts.append("explores breadth-first across topics")
    else:
        depth_label = "balanced"
    dims["depth"] = {"value": round(qpt, 2), "label": depth_label}

    # Self-testing: quizzes per question -> active recall vs. passive.
    qz = n_quizzes / max(1, n_questions)
    if qz >= 0.3:
        st_label = "active recall"; parts.append("self-tests often — lean on retrieval practice")
    elif qz < 0.1:
        st_label = "passive"; parts.append("rarely self-tests — proactively offer quick checks")
    else:
        st_label = "moderate"
    dims["self_testing"] = {"value": round(qz, 2), "label": st_label}

    # Pace: median within-session gap between turns.
    if median_gap_sec is not None:
        pace_label = "fast" if median_gap_sec < 45 else "deliberate" if median_gap_sec > 180 else "steady"
        dims["pace"] = {"value": int(median_gap_sec), "label": pace_label}

    if n_misconceptions >= 2:
        parts.append("carries a few active misconceptions — confront them directly")

    return LearningStyle(summary="; ".join(parts[:3]) + ".", dimensions=dims)


def compute_style(conn: psycopg.Connection, student_id: str) -> LearningStyle:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT occurred_at, payload->>'topic_id' AS t FROM episodic_events "
            "WHERE student_id = %s AND event_type = 'question' ORDER BY occurred_at",
            (student_id,),
        )
        rows = cur.fetchall()
        n_questions = len(rows)
        topics = {r["t"] for r in rows if r["t"]}
        gaps = []
        for a, b in zip(rows, rows[1:]):
            d = (b["occurred_at"] - a["occurred_at"]).total_seconds()
            if 0 < d < 1800:  # ignore cross-session breaks (>30 min)
                gaps.append(d)
        median_gap = statistics.median(gaps) if gaps else None

        cur.execute(
            "SELECT (payload->>'score')::float AS s FROM episodic_events "
            "WHERE student_id = %s AND event_type = 'quiz_attempt' AND payload ? 'score'",
            (student_id,),
        )
        scores = [r["s"] for r in cur.fetchall() if r["s"] is not None]

    from ..memory.student import StudentStore
    st = StudentStore(conn)
    mastery = st.mastery_for(student_id)
    overall = (sum(m.score for m in mastery) / len(mastery)) if mastery else 0.0
    n_misc = len(st.active_misconceptions(student_id))

    return _classify(
        n_questions=n_questions, n_topics=len(topics), n_quizzes=len(scores),
        avg_quiz=(sum(scores) / len(scores) if scores else None),
        overall_mastery=overall, median_gap_sec=median_gap, n_misconceptions=n_misc,
    )

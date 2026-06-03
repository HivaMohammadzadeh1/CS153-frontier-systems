"""LearnerProfile — the per-user adaptation snapshot.

Consolidates a student's decayed mastery, strengths/gaps, active misconceptions,
and due-for-review concepts into one structure that both (a) calibrates the tutor
prompt and (b) backs the "Your AI" view. Built fresh each turn so the agent
always adapts to the latest state.
"""

from dataclasses import dataclass, field

import psycopg

from ..memory.decay import effective_score
from ..memory.student import StudentStore


@dataclass
class LearnerProfile:
    student_id: str
    overall_mastery: float = 0.0
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    misconceptions: list[str] = field(default_factory=list)
    due_for_review: list[str] = field(default_factory=list)
    streak: int = 0
    learning_style: str = ""        # one-line "how this student learns" summary

    def prompt_block(self) -> str:
        """Render the STUDENT PROFILE block injected into the tutor system prompt."""
        parts = []
        if self.weaknesses:
            parts.append(f"- Mastery is LOW for: {', '.join(self.weaknesses)}")
        if self.strengths:
            parts.append(f"- Mastery is HIGH for: {', '.join(self.strengths)}")
        if self.misconceptions:
            parts.append(f"- Active misconceptions to address: {'; '.join(self.misconceptions)}")
        if self.due_for_review:
            parts.append(f"- Due for review (refresh gently if relevant): {', '.join(self.due_for_review)}")
        if self.learning_style:
            parts.append(f"- Learning style: {self.learning_style}")
        if not parts:
            return ""
        pct = round(self.overall_mastery * 100)
        return f"STUDENT PROFILE (overall mastery {pct}%):\n" + "\n".join(parts) + "\n\n"

    def to_dict(self) -> dict:
        return {
            "student_id": self.student_id,
            "overall_mastery": round(self.overall_mastery, 3),
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "misconceptions": self.misconceptions,
            "due_for_review": self.due_for_review,
            "streak": self.streak,
            "learning_style": self.learning_style,
        }


def build_profile(conn: psycopg.Connection, student_id: str, *, streak: int = 0) -> LearnerProfile:
    student = StudentStore(conn)
    student.ensure_student(student_id)
    mastery = student.mastery_for(student_id)

    # Decay-adjusted score per concept.
    scored = [
        (m.concept_id, effective_score(m.score, m.confidence, m.last_updated), m.confidence)
        for m in mastery
    ]
    weak_ids = [cid for cid, eff, conf in scored if eff < 0.4 and conf > 0.2]
    strong_ids = [cid for cid, eff, conf in scored if eff > 0.7 and conf > 0.3]
    due_ids = student.due_for_review(student_id)

    # Infer how this student learns (Loop 1). Best-effort: never break the profile.
    try:
        from .learning_style import compute_style
        style_summary = compute_style(conn, student_id).summary
    except Exception:
        style_summary = ""

    title = _title_lookup(conn, set(weak_ids + strong_ids + due_ids))
    overall = 0.0
    total_conf = sum(c for _, _, c in scored)
    if total_conf:
        overall = sum(eff * c for _, eff, c in scored) / total_conf

    return LearnerProfile(
        student_id=student_id,
        overall_mastery=overall,
        strengths=[title[i] for i in strong_ids if i in title][:5],
        weaknesses=[title[i] for i in weak_ids if i in title][:5],
        misconceptions=[m["description"][:120] for m in student.active_misconceptions(student_id)][:3],
        due_for_review=[title[i] for i in due_ids if i in title][:5],
        streak=streak,
        learning_style=style_summary,
    )


def _title_lookup(conn: psycopg.Connection, ids: set[str]) -> dict[str, str]:
    if not ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id::text AS id, title FROM semantic_items WHERE id::text = ANY(%s)",
            (list(ids),),
        )
        return {r["id"]: r["title"] for r in cur.fetchall()}

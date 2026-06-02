import psycopg
from ..schemas.memory import MasteryEntry


class StudentStore:
    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    def ensure_student(self, student_id: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO students (id) VALUES (%s) ON CONFLICT DO NOTHING",
                (student_id,),
            )

    def update_mastery(
        self,
        student_id: str,
        concept_id: str,
        score: float,
        confidence: float,
    ) -> None:
        self.ensure_student(student_id)
        with self.conn.cursor() as cur:
            # Confidence-weighted running average: new evidence is blended with
            # the prior in proportion to confidence, so a single noisy quiz can't
            # wipe accumulated history. Confidence itself accrues toward 1.0.
            cur.execute(
                """
                INSERT INTO mastery (student_id, concept_id, score, confidence)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (student_id, concept_id) DO UPDATE SET
                    score = COALESCE(
                        (mastery.confidence * mastery.score
                            + EXCLUDED.confidence * EXCLUDED.score)
                        / NULLIF(mastery.confidence + EXCLUDED.confidence, 0),
                        EXCLUDED.score),
                    confidence = LEAST(
                        1.0,
                        mastery.confidence + EXCLUDED.confidence * (1 - mastery.confidence)),
                    -- SM-2-lite: a passing grade (>=0.6) advances the schedule,
                    -- a failing grade resets it to a 1-day interval.
                    reps = CASE WHEN EXCLUDED.score >= 0.6 THEN mastery.reps + 1 ELSE 0 END,
                    interval_days = CASE
                        WHEN EXCLUDED.score < 0.6 THEN 1
                        WHEN mastery.reps <= 0 THEN 1
                        WHEN mastery.reps = 1 THEN 3
                        ELSE LEAST(mastery.interval_days * 2.0, 365)
                    END,
                    next_review_at = now() + ((CASE
                        WHEN EXCLUDED.score < 0.6 THEN 1
                        WHEN mastery.reps <= 0 THEN 1
                        WHEN mastery.reps = 1 THEN 3
                        ELSE LEAST(mastery.interval_days * 2.0, 365)
                    END) || ' days')::interval,
                    last_updated = now()
                """,
                (student_id, concept_id, score, confidence),
            )

    def due_for_review(self, student_id: str) -> list[str]:
        """Concept ids whose spaced-repetition review is due (next_review_at <= now)."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT concept_id::text AS concept_id
                FROM mastery
                WHERE student_id = %s
                  AND next_review_at IS NOT NULL
                  AND next_review_at <= now()
                ORDER BY next_review_at ASC
                """,
                (student_id,),
            )
            return [r["concept_id"] for r in cur.fetchall()]

    def mastery_for(self, student_id: str) -> list[MasteryEntry]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT student_id, concept_id::text, score, confidence, last_updated
                FROM mastery WHERE student_id = %s
                """,
                (student_id,),
            )
            return [MasteryEntry(**r) for r in cur.fetchall()]

    def record_misconception(
        self,
        student_id: str,
        *,
        concept_id: str | None,
        description: str,
        evidence: str | None = None,
        topic_id: str | None = None,
    ) -> str:
        self.ensure_student(student_id)
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO misconceptions (student_id, concept_id, description, evidence, topic_id)
                VALUES (%s, %s, %s, %s, %s) RETURNING id::text
                """,
                (student_id, concept_id, description, evidence, topic_id),
            )
            return cur.fetchone()["id"]

    def active_misconceptions(self, student_id: str) -> list[dict]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id::text, description, evidence, concept_id::text, topic_id, detected_at
                FROM misconceptions
                WHERE student_id = %s AND resolved = FALSE
                ORDER BY detected_at DESC
                """,
                (student_id,),
            )
            return list(cur.fetchall())

    def resolve_misconception(self, misconception_id: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE misconceptions
                SET resolved = TRUE, resolved_at = now()
                WHERE id = %s
                """,
                (misconception_id,),
            )

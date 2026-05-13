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
            cur.execute(
                """
                INSERT INTO mastery (student_id, concept_id, score, confidence)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (student_id, concept_id) DO UPDATE SET
                    score = EXCLUDED.score,
                    confidence = EXCLUDED.confidence,
                    last_updated = now()
                """,
                (student_id, concept_id, score, confidence),
            )

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
    ) -> str:
        self.ensure_student(student_id)
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO misconceptions (student_id, concept_id, description, evidence)
                VALUES (%s, %s, %s, %s) RETURNING id::text
                """,
                (student_id, concept_id, description, evidence),
            )
            return cur.fetchone()["id"]

    def active_misconceptions(self, student_id: str) -> list[dict]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id::text, description, evidence, concept_id::text, detected_at
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

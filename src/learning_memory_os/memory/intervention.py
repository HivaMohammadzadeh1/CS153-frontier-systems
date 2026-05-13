import psycopg


class InterventionStore:
    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    def record(
        self,
        *,
        student_id: str,
        misconception_id: str | None,
        strategy: str,
        outcome: str | None = None,
        notes: str | None = None,
    ) -> str:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO interventions
                    (student_id, misconception_id, strategy, outcome, notes)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id::text
                """,
                (student_id, misconception_id, strategy, outcome, notes),
            )
            return cur.fetchone()["id"]

    def for_student(self, student_id: str) -> list[dict]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id::text, strategy, outcome, notes, occurred_at,
                       misconception_id::text AS misconception_id
                FROM interventions
                WHERE student_id = %s
                ORDER BY occurred_at DESC
                """,
                (student_id,),
            )
            return list(cur.fetchall())

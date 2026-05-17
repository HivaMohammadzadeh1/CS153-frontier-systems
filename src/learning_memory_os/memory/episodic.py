import json
import psycopg
from ..schemas.memory import EpisodicEvent
from .store import vec_literal


class EpisodicStore:
    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    def append(
        self,
        *,
        student_id: str,
        event_type: str,
        payload: dict,
        embedding: list[float] | None = None,
        conversation_id: str | None = None,
    ) -> str:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO episodic_events (student_id, event_type, payload, embedding, conversation_id, occurred_at)
                VALUES (%s, %s, %s::jsonb, %s::vector, %s::uuid, clock_timestamp())
                RETURNING id::text
                """,
                (
                    student_id,
                    event_type,
                    json.dumps(payload),
                    vec_literal(embedding) if embedding else None,
                    conversation_id,
                ),
            )
            return cur.fetchone()["id"]

    def recent(self, student_id: str, limit: int = 20) -> list[EpisodicEvent]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id::text, student_id, event_type, payload, occurred_at
                FROM episodic_events
                WHERE student_id = %s
                ORDER BY occurred_at DESC
                LIMIT %s
                """,
                (student_id, limit),
            )
            return [EpisodicEvent(**r) for r in cur.fetchall()]

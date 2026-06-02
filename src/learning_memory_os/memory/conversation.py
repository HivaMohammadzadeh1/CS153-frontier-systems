import psycopg


class ConversationStore:
    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    def create(self, student_id: str, title: str = "New chat") -> str:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO conversations (student_id, title) VALUES (%s, %s) RETURNING id::text",
                (student_id, title),
            )
            return cur.fetchone()["id"]

    def list_for_student(self, student_id: str, limit: int = 100) -> list[dict]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id::text, title, created_at, last_message_at
                FROM conversations
                WHERE student_id = %s
                ORDER BY last_message_at DESC
                LIMIT %s
                """,
                (student_id, limit),
            )
            return list(cur.fetchall())

    def messages(self, conversation_id: str) -> list[dict]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT event_type, payload, occurred_at
                FROM episodic_events
                WHERE conversation_id = %s::uuid AND event_type IN ('question', 'tutor_reply')
                ORDER BY occurred_at ASC
                """,
                (conversation_id,),
            )
            return list(cur.fetchall())

    def set_title(self, conversation_id: str, title: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE conversations SET title = %s WHERE id = %s::uuid",
                (title, conversation_id),
            )

    def touch(self, conversation_id: str) -> None:
        """Bump last_message_at to now so the conversation rises to the top of the list."""
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE conversations SET last_message_at = now() WHERE id = %s::uuid",
                (conversation_id,),
            )

    def get_title(self, conversation_id: str) -> str | None:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT title FROM conversations WHERE id = %s::uuid",
                (conversation_id,),
            )
            row = cur.fetchone()
            return row["title"] if row else None

    def owner(self, conversation_id: str) -> str | None:
        """The student_id that owns a conversation (for access-control checks)."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT student_id FROM conversations WHERE id = %s::uuid",
                (conversation_id,),
            )
            row = cur.fetchone()
            return row["student_id"] if row else None

    def delete(self, conversation_id: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM conversations WHERE id = %s::uuid", (conversation_id,))

import json
import psycopg
from ..schemas.memory import MemoryItem
from .store import vec_literal


class SemanticStore:
    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    def insert(self, item: MemoryItem) -> str:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO semantic_items
                    (topic_id, artifact_type, title, body, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s::vector)
                RETURNING id
                """,
                (
                    item.topic_id,
                    item.artifact_type.value if item.artifact_type else "concept",
                    item.title,
                    item.body,
                    json.dumps(item.metadata),
                    vec_literal(item.embedding) if item.embedding else None,
                ),
            )
            row = cur.fetchone()
            return str(row["id"])

    def by_topic(self, topic_id: str) -> list[MemoryItem]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id::text, topic_id, artifact_type, title, body, metadata,
                       embedding::text AS embedding
                FROM semantic_items WHERE topic_id = %s ORDER BY created_at
                """,
                (topic_id,),
            )
            return [self._row_to_item(r) for r in cur.fetchall()]

    def vector_search(self, *, query: list[float], k: int = 5) -> list[MemoryItem]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id::text, topic_id, artifact_type, title, body, metadata,
                       (embedding <=> %s::vector) AS distance
                FROM semantic_items
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (vec_literal(query), vec_literal(query), k),
            )
            return [self._row_to_item(r) for r in cur.fetchall()]

    @staticmethod
    def _row_to_item(r: dict) -> MemoryItem:
        body = r["body"] or ""
        raw_emb = r.get("embedding")
        return MemoryItem(
            id=r["id"],
            tier="semantic",
            artifact_type=r["artifact_type"],
            topic_id=r["topic_id"],
            title=r["title"],
            body=body,
            token_estimate=max(1, len(body) // 4),
            metadata=r["metadata"] or {},
            embedding=_parse_vec(raw_emb) if raw_emb else [],
        )

    def count_by_topic(self, topic_id: str) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) AS n FROM semantic_items WHERE topic_id = %s",
                (topic_id,),
            )
            return int(cur.fetchone()["n"])

    def delete_by_topic(self, topic_id: str) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM semantic_items WHERE topic_id = %s",
                (topic_id,),
            )
            return cur.rowcount


def _parse_vec(s: str | None) -> list[float]:
    if not s:
        return []
    # pgvector text format: '[0.1,0.2,0.3]' (no spaces)
    return [float(x) for x in s.strip("[]").split(",") if x]

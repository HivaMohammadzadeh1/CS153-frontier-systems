import random
import psycopg
from .schemas import PoolItem, StudentState


def sample_candidate_pool(
    conn: psycopg.Connection,
    *,
    target_topic: str,
    pool_size: int = 15,
    other_topic_noise: int = 5,
) -> list[PoolItem]:
    """Sample a realistic candidate pool: mostly target-topic artifacts + some distractors."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id::text AS id, title, body
            FROM semantic_items WHERE topic_id = %s
            ORDER BY random() LIMIT %s
            """,
            (target_topic, max(0, pool_size - other_topic_noise)),
        )
        target_rows = list(cur.fetchall())

        cur.execute(
            """
            SELECT id::text AS id, title, body
            FROM semantic_items WHERE topic_id <> %s
            ORDER BY random() LIMIT %s
            """,
            (target_topic, other_topic_noise),
        )
        noise_rows = list(cur.fetchall())

    items: list[PoolItem] = []
    for r in target_rows + noise_rows:
        body = r["body"] or ""
        # Short id = first 8 hex chars of the UUID
        short_id = r["id"].replace("-", "")[:8]
        items.append(
            PoolItem(
                id=short_id,
                title=r["title"],
                body_excerpt=body[:300],
                token_estimate=max(1, len(body) // 4),
            )
        )
    random.shuffle(items)
    return items


def sample_student_state(
    conn: psycopg.Connection,
    *,
    student_id: str,
    target_concepts: list[str],
) -> StudentState:
    """Synthesize a plausible student state: bimodal mastery + 0-2 active misconceptions."""
    mastery: dict[str, float] = {}
    for c in target_concepts:
        mastery[c] = random.choice([random.uniform(0.0, 0.4), random.uniform(0.6, 1.0)])

    misconceptions: list[str] = []
    with conn.cursor() as cur:
        cur.execute(
            "SELECT body FROM semantic_items WHERE artifact_type = 'misconception' "
            "ORDER BY random() LIMIT 2"
        )
        for r in cur.fetchall():
            if random.random() < 0.5:
                misconceptions.append((r["body"] or "")[:200])

    return StudentState(
        student_id=student_id,
        mastery=mastery,
        active_misconceptions=misconceptions,
        recent_episodic_ids=[],
    )

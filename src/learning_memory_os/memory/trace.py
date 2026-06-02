"""Per-user learning traces — captured each chat turn for later fine-tuning.

Each row is a superset of the synthetic ``Trajectory`` (student_state, task,
candidate_pool, selection) plus the tutor reply and an outcome ``reward``, so the
same captured data can train either the context router or a tutor model.
"""

import json
import psycopg

from ..trajectories.schemas import PoolItem, StudentState, TaskType, Trajectory


class TraceStore:
    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    def record_turn(
        self,
        *,
        student_id: str,
        task_text: str,
        budget: int,
        student_state: dict,
        candidate_pool: list[dict],
        selected_ids: list[str],
        dropped_ids: list[str] | None = None,
        scores: dict | None = None,
        reply: str | None = None,
        model: str | None = None,
        task_type: str = "explain",
        conversation_id: str | None = None,
        turn_ordinal: int | None = None,
    ) -> str:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO learning_traces
                    (student_id, conversation_id, turn_ordinal, task_type, task_text,
                     budget, student_state, candidate_pool, selected_ids, dropped_ids,
                     scores, reply, model, occurred_at)
                VALUES (%s, %s::uuid, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb,
                        %s::jsonb, %s::jsonb, %s, %s, clock_timestamp())
                RETURNING id::text
                """,
                (
                    student_id, conversation_id, turn_ordinal, task_type, task_text,
                    budget, json.dumps(student_state), json.dumps(candidate_pool),
                    json.dumps(selected_ids), json.dumps(dropped_ids or []),
                    json.dumps(scores or {}), reply, model,
                ),
            )
            return cur.fetchone()["id"]

    def attach_reward(self, student_id: str, reward: float) -> int:
        """Label the student's most recently captured turn with an outcome reward.

        Feedback (👍/👎) and quiz scores arrive right after the turn they concern,
        so the latest trace is the correct target.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE learning_traces SET reward = %s
                WHERE id = (
                    SELECT id FROM learning_traces
                    WHERE student_id = %s
                    ORDER BY occurred_at DESC LIMIT 1
                )
                """,
                (reward, student_id),
            )
            return cur.rowcount

    def count(self, student_id: str) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) AS n FROM learning_traces WHERE student_id = %s",
                (student_id,),
            )
            return int(cur.fetchone()["n"])

    def recent(self, student_id: str, limit: int = 20) -> list[dict]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id::text, task_type, task_text, reward, occurred_at,
                       jsonb_array_length(selected_ids) AS n_selected
                FROM learning_traces
                WHERE student_id = %s
                ORDER BY occurred_at DESC
                LIMIT %s
                """,
                (student_id, limit),
            )
            return list(cur.fetchall())

    def delete_for_student(self, student_id: str) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM learning_traces WHERE student_id = %s", (student_id,)
            )
            return cur.rowcount

    def _where(self, student_id, min_reward):
        clauses, params = [], []
        if student_id is not None:
            clauses.append("student_id = %s")
            params.append(student_id)
        if min_reward is not None:
            clauses.append("reward IS NOT NULL AND reward >= %s")
            params.append(min_reward)
        return (("WHERE " + " AND ".join(clauses)) if clauses else ""), params

    def export_records(
        self, student_id: str | None = None, *, min_reward: float | None = None
    ) -> list[dict]:
        """Rich per-turn records for tutor / behavior-cloning fine-tuning.

        Unlike ``export_trajectories`` (router format), these keep the actual
        ``reply`` and ``reward`` alongside the prompt context.
        """
        where, params = self._where(student_id, min_reward)
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id::text, student_id, task_type, task_text, budget,
                       student_state, candidate_pool, selected_ids, reply, reward, model
                FROM learning_traces {where}
                ORDER BY occurred_at ASC
                """,
                params,
            )
            return [dict(r) for r in cur.fetchall()]

    def export_trajectories(
        self, student_id: str | None = None, *, min_reward: float | None = None
    ) -> list[Trajectory]:
        """Map captured rows onto the router-training ``Trajectory`` schema.

        ``min_reward`` filters to turns whose outcome was at least that good
        (e.g. 0.0 keeps positively/neutrally-rated turns); ``None`` keeps all.
        """
        where, params = self._where(student_id, min_reward)
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id::text, student_id, task_type, task_text, budget,
                       student_state, candidate_pool, selected_ids
                FROM learning_traces {where}
                ORDER BY occurred_at ASC
                """,
                params,
            )
            rows = cur.fetchall()
        out: list[Trajectory] = []
        for r in rows:
            st = r["student_state"] or {}
            try:
                task_type = TaskType(r["task_type"])
            except ValueError:
                task_type = TaskType.EXPLAIN
            out.append(
                Trajectory(
                    id=r["id"],
                    student_state=StudentState(
                        student_id=r["student_id"],
                        mastery=st.get("mastery", {}),
                        active_misconceptions=st.get("active_misconceptions", []),
                        recent_episodic_ids=st.get("recent_episodic_ids", []),
                    ),
                    task_type=task_type,
                    task_text=r["task_text"],
                    budget=r["budget"],
                    candidate_pool=[PoolItem(**p) for p in (r["candidate_pool"] or [])],
                    oracle_selection=r["selected_ids"] or [],
                )
            )
        return out

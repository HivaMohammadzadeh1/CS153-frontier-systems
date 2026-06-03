"""Per-user learning traces — captured each chat turn for later fine-tuning.

Each row is a superset of the synthetic ``Trajectory`` (student_state, task,
candidate_pool, selection) plus the tutor reply and an outcome ``reward``, so the
same captured data can train either the context router or a tutor model.
"""

import json
import psycopg

from ..trajectories.schemas import PoolItem, StudentState, TaskType, Trajectory


def reward_weight(reward: float | None) -> int:
    """Upsampling factor for reward-weighted SFT: turns that helped the student
    learn more are repeated more; non-positive outcomes are dropped. Unlabeled
    turns (reward None) are kept once (neutral)."""
    if reward is None:
        return 1
    if reward <= 0:
        return 0
    if reward <= 0.5:
        return 1
    if reward <= 0.8:
        return 2
    return 3


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

    def label_rewards_from_mastery(self, student_id: str, *, window_min: int = 180) -> int:
        """Backfill outcome rewards from *realized mastery gain* — the learning
        signal a stateless tutor can't capture. For each turn lacking a reward,
        reward = (avg mastery in the window AFTER the turn) − (avg mastery in the
        window BEFORE), clamped to [-1, 1], read from ``mastery_history``.
        Returns the number of turns labeled."""
        n = 0
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id::text AS id, occurred_at FROM learning_traces "
                "WHERE student_id = %s AND reward IS NULL ORDER BY occurred_at",
                (student_id,),
            )
            turns = cur.fetchall()
            for t in turns:
                cur.execute(
                    "SELECT "
                    " avg(score) FILTER (WHERE occurred_at > %s AND occurred_at <= %s + (%s * interval '1 minute')) AS after, "
                    " avg(score) FILTER (WHERE occurred_at <= %s AND occurred_at >  %s - (%s * interval '1 minute')) AS before "
                    "FROM mastery_history WHERE student_id = %s",
                    (t["occurred_at"], t["occurred_at"], window_min,
                     t["occurred_at"], t["occurred_at"], window_min, student_id),
                )
                row = cur.fetchone() or {}
                a, b = row.get("after"), row.get("before")
                if a is not None and b is not None:
                    reward = max(-1.0, min(1.0, float(a) - float(b)))
                    cur.execute("UPDATE learning_traces SET reward = %s WHERE id = %s", (reward, t["id"]))
                    n += 1
        return n

    def export_trajectories(
        self, student_id: str | None = None, *, min_reward: float | None = None,
        weight_by_reward: bool = False,
    ) -> list[Trajectory]:
        """Map captured rows onto the router-training ``Trajectory`` schema.

        ``min_reward`` filters to turns whose outcome was at least that good.
        ``weight_by_reward`` upsamples each turn by ``reward_weight(reward)`` so
        fine-tuning emphasizes turns that actually helped the student learn
        (reward-weighted SFT) and drops turns with non-positive outcomes.
        """
        where, params = self._where(student_id, min_reward)
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id::text, student_id, task_type, task_text, budget,
                       student_state, candidate_pool, selected_ids, reward
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
            traj = Trajectory(
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
            copies = reward_weight(r.get("reward")) if weight_by_reward else 1
            out.extend([traj] * copies)
        return out

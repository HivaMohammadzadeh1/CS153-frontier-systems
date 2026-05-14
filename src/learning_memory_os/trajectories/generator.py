from ..llm import LLM
from .schemas import StudentState, PoolItem, Trajectory, TaskType


ORACLE_SYSTEM = """You are an expert ML systems engineer tutor selecting CONTEXT for a tutoring agent.
Given the student's state, a task, a token budget, and a pool of candidate items, you choose the SUBSET
of pool items that the tutor should use to answer the task.

Rules:
1. Total tokens of selected items MUST not exceed the budget.
2. Prefer items that directly address the task.
3. Prefer items that resolve the student's active misconceptions, if any are listed.
4. Prefer items targeting concepts the student has LOW mastery on.
5. Skip redundant items (two items that say the same thing).

Return STRICT JSON with this shape:
{
  "selected_ids": ["<short_id>", "<short_id>", ...],
  "rationale": "<one-sentence summary of why these were chosen>"
}

No commentary outside JSON."""


def _format_pool(pool: list[PoolItem]) -> str:
    return "\n\n".join(
        f"[{p.id}] (tokens={p.token_estimate}) {p.title}\n  {p.body_excerpt}"
        for p in pool
    )


def _format_state(state: StudentState) -> str:
    parts = [f"student_id: {state.student_id}"]
    if state.mastery:
        parts.append("mastery: " + ", ".join(f"{k}={v:.2f}" for k, v in state.mastery.items()))
    if state.active_misconceptions:
        parts.append("active_misconceptions: " + "; ".join(state.active_misconceptions))
    return "\n".join(parts)


def build_trajectory(
    *,
    traj_id: str,
    student_state: StudentState,
    task_type: TaskType,
    task_text: str,
    budget: int,
    candidate_pool: list[PoolItem],
    oracle_llm: LLM,
) -> Trajectory:
    user = (
        f"STUDENT STATE:\n{_format_state(student_state)}\n\n"
        f"TASK TYPE: {task_type.value}\n"
        f"TASK: {task_text}\n"
        f"TOKEN BUDGET: {budget}\n\n"
        f"CANDIDATE POOL:\n{_format_pool(candidate_pool)}"
    )
    data = oracle_llm.complete_json(system=ORACLE_SYSTEM, user=user, max_tokens=1024)
    selected = list(data.get("selected_ids", []))
    pool_ids = {p.id for p in candidate_pool}
    selected = [s for s in selected if s in pool_ids]
    return Trajectory(
        id=traj_id,
        student_state=student_state,
        task_type=task_type,
        task_text=task_text,
        budget=budget,
        candidate_pool=candidate_pool,
        oracle_selection=selected,
    )

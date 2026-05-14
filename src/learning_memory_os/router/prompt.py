import re
from ..trajectories.schemas import StudentState, PoolItem, TaskType


ROUTER_INSTRUCTION = (
    "You are a context-selection router. Given the student state, a task, a token budget, "
    "and a pool of candidate items, output ONLY a comma-separated list of the item IDs you "
    "select. No prose, no JSON, no brackets. Total tokens of selected items must not exceed "
    "the budget. Output an empty line if no items should be selected."
)


def _format_pool(pool: list[PoolItem]) -> str:
    return "\n".join(
        f"[{p.id}] (tokens={p.token_estimate}) {p.title} :: {p.body_excerpt}"
        for p in pool
    )


def _format_state(state: StudentState) -> str:
    parts = [f"id={state.student_id}"]
    if state.mastery:
        parts.append("mastery=" + ",".join(f"{k}:{v:.2f}" for k, v in state.mastery.items()))
    if state.active_misconceptions:
        parts.append("misconceptions=" + " | ".join(state.active_misconceptions))
    return "\n".join(parts)


def format_router_input(
    *,
    student_state: StudentState,
    task_type: TaskType,
    task_text: str,
    budget: int,
    candidate_pool: list[PoolItem],
) -> str:
    return (
        f"{ROUTER_INSTRUCTION}\n\n"
        f"STUDENT:\n{_format_state(student_state)}\n\n"
        f"TASK [{task_type.value}] (budget={budget}): {task_text}\n\n"
        f"POOL:\n{_format_pool(candidate_pool)}\n\n"
        f"SELECTED IDS:"
    )


_ID_RE = re.compile(r"[a-f0-9]{8}")


def parse_router_output(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    if text.strip().lower() in {"none", "[]", "{}", "null"}:
        return []
    return _ID_RE.findall(text.lower())

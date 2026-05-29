"""Frontier-API router: a strong LLM used zero-shot as the selector (upper-bound baseline)."""

from ..llm import LLM
from ..trajectories.schemas import StudentState, PoolItem, TaskType
from .prompt import format_router_input, parse_router_output


class FrontierAPIRouter:
    def __init__(self, llm: LLM):
        self.llm = llm

    def route(
        self,
        *,
        student_state: StudentState,
        task_type: TaskType,
        task_text: str,
        budget: int,
        candidate_pool: list[PoolItem],
    ) -> list[str]:
        prompt = format_router_input(
            student_state=student_state,
            task_type=task_type,
            task_text=task_text,
            budget=budget,
            candidate_pool=candidate_pool,
        )
        text = self.llm.complete(
            system="You are a context-selection router.",
            user=prompt,
            max_tokens=256,
        )
        return parse_router_output(text)

from .schemas import Trajectory
from ..router.prompt import format_router_input


def trajectory_to_training_pair(t: Trajectory) -> dict:
    """Render a trajectory as a single (input, target) text pair for SFT."""
    input_text = format_router_input(
        student_state=t.student_state,
        task_type=t.task_type,
        task_text=t.task_text,
        budget=t.budget,
        candidate_pool=t.candidate_pool,
    )
    target = ",".join(t.oracle_selection)
    return {"input": input_text, "target": target}

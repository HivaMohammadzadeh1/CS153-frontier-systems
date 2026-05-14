from enum import Enum
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    EXPLAIN = "explain"
    QUIZ = "quiz"
    REVIEW = "review"
    LAB = "lab"


class PoolItem(BaseModel):
    id: str
    title: str
    body_excerpt: str
    token_estimate: int


class StudentState(BaseModel):
    student_id: str
    mastery: dict[str, float] = Field(default_factory=dict)
    active_misconceptions: list[str] = Field(default_factory=list)
    recent_episodic_ids: list[str] = Field(default_factory=list)


class Trajectory(BaseModel):
    id: str
    student_state: StudentState
    task_type: TaskType
    task_text: str
    budget: int
    candidate_pool: list[PoolItem]
    oracle_selection: list[str]

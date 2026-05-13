from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field

from .artifacts import Artifact, ArtifactType, artifact_to_body


Tier = Literal["semantic", "student", "episodic", "intervention"]


class MemoryItem(BaseModel):
    """Uniform representation used by the selector."""

    id: str
    tier: Tier
    artifact_type: ArtifactType | None = None
    topic_id: str | None = None
    title: str
    body: str
    token_estimate: int
    metadata: dict = Field(default_factory=dict)
    embedding: list[float] = Field(default_factory=list)
    created_at: datetime | None = None

    @classmethod
    def from_artifact(
        cls,
        a: Artifact,
        *,
        embedding: list[float],
        item_id: str | None = None,
    ) -> "MemoryItem":
        body = artifact_to_body(a)
        return cls(
            id=item_id or f"sem:{a.topic_id}:{a.title}",
            tier="semantic",
            artifact_type=a.artifact_type,
            topic_id=a.topic_id,
            title=a.title or a.artifact_type.value,
            body=body,
            token_estimate=max(1, len(body) // 4),
            embedding=embedding,
        )


class MasteryEntry(BaseModel):
    student_id: str
    concept_id: str
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    last_updated: datetime | None = None


class EpisodicEvent(BaseModel):
    id: str | None = None
    student_id: str
    event_type: str
    payload: dict
    occurred_at: datetime | None = None
    embedding: list[float] = Field(default_factory=list)


class InterventionRecord(BaseModel):
    id: str | None = None
    student_id: str
    misconception_id: str | None = None
    strategy: str
    outcome: str | None = None
    notes: str | None = None
    occurred_at: datetime | None = None

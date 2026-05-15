from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field, field_validator


class ArtifactType(str, Enum):
    CONCEPT = "concept"
    EXAMPLE = "example"
    MISCONCEPTION = "misconception"
    EXERCISE = "exercise"
    CODE_PATTERN = "code_pattern"
    PAPER_CLAIM = "paper_claim"


class _ArtifactBase(BaseModel):
    topic_id: str
    title: str = ""
    artifact_type: ArtifactType


class Concept(_ArtifactBase):
    artifact_type: Literal[ArtifactType.CONCEPT] = ArtifactType.CONCEPT
    definition: str
    deep_explanation: str
    prerequisites: list[str] = Field(default_factory=list)


class Example(_ArtifactBase):
    artifact_type: Literal[ArtifactType.EXAMPLE] = ArtifactType.EXAMPLE
    concept_title: str
    body: str


class Misconception(_ArtifactBase):
    artifact_type: Literal[ArtifactType.MISCONCEPTION] = ArtifactType.MISCONCEPTION
    statement: str
    correction: str


class Exercise(_ArtifactBase):
    artifact_type: Literal[ArtifactType.EXERCISE] = ArtifactType.EXERCISE
    prompt: str
    starter_code: str = ""
    rubric: str

    @field_validator("starter_code", mode="before")
    @classmethod
    def coerce_none_to_empty(cls, v: object) -> str:
        return "" if v is None else v


class CodePattern(_ArtifactBase):
    artifact_type: Literal[ArtifactType.CODE_PATTERN] = ArtifactType.CODE_PATTERN
    body: str


class PaperClaim(_ArtifactBase):
    artifact_type: Literal[ArtifactType.PAPER_CLAIM] = ArtifactType.PAPER_CLAIM
    claim: str
    source: str
    evidence: str


Artifact = Concept | Example | Misconception | Exercise | CodePattern | PaperClaim


def artifact_to_body(a: Artifact) -> str:
    """Canonical text representation for embedding and serialization."""
    if isinstance(a, Concept):
        return f"{a.title}\n{a.definition}\n{a.deep_explanation}"
    if isinstance(a, Example):
        return f"Example for {a.concept_title}: {a.body}"
    if isinstance(a, Misconception):
        return f"Misconception: {a.statement}\nCorrection: {a.correction}"
    if isinstance(a, Exercise):
        return f"Exercise: {a.prompt}\nRubric: {a.rubric}"
    if isinstance(a, CodePattern):
        return f"Code pattern: {a.title}\n{a.body}"
    if isinstance(a, PaperClaim):
        return f"Claim: {a.claim}\nSource: {a.source}\nEvidence: {a.evidence}"
    raise ValueError(f"Unknown artifact: {a}")

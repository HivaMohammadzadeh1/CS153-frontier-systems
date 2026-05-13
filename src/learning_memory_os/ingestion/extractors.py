from ..llm import LLM
from ..schemas.artifacts import (
    Concept,
    Example,
    Misconception,
    Exercise,
    CodePattern,
    PaperClaim,
    Artifact,
)


EXTRACTION_SYSTEM = """You extract structured ML-systems-engineering teaching artifacts from source text.
Return STRICT JSON with these top-level keys, each holding an array:
  - concepts: [{title, definition, deep_explanation, prerequisites[]}]
  - misconceptions: [{statement, correction}]
  - examples: [{concept_title, body}]
  - exercises: [{title, prompt, starter_code, rubric}]
  - code_patterns: [{title, body}]
  - paper_claims: [{claim, source, evidence}]
Be conservative: only emit items grounded in the source. No commentary outside JSON."""


class ArtifactExtractor:
    def __init__(self, llm: LLM):
        self.llm = llm

    def extract(self, *, topic_id: str, source_text: str) -> list[Artifact]:
        data = self.llm.complete_json(
            system=EXTRACTION_SYSTEM,
            user=f"TOPIC: {topic_id}\n\nSOURCE:\n{source_text}",
            max_tokens=6000,
        )
        out: list[Artifact] = []
        for c in data.get("concepts", []):
            out.append(Concept(topic_id=topic_id, **c))
        for m in data.get("misconceptions", []):
            out.append(Misconception(topic_id=topic_id, title=m["statement"][:60], **m))
        for e in data.get("examples", []):
            out.append(
                Example(topic_id=topic_id, title=f"Example: {e['concept_title']}", **e)
            )
        for x in data.get("exercises", []):
            out.append(Exercise(topic_id=topic_id, **x))
        for cp in data.get("code_patterns", []):
            out.append(CodePattern(topic_id=topic_id, **cp))
        for pc in data.get("paper_claims", []):
            out.append(
                PaperClaim(topic_id=topic_id, title=pc["claim"][:60], **pc)
            )
        return out

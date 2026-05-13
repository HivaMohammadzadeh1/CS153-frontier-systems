from unittest.mock import MagicMock
from learning_memory_os.ingestion.extractors import ArtifactExtractor


def test_extractor_parses_concept_and_misconception():
    fake_llm = MagicMock()
    fake_llm.complete_json.return_value = {
        "concepts": [
            {
                "title": "KV cache",
                "definition": "Cache of attention K/V from prior tokens.",
                "deep_explanation": "Long form.",
                "prerequisites": [],
            }
        ],
        "misconceptions": [
            {"statement": "KV cache stores token ids.", "correction": "It stores K and V tensors."},
        ],
        "examples": [],
        "exercises": [],
        "code_patterns": [],
        "paper_claims": [],
    }

    ex = ArtifactExtractor(llm=fake_llm)
    arts = ex.extract(topic_id="kv_cache", source_text="...lecture transcript...")
    types = sorted(a.artifact_type.value for a in arts)
    assert types == ["concept", "misconception"]
    assert arts[0].topic_id == "kv_cache"

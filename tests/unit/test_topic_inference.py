"""Tests for conversation-level topic inference."""

import math

from learning_memory_os.agents.topic_inference import (
    TopicCentroid,
    TopicCentroids,
    infer_topic,
)


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _centroids() -> TopicCentroids:
    # Three orthogonal-ish "topic" axes.
    return TopicCentroids([
        TopicCentroid(topic_id="kv_cache", vector=_unit([1.0, 0.0, 0.0])),
        TopicCentroid(topic_id="quantization", vector=_unit([0.0, 1.0, 0.0])),
        TopicCentroid(topic_id="inference_latency", vector=_unit([0.0, 0.0, 1.0])),
    ])


def test_high_similarity_yields_auto_decision():
    # Embed function returns a vector almost identical to kv_cache centroid.
    def fake_embed(text: str) -> list[float]:
        return _unit([0.99, 0.05, 0.05])

    result = infer_topic(
        conversation=[{"role": "user", "content": "what about the kv cache?"}],
        centroids=_centroids(),
        embed_fn=fake_embed,
    )
    assert result.topic_id == "kv_cache"
    assert result.confidence >= 0.6
    assert result.decision == "auto"


def test_mid_similarity_yields_inferred_decision():
    # Tuned: cosine with [1,0,0] is 1.05 / sqrt(1.1025+1+1) ≈ 0.596.
    def fake_embed(text: str) -> list[float]:
        return _unit([1.05, 1.0, 1.0])

    result = infer_topic(
        conversation=[{"role": "user", "content": "ambiguous question"}],
        centroids=_centroids(),
        embed_fn=fake_embed,
    )
    assert result.topic_id == "kv_cache"
    assert 0.4 <= result.confidence < 0.6
    assert result.decision == "inferred"


def test_low_similarity_yields_ask_decision_and_no_topic():
    # Query that points away from every topic axis: max cosine is negative,
    # which clamps to confidence=0 (below the 0.4 inferred threshold).
    # In production (1536-d embeddings) most "off-topic" queries land here
    # naturally because real centroids don't span the whole space.
    def fake_embed(text: str) -> list[float]:
        return _unit([-1.0, -1.0, -1.0])

    result = infer_topic(
        conversation=[{"role": "user", "content": "totally unrelated noise"}],
        centroids=_centroids(),
        embed_fn=fake_embed,
    )
    assert result.topic_id is None
    assert result.confidence < 0.4
    assert result.decision == "ask"


def test_uses_last_n_user_turns_plus_current_for_inference():
    # The current turn says nothing topical ("about this topic?").
    # The earlier user turns mention "kv cache". Inference should still pick kv_cache.
    seen_inputs: list[str] = []

    def fake_embed(text: str) -> list[float]:
        seen_inputs.append(text)
        # Return a vector aligned with kv_cache only if "kv cache" is in the text.
        if "kv cache" in text:
            return _unit([1.0, 0.05, 0.05])
        return _unit([0.3, 0.3, 0.3])  # ambiguous

    conversation = [
        {"role": "user", "content": "Tell me about kv cache eviction"},
        {"role": "assistant", "content": "Sure — KV caches store ..."},
        {"role": "user", "content": "What's the common misconception about this topic?"},
    ]
    result = infer_topic(
        conversation=conversation,
        centroids=_centroids(),
        embed_fn=fake_embed,
        history_turns=4,
    )
    assert "kv cache" in seen_inputs[0].lower()
    assert result.topic_id == "kv_cache"
    assert result.decision in {"auto", "inferred"}


def test_assistant_turns_excluded_from_inference_input():
    seen_inputs: list[str] = []

    def fake_embed(text: str) -> list[float]:
        seen_inputs.append(text)
        return _unit([1.0, 0.0, 0.0])

    conversation = [
        {"role": "user", "content": "USER_ONE"},
        {"role": "assistant", "content": "ASSISTANT_REPLY_SHOULD_NOT_APPEAR"},
        {"role": "user", "content": "USER_TWO"},
    ]
    infer_topic(conversation=conversation, centroids=_centroids(), embed_fn=fake_embed)
    assert "ASSISTANT_REPLY_SHOULD_NOT_APPEAR" not in seen_inputs[0]


def test_topic_centroids_from_seeds_averages_embeddings():
    # Deterministic embed_fn keyed by text content.
    def fake_embed(text: str) -> list[float]:
        if "cache" in text:
            return [1.0, 0.0]
        return [0.0, 1.0]

    centroids = TopicCentroids.from_seeds(
        seeds={
            "kv_cache": ["cache seed A", "cache seed B"],
            "other": ["something else"],
        },
        embed_fn=fake_embed,
    )
    by_id = {c.topic_id: c.vector for c in centroids.all()}
    # Both cache seeds → [1,0], averaged → [1,0], normalized → [1,0].
    assert by_id["kv_cache"] == [1.0, 0.0]
    assert by_id["other"] == [0.0, 1.0]

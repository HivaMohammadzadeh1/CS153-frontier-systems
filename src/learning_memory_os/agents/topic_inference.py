"""Conversation-level topic inference.

Replaces the manual "Topic focus" dropdown's default. Embeds the last N user
turns plus the current one, compares cosine similarity against each topic's
precomputed centroid, and returns a top-1 topic with a confidence-based
decision: auto / inferred / ask.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal

CONFIDENCE_AUTO = 0.6
CONFIDENCE_INFERRED = 0.4


@dataclass(frozen=True)
class TopicCentroid:
    topic_id: str
    vector: list[float]


@dataclass(frozen=True)
class TopicInferenceResult:
    topic_id: str | None
    confidence: float
    decision: Literal["auto", "inferred", "ask"]


class TopicCentroids:
    def __init__(self, centroids: list[TopicCentroid]):
        self._centroids = list(centroids)

    def all(self) -> list[TopicCentroid]:
        return list(self._centroids)

    @classmethod
    def from_seeds(
        cls,
        seeds: dict[str, list[str]],
        embed_fn: Callable[[str], list[float]],
    ) -> "TopicCentroids":
        out: list[TopicCentroid] = []
        for topic_id, texts in seeds.items():
            if not texts:
                continue
            vectors = [embed_fn(t) for t in texts]
            mean = _mean(vectors)
            out.append(TopicCentroid(topic_id=topic_id, vector=_unit(mean)))
        return cls(out)


def infer_topic(
    *,
    conversation: list[dict],
    centroids: TopicCentroids,
    embed_fn: Callable[[str], list[float]],
    history_turns: int = 4,
) -> TopicInferenceResult:
    user_turns = [m["content"] for m in conversation if m.get("role") == "user"]
    if not user_turns:
        return TopicInferenceResult(topic_id=None, confidence=0.0, decision="ask")

    if len(user_turns) > history_turns:
        user_turns = user_turns[-history_turns:]
    query_text = "\n".join(user_turns)

    query_vec = _unit(embed_fn(query_text))
    best_topic: str | None = None
    best_score = -1.0
    for c in centroids.all():
        score = _dot(query_vec, c.vector)
        if score > best_score:
            best_score = score
            best_topic = c.topic_id

    confidence = max(0.0, best_score)
    if confidence >= CONFIDENCE_AUTO:
        return TopicInferenceResult(topic_id=best_topic, confidence=confidence, decision="auto")
    if confidence >= CONFIDENCE_INFERRED:
        return TopicInferenceResult(topic_id=best_topic, confidence=confidence, decision="inferred")
    return TopicInferenceResult(topic_id=None, confidence=confidence, decision="ask")


def _mean(vectors: list[list[float]]) -> list[float]:
    n = len(vectors)
    dim = len(vectors[0])
    out = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            out[i] += v[i]
    return [x / n for x in out]


def _unit(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    if norm == 0.0:
        return list(v)
    return [x / norm for x in v]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

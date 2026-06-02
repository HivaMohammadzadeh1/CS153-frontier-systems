"""Forgetting-curve decay for mastery scores (Track B, read-side only).

Stored mastery is never mutated by decay; callers apply ``effective_score`` when
they *read* mastery so that knowledge fades realistically over time. Well-learned,
high-confidence concepts decay slowly; shaky ones fade fast.
"""

import math
from datetime import datetime, timezone


def half_life_days(score: float, confidence: float) -> float:
    """Retention half-life in days. Ranges ~3 days (weak) to ~33 days (mastered)."""
    return 3.0 + 30.0 * max(0.0, min(1.0, score)) * max(0.0, min(1.0, confidence))


def effective_score(
    score: float,
    confidence: float,
    last_updated: datetime | None,
    *,
    now: datetime | None = None,
) -> float:
    """Apply exponential forgetting to a stored mastery score."""
    if last_updated is None:
        return score
    now = now or datetime.now(timezone.utc)
    if last_updated.tzinfo is None:
        last_updated = last_updated.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (now - last_updated).total_seconds() / 86400.0)
    return score * math.pow(0.5, age_days / half_life_days(score, confidence))

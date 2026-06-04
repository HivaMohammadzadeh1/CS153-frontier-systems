"""Unit tests for the calibrated readiness verdict + rubric weighting.

The verdict is pure given the interview rows, so we drive it with a tiny fake
connection/cursor instead of a live DB.
"""
from learning_memory_os.agents.interview_prompts import (
    weighted_overall, CATEGORIES, CRITICAL_CATEGORIES, CATEGORY_WEIGHTS,
)
from learning_memory_os.api import _readiness_verdict, _tier_for


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, *a, **k):
        pass

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def cursor(self):
        return _FakeCursor(self._rows)


def _row(overall, **cats):
    scores = {c: cats.get(c, overall) for c in CATEGORIES}
    return {"overall_score": overall, "evaluation": {"category_scores": scores}}


def test_weights_sum_to_one():
    for role, w in CATEGORY_WEIGHTS.items():
        assert abs(sum(w.values()) - 1.0) < 1e-9, role


def test_communication_is_a_multiplier_not_a_driver():
    # Polished prose (high communication) can't rescue weak technicals.
    weak_tech = {c: 40 for c in CATEGORIES}
    weak_tech["communication"] = 100
    assert weighted_overall(weak_tech, "ml_infra") < 50


def test_tier_boundaries():
    assert _tier_for(95)[0] == "frontier"
    assert _tier_for(80)[0] == "ready"
    assert _tier_for(70)[0] == "borderline"
    assert _tier_for(60)[0] == "not_ready"
    assert _tier_for(20)[0] == "remediation"


def test_no_data_verdict():
    v = _readiness_verdict(_FakeConn([]), "s1")
    assert v["tier"] == "no_data"
    assert v["interview_ready"] is False


def test_three_strong_interviews_are_ready():
    rows = [_row(86), _row(84), _row(85)]  # most-recent first
    v = _readiness_verdict(_FakeConn(rows), "s1")
    assert v["interview_count"] == 3
    assert v["interview_ready"] is True
    assert v["tier"] in ("ready", "frontier")


def test_critical_failure_blocks_ready():
    # High average but a load-bearing skill is failing in the latest interview.
    latest = _row(88, **{CRITICAL_CATEGORIES[0]: 40})
    rows = [latest, _row(88), _row(88)]
    v = _readiness_verdict(_FakeConn(rows), "s1")
    assert v["interview_ready"] is False
    assert v["tier"] == "not_ready"
    assert any(cf["category"] == CRITICAL_CATEGORIES[0] for cf in v["critical_failures"])


def test_two_interviews_not_yet_confirmed():
    rows = [_row(90), _row(90)]
    v = _readiness_verdict(_FakeConn(rows), "s1")
    assert v["interview_ready"] is False  # need >= 3
    assert "more interview" in v["label"]

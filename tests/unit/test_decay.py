from datetime import datetime, timedelta, timezone

from learning_memory_os.memory.decay import effective_score, half_life_days


NOW = datetime(2026, 5, 30, tzinfo=timezone.utc)


def test_no_decay_when_fresh():
    assert effective_score(0.8, 0.9, NOW, now=NOW) == 0.8


def test_decays_over_time():
    fresh = effective_score(0.8, 0.5, NOW - timedelta(days=1), now=NOW)
    stale = effective_score(0.8, 0.5, NOW - timedelta(days=30), now=NOW)
    assert fresh > stale
    assert 0.0 <= stale < 0.8


def test_higher_confidence_and_score_decays_slower():
    # Same elapsed time; the better-learned concept retains more.
    weak = effective_score(0.5, 0.2, NOW - timedelta(days=14), now=NOW)
    strong = effective_score(0.5, 0.95, NOW - timedelta(days=14), now=NOW)
    assert strong > weak
    assert half_life_days(0.9, 0.9) > half_life_days(0.2, 0.2)


def test_none_last_updated_returns_raw():
    assert effective_score(0.6, 0.5, None) == 0.6

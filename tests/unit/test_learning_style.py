from learning_memory_os.agents.learning_style import _classify


def test_new_learner_has_no_inferred_style():
    s = _classify(n_questions=1, n_topics=1, n_quizzes=0, avg_quiz=None,
                  overall_mastery=0.0, median_gap_sec=None, n_misconceptions=0)
    assert "New learner" in s.summary
    assert s.dimensions == {}


def test_advanced_depth_first_active_recall():
    s = _classify(n_questions=20, n_topics=4, n_quizzes=8, avg_quiz=0.85,
                  overall_mastery=0.8, median_gap_sec=30, n_misconceptions=0)
    assert s.dimensions["level"]["label"] == "advanced"
    assert s.dimensions["depth"]["label"] == "depth-first"        # 20/4 = 5 >= 3
    assert s.dimensions["self_testing"]["label"] == "active recall"  # 8/20 = 0.4 >= 0.3
    assert s.dimensions["pace"]["label"] == "fast"                # 30 < 45


def test_foundations_breadth_passive():
    s = _classify(n_questions=12, n_topics=12, n_quizzes=0, avg_quiz=None,
                  overall_mastery=0.2, median_gap_sec=300, n_misconceptions=3)
    assert s.dimensions["level"]["label"] == "building foundations"
    assert s.dimensions["depth"]["label"] == "breadth-first"      # 12/12 = 1 <= 1.4
    assert s.dimensions["self_testing"]["label"] == "passive"     # 0 quizzes
    assert s.dimensions["pace"]["label"] == "deliberate"          # 300 > 180
    assert "misconceptions" in s.summary or s.summary  # non-empty

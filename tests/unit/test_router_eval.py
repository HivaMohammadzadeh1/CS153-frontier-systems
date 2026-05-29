from learning_memory_os.eval.router_eval import evaluate


def test_perfect_match():
    m = evaluate([["a", "b"]], [["a", "b"]])
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.jaccard == 1.0
    assert m.n == 1


def test_no_overlap():
    m = evaluate([["a"]], [["b"]])
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.jaccard == 0.0


def test_partial_overlap():
    m = evaluate([["a", "b", "c"]], [["a", "b", "d"]])
    # tp=2, fp=1, fn=1 -> precision=2/3, recall=2/3, jaccard=2/4=0.5
    assert abs(m.precision - 2 / 3) < 1e-6
    assert abs(m.recall - 2 / 3) < 1e-6
    assert m.jaccard == 0.5


def test_empty_predictions_list():
    m = evaluate([], [])
    assert m.n == 0
    assert m.precision == 0.0


def test_both_empty_selection_is_perfect_jaccard():
    # A trajectory where oracle selected nothing and the router also selected
    # nothing should not be punished: jaccard of two empty sets is 1.0.
    m = evaluate([[]], [[]])
    assert m.jaccard == 1.0

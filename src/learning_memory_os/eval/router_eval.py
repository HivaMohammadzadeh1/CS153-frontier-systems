"""Set-overlap metrics for router selections vs. the oracle ground truth."""

from dataclasses import dataclass


@dataclass
class RouterMetrics:
    precision: float
    recall: float
    jaccard: float
    n: int


def _binary_metrics(pred: list[str], gold: list[str]) -> tuple[float, float, float]:
    sp, sg = set(pred), set(gold)
    tp = len(sp & sg)
    fp = len(sp - sg)
    fn = len(sg - sp)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    union = sp | sg
    # Two empty selections agree perfectly.
    jaccard = tp / len(union) if union else 1.0
    return precision, recall, jaccard


def evaluate(predictions: list[list[str]], gold: list[list[str]]) -> RouterMetrics:
    assert len(predictions) == len(gold)
    if not predictions:
        return RouterMetrics(precision=0.0, recall=0.0, jaccard=0.0, n=0)
    ps, rs, js = [], [], []
    for p, g in zip(predictions, gold):
        pp, rr, jj = _binary_metrics(p, g)
        ps.append(pp)
        rs.append(rr)
        js.append(jj)
    return RouterMetrics(
        precision=sum(ps) / len(ps),
        recall=sum(rs) / len(rs),
        jaccard=sum(js) / len(js),
        n=len(predictions),
    )

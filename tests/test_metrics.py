import numpy as np

from qir_route.baseline import minmax_rowwise
from qir_route.metrics import evaluate_single_positive


def test_metrics_use_one_based_ranks_and_cutoffs() -> None:
    scores = np.asarray([[0.9, 0.1, 0.2], [0.2, 0.3, 0.1]], dtype=np.float32)
    metrics, ranks = evaluate_single_positive(scores, [0, 0], [1, 3])
    assert ranks == [1, 2]
    assert metrics["Recall@1"] == 0.5
    assert metrics["MRR@3"] == 0.75
    assert np.isclose(metrics["nDCG@3"], (1.0 + 1.0 / np.log2(3.0)) / 2.0)


def test_minmax_constant_rows_are_zero() -> None:
    scores = np.asarray([[2.0, 2.0], [1.0, 3.0]], dtype=np.float32)
    normalized = minmax_rowwise(scores)
    assert np.array_equal(normalized[0], np.zeros(2))
    assert np.array_equal(normalized[1], np.asarray([0.0, 1.0]))

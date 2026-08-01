from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def stable_order(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-scores, kind="mergesort")


def evaluate_single_positive(
    scores: np.ndarray,
    gold_indices: Iterable[int],
    ks: Iterable[int],
) -> tuple[dict[str, float], list[int]]:
    score_matrix = np.asarray(scores)
    gold = [int(index) for index in gold_indices]
    if score_matrix.ndim != 2:
        raise ValueError("scores must have shape [queries, documents]")
    if score_matrix.shape[0] != len(gold):
        raise ValueError("one gold index is required for every query")

    cutoffs = sorted({int(k) for k in ks if int(k) > 0})
    ranks: list[int] = []
    for query_index, gold_index in enumerate(gold):
        if not 0 <= gold_index < score_matrix.shape[1]:
            raise ValueError(f"gold index {gold_index} is outside the corpus")
        order = stable_order(score_matrix[query_index])
        rank = int(np.flatnonzero(order == gold_index)[0]) + 1
        ranks.append(rank)

    rank_array = np.asarray(ranks, dtype=np.int64)
    metrics: dict[str, float] = {}
    for k in cutoffs:
        found = rank_array <= min(k, score_matrix.shape[1])
        reciprocal = np.where(found, 1.0 / rank_array, 0.0)
        discounted = np.where(found, 1.0 / np.log2(rank_array + 1), 0.0)
        metrics[f"HitRate@{k}"] = float(found.mean())
        metrics[f"Recall@{k}"] = float(found.mean())
        metrics[f"MRR@{k}"] = float(reciprocal.mean())
        metrics[f"nDCG@{k}"] = float(discounted.mean())
    return metrics, ranks

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

from qir_route.stage_a.ablation import paired_bootstrap_confidence_interval

STRATEGY_ORDER = [
    "frozen_fused_top50",
    "dense_top50",
    "dense_top100",
    "fused50_union_dense50",
    "fused50_union_dense100",
]


def full_dense_ranking(
    query_embeddings: np.ndarray, document_embeddings: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if query_embeddings.ndim != 2 or document_embeddings.ndim != 2:
        raise ValueError("stored embeddings must be two-dimensional")
    if query_embeddings.shape != document_embeddings.shape:
        raise ValueError("validation query/document embeddings must be aligned")
    scores = query_embeddings @ document_embeddings.T
    order = np.argsort(-scores, axis=1, kind="stable")
    return scores, order.astype(np.int32)


def stable_union(
    first: Sequence[int], second: Sequence[int], maximum: int
) -> list[int]:
    if maximum <= 0:
        raise ValueError("maximum candidate count must be positive")
    result: list[int] = []
    seen: set[int] = set()
    for value in (*first, *second):
        item = int(value)
        if item not in seen:
            seen.add(item)
            result.append(item)
            if len(result) == maximum:
                break
    return result


def positive_ranks(rankings: Sequence[Sequence[int]]) -> np.ndarray:
    ranks = np.full(len(rankings), np.nan, dtype=np.float64)
    for query_index, ranking in enumerate(rankings):
        for position, document_index in enumerate(ranking, start=1):
            if int(document_index) == query_index:
                ranks[query_index] = position
                break
    return ranks


def _metric_vectors(ranks: np.ndarray, cutoff: int) -> np.ndarray:
    return (np.isfinite(ranks) & (ranks <= cutoff)).astype(np.float64)


def _ranking_metrics(ranks: np.ndarray, candidate_counts: np.ndarray) -> dict[str, Any]:
    recall = {
        f"recall_at_{cutoff}": float(_metric_vectors(ranks, cutoff).mean())
        for cutoff in (10, 20, 50, 100)
    }
    reciprocal = np.where(
        np.isfinite(ranks) & (ranks <= 10),
        1.0 / np.nan_to_num(ranks, nan=np.inf),
        0.0,
    )
    ndcg = np.where(
        np.isfinite(ranks) & (ranks <= 10),
        1.0 / np.log2(np.nan_to_num(ranks, nan=np.inf) + 1.0),
        0.0,
    )
    counts = Counter(int(value) for value in candidate_counts)
    return {
        **recall,
        "mrr_at_10": float(reciprocal.mean()),
        "ndcg_at_10": float(ndcg.mean()),
        "candidate_recall": float(np.isfinite(ranks).mean()),
        "mean_candidate_count": float(candidate_counts.mean()),
        "candidate_count_distribution": {
            str(key): value for key, value in sorted(counts.items())
        },
    }


def evaluate_strategies(
    frozen_indices: np.ndarray,
    dense_order: np.ndarray,
    *,
    bootstrap: dict[str, Any],
) -> tuple[dict[str, list[list[int]]], dict[str, dict[str, Any]]]:
    if frozen_indices.ndim != 2 or dense_order.ndim != 2:
        raise ValueError("candidate rankings must be two-dimensional")
    if frozen_indices.shape[0] != dense_order.shape[0]:
        raise ValueError("candidate rankings must have one row per query")
    frozen = [[int(value) for value in row] for row in frozen_indices]
    dense50 = [[int(value) for value in row[:50]] for row in dense_order]
    dense100 = [[int(value) for value in row[:100]] for row in dense_order]
    rankings = {
        "frozen_fused_top50": frozen,
        "dense_top50": dense50,
        "dense_top100": dense100,
        "fused50_union_dense50": [
            stable_union(first, second, 100)
            for first, second in zip(frozen, dense50, strict=True)
        ],
        "fused50_union_dense100": [
            stable_union(first, second, 150)
            for first, second in zip(frozen, dense100, strict=True)
        ],
    }
    baseline_ranks = positive_ranks(rankings["frozen_fused_top50"])
    baseline_present = np.isfinite(baseline_ranks)
    baseline_values = baseline_present.astype(np.float64)
    metrics: dict[str, dict[str, Any]] = {}
    frozen_sets = [set(row) for row in frozen]
    for name in STRATEGY_ORDER:
        strategy_rankings = rankings[name]
        ranks = positive_ranks(strategy_rankings)
        present = np.isfinite(ranks)
        counts = np.asarray([len(row) for row in strategy_rankings], dtype=np.int32)
        overlaps = np.asarray(
            [
                len(frozen_sets[index].intersection(row))
                for index, row in enumerate(strategy_rankings)
            ],
            dtype=np.float64,
        )
        missing_count = int((~baseline_present).sum())
        present_count = int(baseline_present.sum())
        recovered = (~baseline_present) & present
        lost = baseline_present & ~present
        values = present.astype(np.float64)
        summary = _ranking_metrics(ranks, counts)
        summary.update(
            {
                "overlap_with_frozen_top50_mean_count": float(overlaps.mean()),
                "overlap_with_frozen_top50_mean_fraction": float(
                    (overlaps / 50.0).mean()
                ),
                "previously_impossible_recovered_count": int(recovered.sum()),
                "previously_impossible_recovered_percentage": (
                    float(recovered.sum() / missing_count) if missing_count else 0.0
                ),
                "already_retrievable_lost_count": int(lost.sum()),
                "already_retrievable_lost_percentage": (
                    float(lost.sum() / present_count) if present_count else 0.0
                ),
                "paired_candidate_recall_delta_vs_frozen": float(
                    (values - baseline_values).mean()
                ),
                "paired_bootstrap_ci": paired_bootstrap_confidence_interval(
                    values,
                    baseline_values,
                    replicates=int(bootstrap["replicates"]),
                    confidence=float(bootstrap["confidence"]),
                    seed=int(bootstrap["seed"]),
                ),
                "conditional_recovered_query_metrics": _ranking_metrics(
                    ranks[recovered], counts[recovered]
                )
                if recovered.any()
                else None,
            }
        )
        metrics[name] = summary
    return rankings, metrics


def _entropy(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return -(probabilities * np.log(probabilities + 1e-12)).sum(axis=1)


def _quartiles(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    integer_ranks = np.empty(len(values), dtype=np.int64)
    integer_ranks[order] = np.arange(len(values))
    quartile_indices = np.minimum(integer_ranks * 4 // len(values), 3)
    return np.asarray(["q1", "q2", "q3", "q4"])[quartile_indices]


def build_per_query_recovery(
    qids: np.ndarray,
    context_hashes: np.ndarray,
    dense_scores: np.ndarray,
    rankings: dict[str, list[list[int]]],
) -> pd.DataFrame:
    fused_ranks = positive_ranks(rankings["frozen_fused_top50"])
    dense_ranks = positive_ranks(
        [
            list(map(int, row))
            for row in np.argsort(-dense_scores, axis=1, kind="stable")
        ]
    )
    dense50_ranks = positive_ranks(rankings["dense_top50"])
    dense100_ranks = positive_ranks(rankings["dense_top100"])
    union50_ranks = positive_ranks(rankings["fused50_union_dense50"])
    union100_ranks = positive_ranks(rankings["fused50_union_dense100"])
    natural = np.isfinite(fused_ranks)
    overlap = np.asarray(
        [
            len(set(fused).intersection(dense))
            for fused, dense in zip(
                rankings["frozen_fused_top50"], rankings["dense_top50"], strict=True
            )
        ],
        dtype=np.int32,
    )
    jaccard = overlap / (100.0 - overlap)
    sorted_scores = np.take_along_axis(
        dense_scores,
        np.argsort(-dense_scores, axis=1, kind="stable")[:, :5],
        axis=1,
    )
    recovered50 = ~natural & np.isfinite(dense50_ranks)
    recovered100 = ~natural & np.isfinite(dense100_ranks)
    categories = np.full(len(qids), "unrecovered", dtype=object)
    categories[natural] = "already_retrievable"
    categories[recovered100] = "recovered_dense100_only"
    categories[recovered50] = "recovered_dense50"
    frame = pd.DataFrame(
        {
            "query_id": qids.astype(str),
            "source_document_group": context_hashes.astype(str),
            "natural_positive_in_fused_top50": natural,
            "positive_rank_fused": fused_ranks,
            "positive_rank_dense": dense_ranks,
            "positive_in_dense_top50": np.isfinite(dense50_ranks),
            "positive_in_dense_top100": np.isfinite(dense100_ranks),
            "recovered_by_dense50": recovered50,
            "recovered_by_dense100": recovered100,
            "recovered_by_union50": ~natural & np.isfinite(union50_ranks),
            "recovered_by_union100": ~natural & np.isfinite(union100_ranks),
            "fused_dense_overlap_count": overlap,
            "fused_dense_jaccard_at_50": jaccard,
            "dense_top1_top2_margin": sorted_scores[:, 0] - sorted_scores[:, 1],
            "dense_top1_top5_margin": sorted_scores[:, 0] - sorted_scores[:, 4],
            "dense_entropy": _entropy(dense_scores),
            "retrieval_recovery_category": categories,
            "diagnostic_only": True,
            "can_promote_frozen_qi_method": False,
        }
    )
    frame["dense_entropy_quartile"] = _quartiles(frame["dense_entropy"].to_numpy())
    frame["dense_margin_quartile"] = _quartiles(
        frame["dense_top1_top2_margin"].to_numpy()
    )
    frame["fused_dense_overlap_quartile"] = _quartiles(
        frame["fused_dense_overlap_count"].to_numpy()
    )
    rank_buckets = np.full(len(frame), "missing", dtype=object)
    rank_buckets[(fused_ranks >= 1) & (fused_ranks <= 1)] = "rank_1"
    rank_buckets[(fused_ranks >= 2) & (fused_ranks <= 10)] = "rank_2_10"
    rank_buckets[(fused_ranks >= 11) & (fused_ranks <= 20)] = "rank_11_20"
    rank_buckets[(fused_ranks >= 21) & (fused_ranks <= 50)] = "rank_21_50"
    frame["frozen_positive_rank_bucket"] = rank_buckets
    return frame


def build_slice_analysis(
    frame: pd.DataFrame,
    rankings: dict[str, list[list[int]]],
    *,
    best_strategy: str,
    bootstrap: dict[str, Any],
    minimum_support: int,
) -> list[dict[str, Any]]:
    baseline_ranks = positive_ranks(rankings["frozen_fused_top50"])
    strategy_ranks = positive_ranks(rankings[best_strategy])
    work = frame.copy()
    work["_baseline_50"] = _metric_vectors(baseline_ranks, 50)
    work["_strategy_50"] = _metric_vectors(strategy_ranks, 50)
    work["_baseline_100"] = np.isfinite(baseline_ranks).astype(float)
    work["_strategy_100"] = _metric_vectors(strategy_ranks, 100)
    work["_strategy_present"] = np.isfinite(strategy_ranks)
    families = {
        "frozen_positive_presence": "natural_positive_in_fused_top50",
        "dense_entropy_quartile": "dense_entropy_quartile",
        "dense_margin_quartile": "dense_margin_quartile",
        "frozen_positive_rank_bucket": "frozen_positive_rank_bucket",
        "fused_dense_overlap_quartile": "fused_dense_overlap_quartile",
    }
    rows: list[dict[str, Any]] = []
    for family, column in families.items():
        for value, grouped in work.groupby(column, observed=True, sort=True):
            subset = cast(pd.DataFrame, grouped)
            rows.append(
                _summarize_slice(
                    subset,
                    family,
                    str(value),
                    bootstrap,
                    minimum_support,
                )
            )
    source_groups = work.groupby("source_document_group", observed=True, sort=True)
    for value, grouped in source_groups:
        subset = cast(pd.DataFrame, grouped)
        if len(subset) < minimum_support:
            continue
        rows.append(
            _summarize_slice(
                subset,
                "source_document_group",
                str(value),
                bootstrap,
                minimum_support,
            )
        )
    return rows


def _summarize_slice(
    subset: pd.DataFrame,
    family: str,
    value: str,
    bootstrap: dict[str, Any],
    minimum_support: int,
) -> dict[str, Any]:
    missing = ~subset["natural_positive_in_fused_top50"]
    recovered = missing & subset["_strategy_present"]
    missing_count = int(missing.sum())
    recovered_count = int(recovered.sum())
    candidate_ci = paired_bootstrap_confidence_interval(
        subset["_strategy_present"].to_numpy(dtype=float),
        subset["natural_positive_in_fused_top50"].to_numpy(dtype=float),
        replicates=int(bootstrap["replicates"]),
        confidence=float(bootstrap["confidence"]),
        seed=int(bootstrap["seed"]),
    )
    count = int(subset["query_id"].nunique())
    return {
        "slice_family": family,
        "slice_value": value,
        "unique_query_count": count,
        "missing_query_count": missing_count,
        "recovered_query_count": recovered_count,
        "recovery_percentage": float(recovered_count / missing_count)
        if missing_count
        else 0.0,
        "recall_at_50_delta": float(
            (subset["_strategy_50"] - subset["_baseline_50"]).mean()
        ),
        "recall_at_100_delta": float(
            (subset["_strategy_100"] - subset["_baseline_100"]).mean()
        ),
        "paired_bootstrap_ci": {
            "recall_at_50": paired_bootstrap_confidence_interval(
                subset["_strategy_50"].to_numpy(dtype=float),
                subset["_baseline_50"].to_numpy(dtype=float),
                replicates=int(bootstrap["replicates"]),
                confidence=float(bootstrap["confidence"]),
                seed=int(bootstrap["seed"]),
            ),
            "recall_at_100": paired_bootstrap_confidence_interval(
                subset["_strategy_100"].to_numpy(dtype=float),
                subset["_baseline_100"].to_numpy(dtype=float),
                replicates=int(bootstrap["replicates"]),
                confidence=float(bootstrap["confidence"]),
                seed=int(bootstrap["seed"]),
            ),
            "candidate_recall": candidate_ci,
        },
        "support_warning": count < minimum_support,
        "can_define_recoverable_slice": count >= minimum_support,
        "diagnostic_only": True,
        "can_promote_frozen_qi_method": False,
    }


def verdict_from_gate(
    *,
    absolute_recall_improvement: float,
    bootstrap_lower: float,
    recovered_missing_percentage: float,
    already_retrievable_lost_count: int,
    integrity_passed: bool,
    query_count: int,
    minimum_support: int,
) -> tuple[str, dict[str, bool]]:
    checks = {
        "absolute_recall_improvement_at_least_0_05": absolute_recall_improvement
        >= 0.05,
        "paired_bootstrap_lower_above_zero": bootstrap_lower > 0.0,
        "missing_queries_recovered_at_least_20_percent": recovered_missing_percentage
        >= 0.20,
        "already_retrievable_queries_lost_is_zero": already_retrievable_lost_count == 0,
        "hashes_and_firewall_pass": integrity_passed,
        "minimum_unique_query_support_met": query_count >= minimum_support,
    }
    if not integrity_passed:
        return "blocked", checks
    if query_count < minimum_support:
        return "insufficient_evidence", checks
    if all(checks.values()):
        return "candidate_ceiling_recoverable", checks
    return "candidate_ceiling_not_recoverable", checks

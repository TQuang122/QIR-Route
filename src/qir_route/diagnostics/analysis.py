from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from qir_route.stage_a.ablation import paired_bootstrap_confidence_interval


def score_entropy(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return -(probabilities * np.log(probabilities.clip(min=1e-12))).sum(axis=1)


def score_margins(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(scores, axis=1)[:, ::-1]
    return ordered[:, 0] - ordered[:, 1], ordered[:, 0] - ordered[:, 4]


def rank_inversion_statistics(
    base_scores: np.ndarray,
    final_scores: np.ndarray,
    positive_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    base_order = np.argsort(-base_scores, axis=1, kind="stable")
    final_order = np.argsort(-final_scores, axis=1, kind="stable")
    base_positions = np.argsort(base_order, axis=1)
    final_positions = np.argsort(final_order, axis=1)
    base_differences = base_positions[:, :, None] - base_positions[:, None, :]
    final_differences = final_positions[:, :, None] - final_positions[:, None, :]
    upper = np.triu(
        np.ones((base_scores.shape[1], base_scores.shape[1]), dtype=bool), k=1
    )
    inversion_count = (((base_differences * final_differences) < 0) & upper[None]).sum(
        axis=(1, 2)
    )
    total_pairs = base_scores.shape[1] * (base_scores.shape[1] - 1) / 2
    has_positive = positive_mask.any(axis=1)
    positive_index = positive_mask.argmax(axis=1)
    row_indices = np.arange(base_scores.shape[0])
    base_rank = base_positions[row_indices, positive_index]
    final_rank = final_positions[row_indices, positive_index]
    improvement = np.where(has_positive, np.maximum(base_rank - final_rank, 0), 0)
    harm = np.where(has_positive, np.maximum(final_rank - base_rank, 0), 0)
    safe_inversions = np.maximum(inversion_count, 1)
    return {
        "inversion_rate": inversion_count / total_pairs,
        "relevance_improving_inversion_fraction": improvement / safe_inversions,
        "relevance_harming_inversion_fraction": harm / safe_inversions,
    }


def add_quartile_column(
    frame: pd.DataFrame, source_column: str, output_column: str
) -> None:
    per_query = frame.groupby("query_id", sort=False)[source_column].mean()
    valid = per_query.dropna()
    if valid.nunique() < 2:
        frame[output_column] = None
        return
    labels = pd.qcut(np.asarray(valid), q=min(4, valid.nunique()), duplicates="drop")
    mapping = {
        str(query_id): str(label)
        for query_id, label in zip(valid.index.tolist(), labels, strict=True)
    }
    frame[output_column] = [mapping.get(str(value)) for value in frame["query_id"]]


def first_rank_bucket(rank: float) -> str:
    if math.isnan(rank):
        return "missing_top50"
    if rank == 1:
        return "rank_1"
    if rank <= 5:
        return "rank_2_5"
    if rank <= 10:
        return "rank_6_10"
    if rank <= 20:
        return "rank_11_20"
    return "rank_21_50"


def summarize_slice(
    frame: pd.DataFrame,
    *,
    slice_family: str,
    slice_value: str,
    minimum_support: int,
    required_consistent_seeds: int,
    bootstrap_config: dict[str, Any],
) -> dict[str, Any]:
    by_query = frame.groupby("query_id", sort=False)[
        ["delta_qi_vs_base", "delta_qi_vs_classical"]
    ].mean()
    support = len(by_query)
    qi_delta = np.asarray(by_query["delta_qi_vs_base"], dtype=np.float64)
    classical_delta = np.asarray(by_query["delta_qi_vs_classical"], dtype=np.float64)
    interval = paired_bootstrap_confidence_interval(
        qi_delta,
        np.zeros_like(qi_delta),
        replicates=int(bootstrap_config["replicates"]),
        confidence=float(bootstrap_config["confidence"]),
        seed=int(bootstrap_config["seed"]),
    )
    seed_means = np.asarray(
        frame.groupby("seed", sort=True)["delta_qi_vs_base"].mean(),
        dtype=np.float64,
    )
    consistent_seed_count = int(np.count_nonzero(seed_means > 0))
    stable = bool(
        support >= minimum_support
        and float(qi_delta.mean()) > 0
        and float(interval["lower"]) > 0
        and consistent_seed_count >= required_consistent_seeds
    )
    return {
        "slice_family": slice_family,
        "slice_value": slice_value,
        "query_count": support,
        "query_seed_row_count": len(frame),
        "mean_delta_qi_vs_base": float(qi_delta.mean()),
        "median_delta_qi_vs_base": float(np.median(qi_delta)),
        "mean_delta_qi_vs_classical": float(classical_delta.mean()),
        "median_delta_qi_vs_classical": float(np.median(classical_delta)),
        "helped_percentage": float((qi_delta > 0).mean() * 100),
        "harmed_percentage": float((qi_delta < 0).mean() * 100),
        "paired_bootstrap_ci": interval,
        "positive_direction_seed_count": consistent_seed_count,
        "required_positive_direction_seed_count": required_consistent_seeds,
        "support_warning": support < minimum_support,
        "stable_qi_regime": stable,
        "diagnostic_only": True,
        "can_promote_frozen_method": False,
    }


def build_slice_analyses(
    frame: pd.DataFrame,
    *,
    minimum_support: int,
    required_consistent_seeds: int,
    bootstrap_config: dict[str, Any],
) -> list[dict[str, Any]]:
    analyses: list[dict[str, Any]] = []
    families = {
        "natural_positive_in_top50": "natural_positive_in_top50",
        "base_entropy_quartile": "base_entropy_quartile",
        "base_margin_quartile": "base_margin_quartile",
        "qi_correction_magnitude_quartile": "qi_correction_magnitude_quartile",
        "first_relevant_rank_bucket": "first_relevant_rank_bucket",
        "seed": "seed",
        "source_document_group": "source_document_group",
    }
    for family, column in families.items():
        for value, subset in frame.dropna(subset=[column]).groupby(column, sort=True):
            if (
                family == "source_document_group"
                and subset["query_id"].nunique() < minimum_support
            ):
                continue
            analyses.append(
                summarize_slice(
                    subset,
                    slice_family=family,
                    slice_value=str(value),
                    minimum_support=minimum_support,
                    required_consistent_seeds=required_consistent_seeds,
                    bootstrap_config=bootstrap_config,
                )
            )
    return analyses


def choose_verdict(
    slices: list[dict[str, Any]], unavailable_required_features: list[str]
) -> tuple[str, dict[str, Any] | None]:
    stable = [item for item in slices if item["stable_qi_regime"]]
    strongest = (
        max(stable, key=lambda item: item["mean_delta_qi_vs_base"]) if stable else None
    )
    if strongest is not None:
        return "stable_qi_regime_found", strongest
    if unavailable_required_features:
        return "insufficient_evidence", None
    return "no_stable_qi_regime", None

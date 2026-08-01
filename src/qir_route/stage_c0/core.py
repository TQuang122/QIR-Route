from __future__ import annotations

import hashlib
import time
import unicodedata
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from qir_route.baseline import tokenize


LANE_SELECTION_ORDER = ("rrf", "dense", "bm25")
AUDIT_LABELS = {"supported", "not_supported"}
AUDIT_REASON_CODES = {
    "non_vietnamese_query",
    "empty_or_corrupt",
    "topic_mismatch",
    "insufficient_answer",
    "ambiguous",
    "other",
}
RAW_TEXT_KEYS = {"query", "query_vi", "query_ede", "title", "passage", "document"}


class DataAcceptanceError(RuntimeError):
    pass


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    normalized = "".join(
        character
        for character in normalized
        if unicodedata.category(character) not in {"Cc", "Cf"}
    )
    return " ".join(normalized.split())


def construct_document_text(title: object, passage: object) -> str:
    clean_title = str(title).strip() if not pd.isna(title) else ""
    clean_passage = str(passage).strip() if not pd.isna(passage) else ""
    return f"{clean_title}\n{clean_passage}" if clean_title else clean_passage


def identifier_digest(namespace: str, query_id: str) -> bytes:
    return hashlib.sha256(f"{namespace}\0{query_id}".encode()).digest()


def partition_query_ids(
    query_ids: Sequence[str], *, decision_id: str, calibration_count: int
) -> dict[str, list[str]]:
    values = [str(value) for value in query_ids]
    if len(values) != len(set(values)):
        raise DataAcceptanceError("train query identifiers are not unique")
    if not 0 < calibration_count < len(values):
        raise ValueError("calibration count must leave two non-empty partitions")
    ordered = sorted(
        values,
        key=lambda query_id: (identifier_digest(decision_id, query_id), query_id),
    )
    calibration = ordered[:calibration_count]
    fit = ordered[calibration_count:]
    if set(calibration).intersection(fit):
        raise RuntimeError("fit and calibration query identifiers overlap")
    return {"fit": fit, "calibration": calibration}


def audit_structural_data(
    corpus: pd.DataFrame,
    queries: pd.DataFrame,
    qrels: pd.DataFrame,
    *,
    decision_id: str,
    calibration_count: int,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    if set(corpus.columns) != {"corpus_id", "title", "passage"}:
        raise DataAcceptanceError("corpus schema does not match the frozen contract")
    if set(queries.columns) != {"query_id", "query_vi"}:
        raise DataAcceptanceError("query schema does not match the frozen contract")
    if set(qrels.columns) != {"query_id", "corpus_id", "score"}:
        raise DataAcceptanceError("qrel schema does not match the frozen contract")

    corpus_ids = corpus["corpus_id"].astype(str)
    query_ids = queries["query_id"].astype(str)
    qrel_query_ids = qrels["query_id"].astype(str)
    qrel_corpus_ids = qrels["corpus_id"].astype(str)
    if bool(corpus["corpus_id"].isna().any()):
        raise DataAcceptanceError("corpus identifiers must not be null")
    if bool(queries[["query_id", "query_vi"]].isna().any().any()):
        raise DataAcceptanceError("query identifiers and query_vi must not be null")
    if bool(qrels[["query_id", "corpus_id"]].isna().any().any()):
        raise DataAcceptanceError("qrel identifiers must not be null")
    if (corpus_ids.str.strip() == "").any() or not corpus_ids.is_unique:
        raise DataAcceptanceError("corpus identifiers must be non-empty and unique")
    if (query_ids.str.strip() == "").any() or not query_ids.is_unique:
        raise DataAcceptanceError("query identifiers must be non-empty and unique")
    if set(query_ids) != set(qrel_query_ids):
        raise DataAcceptanceError("train query and qrel query identifiers differ")
    qrel_counts = qrel_query_ids.value_counts()
    if not bool((qrel_counts == 1).all()) or len(qrel_counts) != len(query_ids):
        raise DataAcceptanceError("every train query must have exactly one qrel")
    if not bool((pd.to_numeric(qrels["score"], errors="coerce") > 0).all()):
        raise DataAcceptanceError("every qrel score must be positive")
    missing_corpus = set(qrel_corpus_ids).difference(corpus_ids)
    if missing_corpus:
        raise DataAcceptanceError("qrels reference missing corpus identifiers")

    normalized_queries = queries["query_vi"].map(normalize_text)
    documents = [
        construct_document_text(title, passage)
        for title, passage in zip(corpus["title"], corpus["passage"], strict=True)
    ]
    if bool((normalized_queries == "").any()):
        raise DataAcceptanceError("query_vi contains empty normalized text")
    if any(normalize_text(document) == "" for document in documents):
        raise DataAcceptanceError("corpus contains empty constructed documents")

    partitions = partition_query_ids(
        query_ids.tolist(),
        decision_id=decision_id,
        calibration_count=calibration_count,
    )
    query_by_id = dict(zip(query_ids, normalized_queries, strict=True))
    fit_text = {query_by_id[query_id] for query_id in partitions["fit"]}
    calibration_text = {query_by_id[query_id] for query_id in partitions["calibration"]}
    cross_partition_duplicates = fit_text.intersection(calibration_text)
    if cross_partition_duplicates:
        raise DataAcceptanceError(
            "normalized duplicate queries cross fit and calibration partitions"
        )

    return partitions, {
        "status": "verified",
        "corpus_count": int(len(corpus)),
        "train_query_count": int(len(queries)),
        "train_qrel_count": int(len(qrels)),
        "fit_count": len(partitions["fit"]),
        "calibration_count": len(partitions["calibration"]),
        "query_qrel_sets_equal": True,
        "one_positive_qrel_per_query": True,
        "qrel_corpus_ids_exist": True,
        "normalized_cross_partition_duplicate_count": 0,
        "raw_text_exported": False,
    }


def select_human_audit_query_ids(
    queries: pd.DataFrame,
    calibration_ids: Sequence[str],
    *,
    decision_id: str,
    samples_per_quartile: int,
) -> list[str]:
    if samples_per_quartile <= 0:
        raise ValueError("samples per quartile must be positive")
    calibration_set = set(map(str, calibration_ids))
    frame = queries.loc[
        queries["query_id"].astype(str).isin(calibration_set), ["query_id", "query_vi"]
    ].copy()
    if len(frame) != len(calibration_set):
        raise DataAcceptanceError("calibration query IDs are not aligned with queries")
    frame["query_id"] = frame["query_id"].astype(str)
    frame["length"] = frame["query_vi"].map(lambda value: len(tokenize(str(value))))
    frame["partition_digest"] = frame["query_id"].map(
        lambda value: identifier_digest(decision_id, value)
    )
    frame = frame.sort_values(
        ["length", "partition_digest", "query_id"], kind="stable"
    ).reset_index(drop=True)
    count = len(frame)
    frame["quartile"] = np.minimum(np.arange(count) * 4 // count, 3)
    selected: list[str] = []
    audit_namespace = f"{decision_id}-AUDIT"
    for quartile in range(4):
        group = frame.loc[frame["quartile"] == quartile].copy()
        if len(group) < samples_per_quartile:
            raise DataAcceptanceError("a query-length quartile is too small for audit")
        group["audit_digest"] = group["query_id"].map(
            lambda value: identifier_digest(audit_namespace, value)
        )
        group = group.sort_values(["audit_digest", "query_id"], kind="stable")
        selected.extend(group["query_id"].head(samples_per_quartile).tolist())
    return selected


def evaluate_human_audit(
    records: Sequence[dict[str, Any]],
    *,
    expected_query_ids: Sequence[str],
    gold_corpus_by_query: dict[str, str],
    minimum_supported: int,
    maximum_non_vietnamese: int,
) -> dict[str, Any]:
    expected = list(map(str, expected_query_ids))
    if len(records) != len(expected):
        raise DataAcceptanceError("human audit record count is not complete")
    observed_ids: list[str] = []
    supported = 0
    non_vietnamese = 0
    empty_or_corrupt = 0
    reviewers: set[str] = set()
    for record in records:
        if RAW_TEXT_KEYS.intersection(record):
            raise DataAcceptanceError("human audit artifact contains raw text")
        required = {
            "query_id",
            "corpus_id",
            "label",
            "reason_codes",
            "review_timestamp",
            "rubric_version",
            "reviewer_pseudonym",
        }
        if set(record) != required:
            raise DataAcceptanceError("human audit record schema is invalid")
        query_id = str(record["query_id"])
        corpus_id = str(record["corpus_id"])
        label = str(record["label"])
        reasons = record["reason_codes"]
        if label not in AUDIT_LABELS:
            raise DataAcceptanceError("human audit label is invalid")
        if not isinstance(reasons, list) or not set(map(str, reasons)).issubset(
            AUDIT_REASON_CODES
        ):
            raise DataAcceptanceError("human audit reason code is invalid")
        if corpus_id != gold_corpus_by_query.get(query_id):
            raise DataAcceptanceError("human audit query/corpus pair is not frozen")
        if not str(record["review_timestamp"]).strip():
            raise DataAcceptanceError("human audit timestamp is empty")
        if not str(record["rubric_version"]).strip():
            raise DataAcceptanceError("human audit rubric version is empty")
        if not str(record["reviewer_pseudonym"]).strip():
            raise DataAcceptanceError("human audit reviewer pseudonym is empty")
        reviewer = str(record["reviewer_pseudonym"]).strip()
        if "ambiguous" in reasons and label != "not_supported":
            raise DataAcceptanceError(
                "ambiguous human audit records must be labeled not_supported"
            )
        reviewers.add(reviewer)
        observed_ids.append(query_id)
        supported += int(label == "supported")
        non_vietnamese += int("non_vietnamese_query" in reasons)
        empty_or_corrupt += int("empty_or_corrupt" in reasons)
    if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != set(
        expected
    ):
        raise DataAcceptanceError("human audit query IDs are not the frozen sample")
    if len(reviewers) != 1:
        raise DataAcceptanceError("human audit must use exactly one reviewer")
    checks = {
        "minimum_supported_met": supported >= minimum_supported,
        "maximum_non_vietnamese_met": non_vietnamese <= maximum_non_vietnamese,
        "zero_empty_or_corrupt_met": empty_or_corrupt == 0,
    }
    return {
        "status": "verified",
        "record_count": len(records),
        "supported_count": supported,
        "non_vietnamese_query_count": non_vietnamese,
        "empty_or_corrupt_count": empty_or_corrupt,
        "checks": checks,
        "passed": all(checks.values()),
        "raw_text_exported": False,
    }


def _stable_topk(
    scores: np.ndarray, document_ids: np.ndarray, top_k: int
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64)
    ids = np.asarray(document_ids).astype(str)
    if values.ndim != 1 or ids.shape != values.shape:
        raise ValueError("scores and document IDs must be aligned vectors")
    if not np.isfinite(values).all():
        raise ValueError("candidate scores contain non-finite values")
    count = min(int(top_k), len(values))
    if count <= 0:
        raise ValueError("top-k must be positive")
    if len(values) == count:
        candidates = np.arange(len(values), dtype=np.int32)
    else:
        threshold = np.partition(values, len(values) - count)[len(values) - count]
        above = np.flatnonzero(values > threshold).astype(np.int32)
        tied = np.flatnonzero(values == threshold).astype(np.int32)
        remaining = count - len(above)
        tied = tied[np.argsort(ids[tied], kind="stable")[:remaining]]
        candidates = np.concatenate((above, tied))
    order = candidates[np.lexsort((ids[candidates], -values[candidates]))]
    return order.astype(np.int32), values[order].astype(np.float32)


def bm25_rankings(
    document_texts: Sequence[str],
    query_texts: Sequence[str],
    document_ids: Sequence[str],
    *,
    top_k: int,
    k1: float,
    b: float,
    timing: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if len(document_texts) != len(document_ids):
        raise ValueError("document texts and IDs are not aligned")
    indexing_started = time.perf_counter()
    index = BM25Okapi([tokenize(text) for text in document_texts], k1=k1, b=b)
    indexing_seconds = time.perf_counter() - indexing_started
    rows: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    ids = np.asarray(document_ids, dtype=np.str_)
    query_started = time.perf_counter()
    for query in query_texts:
        order, values = _stable_topk(index.get_scores(tokenize(query)), ids, top_k)
        rows.append(order)
        scores.append(values)
    if timing is not None:
        timing.update(
            {
                "indexing_seconds": indexing_seconds,
                "query_seconds": time.perf_counter() - query_started,
            }
        )
    return np.stack(rows), np.stack(scores)


def exact_dense_rankings(
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    document_ids: Sequence[str],
    *,
    top_k: int,
    query_block_size: int,
    document_block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    queries = np.asarray(query_embeddings, dtype=np.float32)
    documents = np.asarray(document_embeddings, dtype=np.float32)
    ids = np.asarray(document_ids, dtype=np.str_)
    if queries.ndim != 2 or documents.ndim != 2:
        raise ValueError("dense embeddings must be two-dimensional")
    if queries.shape[1] != documents.shape[1] or documents.shape[0] != len(ids):
        raise ValueError("dense embeddings and document IDs are not aligned")
    if min(query_block_size, document_block_size, top_k) <= 0:
        raise ValueError("dense block sizes and top-k must be positive")
    if not np.isfinite(queries).all() or not np.isfinite(documents).all():
        raise ValueError("dense embeddings contain non-finite values")

    result_count = min(top_k, len(documents))
    all_indices = np.empty((len(queries), result_count), dtype=np.int32)
    all_scores = np.empty((len(queries), result_count), dtype=np.float32)
    for query_start in range(0, len(queries), query_block_size):
        query_stop = min(query_start + query_block_size, len(queries))
        query_block = queries[query_start:query_stop]
        best_indices: list[np.ndarray] = [
            np.empty(0, dtype=np.int32) for _ in range(len(query_block))
        ]
        best_scores: list[np.ndarray] = [
            np.empty(0, dtype=np.float32) for _ in range(len(query_block))
        ]
        for document_start in range(0, len(documents), document_block_size):
            document_stop = min(document_start + document_block_size, len(documents))
            block_scores = query_block @ documents[document_start:document_stop].T
            block_indices = np.arange(document_start, document_stop, dtype=np.int32)
            for row_index in range(len(query_block)):
                candidate_indices = np.concatenate(
                    [best_indices[row_index], block_indices]
                )
                candidate_scores = np.concatenate(
                    [best_scores[row_index], block_scores[row_index]]
                )
                local_order, local_scores = _stable_topk(
                    candidate_scores, ids[candidate_indices], result_count
                )
                best_indices[row_index] = candidate_indices[local_order]
                best_scores[row_index] = local_scores
        for row_index, output_index in enumerate(range(query_start, query_stop)):
            all_indices[output_index] = best_indices[row_index]
            all_scores[output_index] = best_scores[row_index]
    return all_indices, all_scores


def reciprocal_rank_fusion(
    bm25_indices: np.ndarray,
    dense_indices: np.ndarray,
    document_ids: Sequence[str],
    *,
    rrf_k: int,
    output_top_k: int,
) -> list[list[int]]:
    if bm25_indices.ndim != 2 or dense_indices.ndim != 2:
        raise ValueError("RRF rankings must be two-dimensional")
    if bm25_indices.shape[0] != dense_indices.shape[0]:
        raise ValueError("RRF rankings must have one row per query")
    if min(rrf_k, output_top_k) <= 0:
        raise ValueError("RRF k and output top-k must be positive")
    ids = np.asarray(document_ids, dtype=np.str_)
    rankings: list[list[int]] = []
    for bm25_row, dense_row in zip(bm25_indices, dense_indices, strict=True):
        scores: dict[int, float] = {}
        for row in (bm25_row, dense_row):
            for rank, raw_index in enumerate(row, start=1):
                index = int(raw_index)
                scores[index] = scores.get(index, 0.0) + 1.0 / (rrf_k + rank)
        order = sorted(scores, key=lambda index: (-scores[index], str(ids[index])))
        rankings.append(order[: min(output_top_k, len(order))])
    return rankings


def evaluate_rankings(
    rankings: Sequence[Sequence[int]],
    *,
    document_ids: Sequence[str],
    gold_document_ids: Sequence[str],
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    if len(rankings) != len(gold_document_ids):
        raise ValueError("rankings require one gold document per query")
    ids = np.asarray(document_ids, dtype=np.str_)
    ranks = np.full(len(rankings), np.inf, dtype=np.float64)
    for query_index, (ranking, gold_id) in enumerate(
        zip(rankings, gold_document_ids, strict=True)
    ):
        for rank, document_index in enumerate(ranking, start=1):
            if str(ids[int(document_index)]) == str(gold_id):
                ranks[query_index] = rank
                break
    vectors = {
        f"recall_at_{cutoff}": (ranks <= cutoff).astype(np.float64)
        for cutoff in (10, 20, 50, 100)
    }
    reciprocal = np.where(ranks <= 10, 1.0 / ranks, 0.0)
    ndcg = np.where(ranks <= 10, 1.0 / np.log2(ranks + 1.0), 0.0)
    vectors["mrr_at_10"] = reciprocal
    vectors["ndcg_at_10"] = ndcg
    metrics = {name: float(values.mean()) for name, values in vectors.items()}
    return metrics, vectors


def choose_candidate_lane(metrics: dict[str, dict[str, float]]) -> str:
    missing = set(LANE_SELECTION_ORDER).difference(metrics)
    if missing:
        raise ValueError(f"candidate metrics are missing lanes: {sorted(missing)}")
    return max(
        LANE_SELECTION_ORDER,
        key=lambda lane: (
            float(metrics[lane]["recall_at_100"]),
            -LANE_SELECTION_ORDER.index(lane),
        ),
    )


def percentile_bootstrap_interval(
    values: np.ndarray, *, replicates: int, confidence: float, seed: int
) -> dict[str, float | int]:
    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1 or sample.size == 0:
        raise ValueError("bootstrap values must be a non-empty vector")
    if replicates <= 0 or not 0 < confidence < 1:
        raise ValueError("bootstrap configuration is invalid")
    generator = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    chunk_size = min(256, replicates)
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        indices = generator.integers(0, sample.size, size=(stop - start, sample.size))
        means[start:stop] = sample[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha])
    return {
        "point_estimate": float(sample.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "confidence": confidence,
        "replicates": replicates,
        "seed": seed,
        "query_count": int(sample.size),
    }


def paired_lane_intervals(
    vectors: dict[str, dict[str, np.ndarray]],
    *,
    selected_lane: str,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict[str, dict[str, dict[str, float | int]]]:
    results: dict[str, dict[str, dict[str, float | int]]] = {}
    ordered_lanes = (selected_lane,) + tuple(
        lane for lane in LANE_SELECTION_ORDER if lane != selected_lane
    )
    pair_index = 0
    for left_index, left_lane in enumerate(ordered_lanes):
        for right_lane in ordered_lanes[left_index + 1 :]:
            pair_name = f"{left_lane}_vs_{right_lane}"
            results[pair_name] = {}
            pair_index += 1
            for metric_index, metric in enumerate(
                (
                    "recall_at_10",
                    "recall_at_20",
                    "recall_at_50",
                    "recall_at_100",
                    "mrr_at_10",
                    "ndcg_at_10",
                )
            ):
                results[pair_name][metric] = _paired_bootstrap_interval(
                    vectors[left_lane][metric],
                    vectors[right_lane][metric],
                    replicates=replicates,
                    confidence=confidence,
                    seed=seed + pair_index * 100 + metric_index,
                )
    return results


def _paired_bootstrap_interval(
    left_values: np.ndarray,
    right_values: np.ndarray,
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict[str, float | int]:
    left = np.asarray(left_values, dtype=np.float64)
    right = np.asarray(right_values, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or left.size == 0:
        raise ValueError("paired bootstrap vectors must be aligned and non-empty")
    if replicates <= 0 or not 0 < confidence < 1:
        raise ValueError("paired bootstrap configuration is invalid")
    differences = left - right
    generator = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    chunk_size = min(256, replicates)
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        indices = generator.integers(
            0, differences.size, size=(stop - start, differences.size)
        )
        means[start:stop] = differences[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha])
    return {
        "point_estimate": float(differences.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "confidence": confidence,
        "replicates": replicates,
        "seed": seed,
        "query_count": int(differences.size),
    }


def stage_c0_verdict(
    *,
    structural_passed: bool,
    human_audit_passed: bool,
    firewall_intact: bool,
    complete: bool,
    recall_at_100: float,
    bootstrap_lower: float,
    minimum_recall: float,
    minimum_bootstrap_lower: float,
) -> tuple[str, dict[str, bool]]:
    checks = {
        "structural_data_gates_passed": structural_passed,
        "human_audit_gates_passed": human_audit_passed,
        "firewall_intact": firewall_intact,
        "run_complete": complete,
        "recall_at_100_at_least_threshold": recall_at_100 >= minimum_recall,
        "bootstrap_lower_at_least_threshold": bootstrap_lower
        >= minimum_bootstrap_lower,
    }
    if not firewall_intact:
        return "firewall_violation", checks
    if not complete:
        return "incomplete_technical_run", checks
    if not structural_passed or not human_audit_passed:
        return "benchmark_rejected", checks
    if (
        not checks["recall_at_100_at_least_threshold"]
        or not checks["bootstrap_lower_at_least_threshold"]
    ):
        return "candidate_ceiling_inadequate", checks
    return "stage_c1_authorized", checks

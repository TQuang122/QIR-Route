from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from qir_route.baseline import minmax_rowwise, tokenize
from qir_route.stage_a.splits import SplitRow


def select_top_candidates(
    bm25_scores: np.ndarray,
    dense_scores: np.ndarray,
    *,
    top_k: int,
    dense_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    if bm25_scores.shape != dense_scores.shape or bm25_scores.ndim != 2:
        raise ValueError("BM25 and dense scores must have the same 2D shape")
    if not 0 <= dense_weight <= 1:
        raise ValueError("dense_weight must be between zero and one")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    fused = dense_weight * minmax_rowwise(dense_scores) + (
        1 - dense_weight
    ) * minmax_rowwise(bm25_scores)
    candidate_count = min(top_k, fused.shape[1])
    order = np.argsort(-fused, axis=1, kind="stable")[:, :candidate_count]
    candidate_scores = np.take_along_axis(fused, order, axis=1)
    return order.astype(np.int32), candidate_scores.astype(np.float32)


def inject_missing_positives(
    candidate_indices: np.ndarray,
    candidate_scores: np.ndarray,
    full_scores: np.ndarray,
) -> int:
    if candidate_indices.shape != candidate_scores.shape:
        raise ValueError("candidate indices and scores must have the same shape")
    if full_scores.shape[0] != candidate_indices.shape[0]:
        raise ValueError("full scores must have one row per query")
    gold_indices = np.arange(candidate_indices.shape[0], dtype=np.int32)
    missing = ~(candidate_indices == gold_indices[:, None]).any(axis=1)
    missing_rows = np.flatnonzero(missing)
    candidate_indices[missing_rows, -1] = gold_indices[missing_rows]
    candidate_scores[missing_rows, -1] = full_scores[missing_rows, missing_rows]
    return int(missing_rows.size)


def build_candidate_cache(
    rows: list[SplitRow],
    *,
    split_name: str,
    encoder: Any,
    model_config: dict[str, Any],
    candidate_config: dict[str, Any],
    output_path: Path,
    force_positive: bool = False,
) -> dict[str, Any]:
    questions = [row.question for row in rows]
    contexts = [row.context for row in rows]
    index = BM25Okapi(
        [tokenize(context) for context in contexts],
        k1=float(candidate_config["bm25_k1"]),
        b=float(candidate_config["bm25_b"]),
    )
    bm25 = np.stack(
        [index.get_scores(tokenize(question)) for question in questions]
    ).astype(np.float32)
    document_embeddings = np.asarray(
        encoder.encode(
            contexts,
            batch_size=int(model_config["batch_size"]),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ),
        dtype=np.float32,
    )
    query_embeddings = np.asarray(
        encoder.encode(
            questions,
            batch_size=int(model_config["batch_size"]),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ),
        dtype=np.float32,
    )
    expected_dimension = int(model_config["expected_dimension"])
    if query_embeddings.shape[1] != expected_dimension:
        raise RuntimeError(
            f"expected {expected_dimension} embedding dimensions, got {query_embeddings.shape[1]}"
        )
    dense = query_embeddings @ document_embeddings.T
    fused = float(candidate_config["dense_weight"]) * minmax_rowwise(dense) + (
        1 - float(candidate_config["dense_weight"])
    ) * minmax_rowwise(bm25)
    candidate_indices, candidate_scores = select_top_candidates(
        bm25,
        dense,
        top_k=int(candidate_config["top_k"]),
        dense_weight=float(candidate_config["dense_weight"]),
    )
    gold_indices = np.arange(len(rows), dtype=np.int32)
    positive_rate_before_injection = float(
        (candidate_indices == gold_indices[:, None]).any(axis=1).mean()
    )
    injected_positive_count = 0
    if force_positive:
        injected_positive_count = inject_missing_positives(
            candidate_indices, candidate_scores, fused
        )
    positive_mask = candidate_indices == gold_indices[:, None]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        qids=np.asarray([row.qid for row in rows], dtype=np.str_),
        context_sha256=np.asarray([row.context_sha256 for row in rows], dtype=np.str_),
        query_embeddings=query_embeddings,
        document_embeddings=document_embeddings,
        candidate_indices=candidate_indices,
        candidate_scores=candidate_scores,
        positive_mask=positive_mask,
    )
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "status": "verified",
        "split": split_name,
        "query_count": len(rows),
        "document_count": len(rows),
        "candidate_count": int(candidate_indices.shape[1]),
        "retrieval_positive_candidate_rate": positive_rate_before_injection,
        "positive_candidate_rate": float(positive_mask.any(axis=1).mean()),
        "positive_injection_enabled": force_positive,
        "positive_injection_count": injected_positive_count,
        "embedding_dimension": expected_dimension,
        "model_id": model_config["id"],
        "model_revision": model_config["revision"],
        "raw_text_exported": False,
        "cache_sha256": digest,
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest

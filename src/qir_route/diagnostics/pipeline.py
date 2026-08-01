from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from qir_route.baseline import resolve_device
from qir_route.diagnostics.analysis import (
    add_quartile_column,
    build_slice_analyses,
    choose_verdict,
    first_rank_bucket,
    rank_inversion_statistics,
    score_entropy,
    score_margins,
)
from qir_route.diagnostics.firewall import (
    assert_diagnostic_path_allowed,
    read_allowed_json,
    verify_test_firewall,
)
from qir_route.diagnostics.provenance import (
    sha256_file,
    verify_frozen_receipts,
    write_provenance_snapshot,
)
from qir_route.quantum import QuantumInspiredHead
from qir_route.stage_a.models import MatchedClassicalHead, standardize_candidate_scores


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_per_query_metrics(
    stage_a2_run: Path, lane: str, seed: int
) -> dict[str, np.ndarray]:
    path = (
        stage_a2_run / "training" / lane / f"seed_{seed}" / "best_per_query_metrics.npz"
    )
    assert_diagnostic_path_allowed(path)
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name].copy() for name in arrays.files}


def _load_training_receipt(stage_a2_run: Path, lane: str, seed: int) -> dict[str, Any]:
    return read_allowed_json(
        stage_a2_run / "training" / lane / f"seed_{seed}" / "training_receipt.json"
    )


def _score_checkpoint(
    stage_a2_run: Path,
    lane: str,
    seed: int,
    *,
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    candidate_indices: np.ndarray,
    base_scores: np.ndarray,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    checkpoint_path = (
        stage_a2_run / "training" / lane / f"seed_{seed}" / "best_checkpoint.pt"
    )
    assert_diagnostic_path_allowed(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    embedding_dimension = int(checkpoint["embedding_dimension"])
    if lane == "residual_qi":
        head: QuantumInspiredHead | MatchedClassicalHead = QuantumInspiredHead(
            embedding_dim=embedding_dimension
        ).to(device)
    elif lane == "residual_classical":
        head = MatchedClassicalHead(embedding_dim=embedding_dimension).to(device)
    else:
        raise ValueError(f"unsupported diagnostic lane: {lane}")
    head.load_state_dict(checkpoint["head_state_dict"])
    head.eval()
    residual_weight = float(checkpoint["residual_weight"])
    raw_batches: list[torch.Tensor] = []
    final_batches: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(query_embeddings), batch_size):
            stop = min(start + batch_size, len(query_embeddings))
            queries = torch.from_numpy(query_embeddings[start:stop]).to(device)
            indices = candidate_indices[start:stop]
            documents = torch.from_numpy(document_embeddings[indices]).to(device)
            if isinstance(head, QuantumInspiredHead):
                raw = head.score(queries, documents, mode="mean")
            else:
                raw = head.score(queries, documents)
            base = torch.from_numpy(base_scores[start:stop]).to(device)
            final = base + residual_weight * standardize_candidate_scores(raw)
            raw_batches.append(raw.cpu())
            final_batches.append(final.cpu())
    return (
        torch.cat(raw_batches).numpy(),
        torch.cat(final_batches).numpy(),
        residual_weight,
    )


def _first_relevant_rank(scores: np.ndarray, positive_mask: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1, kind="stable")
    ranked_positive = np.take_along_axis(positive_mask, order, axis=1)
    has_positive = ranked_positive.any(axis=1)
    ranks = ranked_positive.argmax(axis=1).astype(np.float64) + 1
    ranks[~has_positive] = np.nan
    return ranks


def _aggregate_result_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for scope, subset in {
        "all_validation_queries": frame,
        "natural_positive_in_top50_only": frame[frame["natural_positive_in_top50"]],
    }.items():
        metrics[scope] = {
            "query_count": int(subset["query_id"].nunique()),
            "baseline_ndcg_at_10": float(subset["base_ndcg_at_10"].mean()),
            "residual_qi_ndcg_at_10": float(subset["residual_qi_ndcg_at_10"].mean()),
            "residual_classical_ndcg_at_10": float(
                subset["residual_classical_ndcg_at_10"].mean()
            ),
            "baseline_mrr_at_10": float(subset["base_mrr_at_10"].mean()),
            "residual_qi_mrr_at_10": float(subset["residual_qi_mrr_at_10"].mean()),
            "residual_classical_mrr_at_10": float(
                subset["residual_classical_mrr_at_10"].mean()
            ),
        }
    return metrics


def _optimization_diagnostics(
    stage_a2_run: Path,
    seeds: list[int],
    score_distributions: dict[str, list[dict[str, float]]],
    frame: pd.DataFrame,
) -> dict[str, Any]:
    lanes: dict[str, Any] = {}
    for lane in ["residual_qi", "residual_classical"]:
        receipts = [_load_training_receipt(stage_a2_run, lane, seed) for seed in seeds]
        lanes[lane] = {
            "max_gradient_norm_by_seed": {
                str(item["seed"]): item["max_gradient_norm"] for item in receipts
            },
            "clipping_rate_by_seed": {
                str(item["seed"]): item["clipped_step_rate"] for item in receipts
            },
            "clipping_by_epoch": {
                str(item["seed"]): [
                    {
                        "epoch": epoch["epoch"],
                        "clipped_step_rate": epoch["clipped_step_rate"],
                    }
                    for epoch in item["history"]
                ]
                for item in receipts
            },
            "lambda_trajectory": {
                str(item["seed"]): [
                    {
                        "epoch": epoch["epoch"],
                        "residual_weight": epoch["residual_weight"],
                    }
                    for epoch in item["history"]
                ]
                for item in receipts
            },
            "score_distributions": score_distributions[lane],
            "step_gradient_distribution_available": False,
            "step_gradient_distribution_unavailable_reason": (
                "frozen receipts store maxima and clipping counts, not per-step norms"
            ),
        }
    lanes["rank_inversions"] = {
        "qi_mean_inversion_rate": float(frame["qi_rank_inversion_rate"].mean()),
        "qi_relevance_improving_fraction": float(
            frame["qi_relevance_improving_inversion_fraction"].mean()
        ),
        "qi_relevance_harming_fraction": float(
            frame["qi_relevance_harming_inversion_fraction"].mean()
        ),
        "classical_mean_inversion_rate": float(
            frame["classical_rank_inversion_rate"].mean()
        ),
        "classical_relevance_improving_fraction": float(
            frame["classical_relevance_improving_inversion_fraction"].mean()
        ),
        "classical_relevance_harming_fraction": float(
            frame["classical_relevance_harming_inversion_fraction"].mean()
        ),
    }
    return lanes


def _render_report(metrics: dict[str, Any]) -> str:
    verdict = metrics["verdict"]
    stable = metrics["stable_qi_regime_exists"]
    strongest = metrics["strongest_valid_slice"]
    ceiling = metrics["candidate_ceiling"]
    unavailable = metrics["feature_availability"]["unavailable_required_features"]
    return "\n".join(
        [
            "# Post-Stage-A.2 diagnostic report",
            "",
            "This report is diagnostic only and cannot promote the frozen QI method.",
            "The sealed test split was not read, cached, scored, or inspected.",
            "",
            f"- Verdict: `{verdict}`",
            f"- Stable QI-helpful regime exists: `{str(stable).lower()}`",
            f"- Strongest valid slice: `{strongest if strongest else 'none'}`",
            "- New preregistered Stage A.3 justified: `false`",
            "- Test remained untouched: `true`",
            "",
            "## Candidate ceiling",
            "",
            f"- Natural Recall@10: {ceiling['natural_recall_at_10']:.6f}",
            f"- Natural Recall@20: {ceiling['natural_recall_at_20']:.6f}",
            f"- Natural Recall@50: {ceiling['natural_recall_at_50']:.6f}",
            (
                "- Validation queries impossible to improve by reranking: "
                f"{ceiling['impossible_to_improve_by_reranking_fraction']:.6f}"
            ),
            "",
            "Conditional reranker metrics exclude queries whose positive document is",
            "absent from Top-50. End-to-end metrics retain every validation query.",
            "",
            "## Evidence availability",
            "",
            "Required unavailable features: " + ", ".join(unavailable),
            "These features were not reconstructed from raw dataset text because the",
            "diagnostic firewall forbids reading rows assigned to the test split.",
            "",
            "## Decision",
            "",
            "No Stage A.3 is implemented or authorized by this diagnostic run.",
            "",
        ]
    )


def run_post_a2_diagnostics(config_path: Path) -> Path:
    config_path = config_path.resolve()
    repository_root = config_path.parent.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("diagnostic_only") is not True:
        raise ValueError("post-A.2 configuration must be diagnostic_only")
    stage_a2_run = (repository_root / config["stage_a2_run"]).resolve()
    firewall_receipt = verify_test_firewall(stage_a2_run)
    receipt_hashes_before = verify_frozen_receipts(
        repository_root, dict(config["frozen_receipts"])
    )
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    stage_receipt_hash = sha256_file(stage_a2_run / "stage_a2_receipt.json")
    run_id = f"diagnostic-{config_hash[:8]}-{stage_receipt_hash[:8]}"
    output_dir = (repository_root / config["output_root"] / run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    provenance_receipt = write_provenance_snapshot(
        repository_root, config, output_dir / "provenance_receipt.json"
    )

    validation_cache_path = stage_a2_run / "validation_candidates.npz"
    assert_diagnostic_path_allowed(validation_cache_path)
    with np.load(validation_cache_path, allow_pickle=False) as arrays:
        qids = arrays["qids"].astype(str)
        context_hashes = arrays["context_sha256"].astype(str)
        query_embeddings = arrays["query_embeddings"].copy()
        document_embeddings = arrays["document_embeddings"].copy()
        candidate_indices = arrays["candidate_indices"].copy()
        base_scores = arrays["candidate_scores"].copy()
        positive_mask = arrays["positive_mask"].copy()
    stage_receipt = read_allowed_json(stage_a2_run / "stage_a2_receipt.json")
    seeds = [int(item["seed"]) for item in stage_receipt["lanes"]["residual_qi"]]
    device = resolve_device(str(config["device"]))
    batch_size = int(config["batch_size"])
    dense_candidate_scores = np.einsum(
        "bd,bkd->bk",
        query_embeddings,
        document_embeddings[candidate_indices],
        optimize=True,
    )
    fused_entropy = score_entropy(base_scores)
    dense_entropy = score_entropy(dense_candidate_scores)
    margin_12, margin_15 = score_margins(base_scores)
    first_rank = _first_relevant_rank(base_scores, positive_mask)
    natural_positive = positive_mask.any(axis=1)
    rows: list[pd.DataFrame] = []
    score_distributions: dict[str, list[dict[str, float]]] = {
        "residual_qi": [],
        "residual_classical": [],
    }
    for seed in seeds:
        qi_only_metrics = _load_per_query_metrics(stage_a2_run, "qi_only", seed)
        qi_metrics = _load_per_query_metrics(stage_a2_run, "residual_qi", seed)
        classical_metrics = _load_per_query_metrics(
            stage_a2_run, "residual_classical", seed
        )
        qi_raw, qi_final, qi_lambda = _score_checkpoint(
            stage_a2_run,
            "residual_qi",
            seed,
            query_embeddings=query_embeddings,
            document_embeddings=document_embeddings,
            candidate_indices=candidate_indices,
            base_scores=base_scores,
            device=device,
            batch_size=batch_size,
        )
        classical_raw, classical_final, classical_lambda = _score_checkpoint(
            stage_a2_run,
            "residual_classical",
            seed,
            query_embeddings=query_embeddings,
            document_embeddings=document_embeddings,
            candidate_indices=candidate_indices,
            base_scores=base_scores,
            device=device,
            batch_size=batch_size,
        )
        qi_inversions = rank_inversion_statistics(base_scores, qi_final, positive_mask)
        classical_inversions = rank_inversion_statistics(
            base_scores, classical_final, positive_mask
        )
        qi_receipt = _load_training_receipt(stage_a2_run, "residual_qi", seed)
        classical_receipt = _load_training_receipt(
            stage_a2_run, "residual_classical", seed
        )
        for lane, raw, final in [
            ("residual_qi", qi_raw, qi_final),
            ("residual_classical", classical_raw, classical_final),
        ]:
            score_distributions[lane].append(
                {
                    "seed": seed,
                    "correction_mean": float(raw.mean()),
                    "correction_std": float(raw.std()),
                    "base_mean": float(base_scores.mean()),
                    "base_std": float(base_scores.std()),
                    "residual_mean": float(final.mean()),
                    "residual_std": float(final.std()),
                }
            )
        base_ndcg = qi_metrics["baseline_ndcg_at_10"]
        base_mrr = qi_metrics["baseline_mrr_at_10"]
        seed_frame = pd.DataFrame(
            {
                "query_id": qids,
                "source_document_group": context_hashes,
                "seed": seed,
                "base_ndcg_at_10": base_ndcg,
                "qi_only_ndcg_at_10": qi_only_metrics["model_ndcg_at_10"],
                "residual_qi_ndcg_at_10": qi_metrics["model_ndcg_at_10"],
                "residual_classical_ndcg_at_10": classical_metrics["model_ndcg_at_10"],
                "delta_qi_vs_base": qi_metrics["model_ndcg_at_10"] - base_ndcg,
                "delta_classical_vs_base": (
                    classical_metrics["model_ndcg_at_10"] - base_ndcg
                ),
                "delta_qi_vs_classical": (
                    qi_metrics["model_ndcg_at_10"]
                    - classical_metrics["model_ndcg_at_10"]
                ),
                "base_mrr_at_10": base_mrr,
                "residual_qi_mrr_at_10": qi_metrics["model_mrr_at_10"],
                "residual_classical_mrr_at_10": classical_metrics["model_mrr_at_10"],
                "natural_positive_in_top50": natural_positive,
                "number_of_relevant_documents": np.ones(len(qids), dtype=np.int32),
                "first_relevant_rank_base": first_rank,
                "bm25_dense_jaccard_at_10": np.nan,
                "bm25_dense_jaccard_at_50": np.nan,
                "rbo": np.nan,
                "bm25_entropy": np.nan,
                "dense_entropy": dense_entropy,
                "fused_entropy": fused_entropy,
                "top1_top2_margin": margin_12,
                "top1_top5_margin": margin_15,
                "qi_correction_mean": qi_raw.mean(axis=1),
                "qi_correction_std": qi_raw.std(axis=1),
                "qi_correction_max_absolute_value": np.abs(qi_raw).max(axis=1),
                "classical_correction_mean": classical_raw.mean(axis=1),
                "classical_correction_std": classical_raw.std(axis=1),
                "classical_correction_max_absolute_value": np.abs(classical_raw).max(
                    axis=1
                ),
                "qi_learned_lambda": qi_lambda,
                "classical_learned_lambda": classical_lambda,
                "qi_clipped_step_rate": qi_receipt["clipped_step_rate"],
                "classical_clipped_step_rate": classical_receipt["clipped_step_rate"],
                "qi_max_gradient_norm": qi_receipt["max_gradient_norm"],
                "classical_max_gradient_norm": classical_receipt["max_gradient_norm"],
                "query_length": np.nan,
                "qi_rank_inversion_rate": qi_inversions["inversion_rate"],
                "qi_relevance_improving_inversion_fraction": qi_inversions[
                    "relevance_improving_inversion_fraction"
                ],
                "qi_relevance_harming_inversion_fraction": qi_inversions[
                    "relevance_harming_inversion_fraction"
                ],
                "classical_rank_inversion_rate": classical_inversions["inversion_rate"],
                "classical_relevance_improving_inversion_fraction": (
                    classical_inversions["relevance_improving_inversion_fraction"]
                ),
                "classical_relevance_harming_inversion_fraction": (
                    classical_inversions["relevance_harming_inversion_fraction"]
                ),
                "diagnostic_only": True,
                "can_promote_frozen_method": False,
            }
        )
        rows.append(seed_frame)
    frame = pd.concat(rows, ignore_index=True)
    add_quartile_column(frame, "fused_entropy", "base_entropy_quartile")
    add_quartile_column(frame, "top1_top2_margin", "base_margin_quartile")
    add_quartile_column(
        frame,
        "qi_correction_max_absolute_value",
        "qi_correction_magnitude_quartile",
    )
    frame["first_relevant_rank_bucket"] = frame["first_relevant_rank_base"].map(
        first_rank_bucket
    )
    frame["bm25_dense_disagreement_quartile"] = None
    frame["query_length_bucket"] = None

    per_query_parquet = output_dir / "per_query_diagnostics.parquet"
    per_query_jsonl = output_dir / "per_query_diagnostics.jsonl"
    frame.to_parquet(per_query_parquet, index=False)
    with per_query_jsonl.open("w", encoding="utf-8") as handle:
        for record in frame.to_dict(orient="records"):
            handle.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")

    unavailable_features = [
        "exact_bm25_dense_jaccard_at_10",
        "exact_bm25_dense_jaccard_at_50",
        "exact_rbo",
        "bm25_entropy",
        "bm25_dense_disagreement_quartile",
        "query_length_bucket",
    ]
    slices = build_slice_analyses(
        frame,
        minimum_support=int(config["minimum_stable_support"]),
        required_consistent_seeds=int(config["required_consistent_seeds"]),
        bootstrap_config=dict(config["bootstrap"]),
    )
    verdict, strongest_slice = choose_verdict(slices, unavailable_features)
    natural_rank = first_rank
    candidate_ceiling = {
        "natural_recall_at_10": float(np.nan_to_num(natural_rank <= 10).mean()),
        "natural_recall_at_20": float(np.nan_to_num(natural_rank <= 20).mean()),
        "natural_recall_at_50": float(natural_positive.mean()),
        "oracle_positive_first_ndcg_at_10_within_top50": float(natural_positive.mean()),
        "impossible_to_improve_by_reranking_fraction": float(
            1 - natural_positive.mean()
        ),
        "result_metrics": _aggregate_result_metrics(frame),
        "conditional_vs_end_to_end_distinction": True,
    }
    diagnostic_metrics = {
        "schema_version": 1,
        "status": "verified",
        "diagnostic_only": True,
        "can_promote_frozen_method": False,
        "verdict": verdict,
        "stable_qi_regime_exists": strongest_slice is not None,
        "strongest_valid_slice": strongest_slice,
        "stage_a3_scientifically_justified": False,
        "test_remained_untouched": True,
        "feature_availability": {
            "unavailable_required_features": unavailable_features,
            "reason": (
                "frozen exports omit raw BM25 rankings and query text; reading the full "
                "CSV would access sealed test rows"
            ),
        },
        "candidate_ceiling": candidate_ceiling,
        "slice_analyses": slices,
        "optimization_diagnostics": _optimization_diagnostics(
            stage_a2_run, seeds, score_distributions, frame
        ),
    }
    metrics_path = output_dir / "diagnostic_metrics.json"
    report_path = output_dir / "diagnostic_report.md"
    _write_json(metrics_path, diagnostic_metrics)
    report_path.write_text(_render_report(diagnostic_metrics), encoding="utf-8")

    receipt_hashes_after = verify_frozen_receipts(
        repository_root, dict(config["frozen_receipts"])
    )
    if receipt_hashes_before != receipt_hashes_after:
        raise RuntimeError("historical receipt hashes changed during diagnostics")
    output_hashes = {
        path.name: sha256_file(path)
        for path in [
            per_query_parquet,
            per_query_jsonl,
            metrics_path,
            report_path,
            output_dir / "provenance_receipt.json",
        ]
    }
    diagnostic_receipt = {
        "schema_version": 1,
        "status": "verified",
        "diagnostic_only": True,
        "can_promote_frozen_method": False,
        "verdict": verdict,
        "stable_qi_regime_exists": strongest_slice is not None,
        "strongest_valid_slice": strongest_slice,
        "stage_a3_scientifically_justified": False,
        "test_remained_untouched": True,
        "firewall": firewall_receipt,
        "provenance": provenance_receipt,
        "historical_receipt_sha256_before": receipt_hashes_before,
        "historical_receipt_sha256_after": receipt_hashes_after,
        "historical_receipts_byte_identical": True,
        "validation_query_count": len(qids),
        "seed_count": len(seeds),
        "per_query_row_count": len(frame),
        "output_sha256": output_hashes,
    }
    _write_json(output_dir / "diagnostic_receipt.json", diagnostic_receipt)
    return output_dir

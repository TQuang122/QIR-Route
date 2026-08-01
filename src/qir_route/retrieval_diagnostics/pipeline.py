from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from qir_route.diagnostics.firewall import (
    assert_diagnostic_path_allowed,
    read_allowed_json,
    verify_test_firewall,
)
from qir_route.diagnostics.provenance import sha256_file, verify_frozen_receipts
from qir_route.retrieval_diagnostics.analysis import (
    STRATEGY_ORDER,
    build_per_query_recovery,
    build_slice_analysis,
    evaluate_strategies,
    full_dense_ranking,
    verdict_from_gate,
)


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


def _assert_config_paths_allowed(config: dict[str, Any]) -> None:
    paths = [str(config["stage_a2_run"]), str(config["output_root"])]
    paths.extend(map(str, dict(config["frozen_receipts"])))
    paths.extend(map(str, dict(config["candidate_caches"])))
    paths.extend(map(str, dict(config["candidate_manifests"])))
    for value in paths:
        path = Path(value)
        assert_diagnostic_path_allowed(path)
        lowered = value.casefold()
        if "split_assignments" in lowered or lowered.endswith(".csv"):
            raise RuntimeError(f"candidate audit may not access {value}")


def _verify_cache_hashes(
    repository_root: Path, expected: dict[str, str]
) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative_path, expected_hash in sorted(expected.items()):
        path = (repository_root / relative_path).resolve()
        digest = sha256_file(path)
        if digest != expected_hash:
            raise RuntimeError(
                f"candidate cache hash mismatch for {relative_path}: "
                f"expected {expected_hash}, got {digest}"
            )
        observed[relative_path] = digest
    return observed


def _verify_manifests(
    repository_root: Path,
    manifest_paths: dict[str, str],
    cache_hashes: dict[str, str],
) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation"):
        path = (repository_root / manifest_paths[split]).resolve()
        manifest = read_allowed_json(path)
        cache_path = next(
            key for key in cache_hashes if Path(key).name.startswith(split)
        )
        if manifest.get("split") != split:
            raise RuntimeError(f"manifest split mismatch at {path}")
        if manifest.get("cache_sha256") != cache_hashes[cache_path]:
            raise RuntimeError(f"manifest cache hash mismatch at {path}")
        if manifest.get("raw_text_exported") is not False:
            raise RuntimeError(f"manifest unexpectedly exports raw text at {path}")
        manifests[split] = manifest
    return manifests


def _load_and_verify_validation(
    path: Path, manifest: dict[str, Any]
) -> dict[str, np.ndarray]:
    assert_diagnostic_path_allowed(path)
    with np.load(path, allow_pickle=False) as arrays:
        required = {
            "qids",
            "context_sha256",
            "query_embeddings",
            "document_embeddings",
            "candidate_indices",
            "candidate_scores",
            "positive_mask",
        }
        if set(arrays.files) != required:
            raise RuntimeError(
                "validation cache schema does not match the frozen contract"
            )
        data = {name: arrays[name].copy() for name in arrays.files}
    count = int(manifest["query_count"])
    dimension = int(manifest["embedding_dimension"])
    if data["qids"].shape != (count,) or data["context_sha256"].shape != (count,):
        raise RuntimeError("validation identifiers are not aligned")
    if data["query_embeddings"].shape != (count, dimension):
        raise RuntimeError("validation query embeddings are not aligned")
    if data["document_embeddings"].shape != (count, dimension):
        raise RuntimeError("validation document embeddings are not aligned")
    if len(set(data["qids"].astype(str))) != count:
        raise RuntimeError("validation query identifiers are not unique")
    if (
        not np.isfinite(data["query_embeddings"]).all()
        or not np.isfinite(data["document_embeddings"]).all()
    ):
        raise RuntimeError("validation embeddings contain non-finite values")
    query_norms = np.linalg.norm(data["query_embeddings"], axis=1)
    document_norms = np.linalg.norm(data["document_embeddings"], axis=1)
    if not np.allclose(query_norms, 1.0, atol=1e-4) or not np.allclose(
        document_norms, 1.0, atol=1e-4
    ):
        raise RuntimeError("validation embeddings are not normalized")
    expected_positive = data["candidate_indices"] == np.arange(count)[:, None]
    if not np.array_equal(expected_positive, data["positive_mask"]):
        raise RuntimeError("positive document identity mapping is invalid")
    return data


def _render_report(metrics: dict[str, Any]) -> str:
    best = metrics["best_fixed_strategy"]
    best_metrics = metrics["strategies"][best]
    return "\n".join(
        [
            "# Stage B.0 candidate-ceiling recovery audit",
            "",
            "This artifact is retrieval-diagnostic only and cannot promote the frozen QI method.",
            "The sealed test split was not read, inspected, embedded, cached, scored, or materialized.",
            "",
            f"- Verdict: `{metrics['verdict']}`",
            f"- Best fixed strategy: `{best}`",
            f"- Absolute candidate Recall improvement: {metrics['absolute_recall_improvement']:.6f}",
            f"- Missing-query recovery: {best_metrics['previously_impossible_recovered_percentage']:.6f}",
            f"- Paired bootstrap 95% CI: [{metrics['bootstrap_ci']['lower']:.6f}, {metrics['bootstrap_ci']['upper']:.6f}]",
            f"- Stage B.1 justified: `{str(metrics['stage_b1_justified']).lower()}`",
            "- Stage A.3 justified: `false`",
            "- Test remained untouched: `true`",
            "",
            "## Metric interpretation",
            "",
            "End-to-end Recall, MRR, and nDCG retain all validation queries and use each",
            "strategy's stable ranking. Candidate-ceiling Recall measures whether the gold",
            "document appears anywhere in the strategy's fixed candidate set. Conditional",
            "metrics include only previously impossible queries recovered by that strategy.",
            "",
            "## Fixed strategies",
            "",
            *[
                f"- `{name}`: candidate Recall={metrics['strategies'][name]['candidate_recall']:.6f}, "
                f"Recall@50={metrics['strategies'][name]['recall_at_50']:.6f}, "
                f"Recall@100={metrics['strategies'][name]['recall_at_100']:.6f}"
                for name in STRATEGY_ORDER
            ],
            "",
            "## Frozen decision gate",
            "",
            *[
                f"- {name}: `{str(value).lower()}`"
                for name, value in metrics["gate_checks"].items()
            ],
            "",
            "This audit does not implement Stage B.1 or Stage A.3.",
            "",
        ]
    )


def run_candidate_ceiling_audit(config_path: Path) -> Path:
    config_path = config_path.resolve()
    assert_diagnostic_path_allowed(config_path)
    if config_path.suffix.casefold() != ".json":
        raise RuntimeError("candidate-ceiling audit configuration must be JSON")
    repository_root = config_path.parent.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("diagnostic_only") is not True:
        raise ValueError("candidate-ceiling audit must be diagnostic_only")
    _assert_config_paths_allowed(config)
    stage_a2_run = (repository_root / config["stage_a2_run"]).resolve()
    firewall = verify_test_firewall(stage_a2_run)
    receipt_hashes_before = verify_frozen_receipts(
        repository_root, dict(config["frozen_receipts"])
    )
    cache_hashes_before = _verify_cache_hashes(
        repository_root, dict(config["candidate_caches"])
    )
    manifests = _verify_manifests(
        repository_root,
        dict(config["candidate_manifests"]),
        cache_hashes_before,
    )
    validation_path = stage_a2_run / "validation_candidates.npz"
    data = _load_and_verify_validation(validation_path, manifests["validation"])
    dense_scores, dense_order = full_dense_ranking(
        data["query_embeddings"], data["document_embeddings"]
    )
    rankings, strategy_metrics = evaluate_strategies(
        data["candidate_indices"], dense_order, bootstrap=dict(config["bootstrap"])
    )
    frame = build_per_query_recovery(
        data["qids"], data["context_sha256"], dense_scores, rankings
    )
    union_names = ["fused50_union_dense50", "fused50_union_dense100"]
    best_strategy = max(
        union_names,
        key=lambda name: (
            strategy_metrics[name]["candidate_recall"],
            -union_names.index(name),
        ),
    )
    baseline_recall = strategy_metrics["frozen_fused_top50"]["candidate_recall"]
    best_metrics = strategy_metrics[best_strategy]
    improvement = float(best_metrics["candidate_recall"] - baseline_recall)
    bootstrap_ci = dict(best_metrics["paired_bootstrap_ci"])
    minimum_support = int(config["minimum_slice_support"])
    verdict, gate_checks = verdict_from_gate(
        absolute_recall_improvement=improvement,
        bootstrap_lower=float(bootstrap_ci["lower"]),
        recovered_missing_percentage=float(
            best_metrics["previously_impossible_recovered_percentage"]
        ),
        already_retrievable_lost_count=int(
            best_metrics["already_retrievable_lost_count"]
        ),
        integrity_passed=True,
        query_count=len(frame),
        minimum_support=minimum_support,
    )
    slices = build_slice_analysis(
        frame,
        rankings,
        best_strategy=best_strategy,
        bootstrap=dict(config["bootstrap"]),
        minimum_support=minimum_support,
    )
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    stage_receipt_hash = receipt_hashes_before[
        next(
            path
            for path in receipt_hashes_before
            if path.endswith("stage_a2_receipt.json")
        )
    ]
    run_id = f"b0-{config_hash[:8]}-{stage_receipt_hash[:8]}"
    output_dir = (repository_root / config["output_root"] / run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    frame.to_parquet(output_dir / "per_query_candidate_recovery.parquet", index=False)
    frame.to_json(
        output_dir / "per_query_candidate_recovery.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    receipt_hashes_after = verify_frozen_receipts(
        repository_root, dict(config["frozen_receipts"])
    )
    cache_hashes_after = _verify_cache_hashes(
        repository_root, dict(config["candidate_caches"])
    )
    byte_identical = (
        receipt_hashes_before == receipt_hashes_after
        and cache_hashes_before == cache_hashes_after
    )
    if not byte_identical:
        raise RuntimeError("historical Stage A artifacts changed during the audit")
    integrity = {
        "schema_version": 1,
        "status": "verified",
        "validation_alignment_verified": True,
        "positive_document_identity_mapping_verified": True,
        "frozen_receipt_hashes_before": receipt_hashes_before,
        "frozen_receipt_hashes_after": receipt_hashes_after,
        "frozen_candidate_cache_hashes_before": cache_hashes_before,
        "frozen_candidate_cache_hashes_after": cache_hashes_after,
        "test_firewall": firewall,
        "test_remained_untouched": True,
        "historical_artifacts_byte_identical": True,
        "diagnostic_only": True,
        "can_promote_frozen_qi_method": False,
    }
    metrics = {
        "schema_version": 1,
        "experiment": "candidate_ceiling_b0",
        "verdict": verdict,
        "best_fixed_strategy": best_strategy,
        "absolute_recall_improvement": improvement,
        "bootstrap_ci": bootstrap_ci,
        "gate_checks": gate_checks,
        "strategies": strategy_metrics,
        "slices": slices,
        "metric_scopes": {
            "end_to_end": "all validation queries at the stated rank cutoff",
            "candidate_ceiling": "gold appears anywhere in the fixed candidate set",
            "conditional": "previously impossible queries recovered into the candidate set",
        },
        "stage_b1_justified": verdict == "candidate_ceiling_recoverable",
        "stage_a3_justified": False,
        "test_remained_untouched": True,
        "diagnostic_only": True,
        "can_promote_frozen_qi_method": False,
    }
    receipt = {
        "schema_version": 1,
        "status": "verified",
        "verdict": verdict,
        "best_fixed_strategy": best_strategy,
        "absolute_recall_improvement": improvement,
        "bootstrap_ci": bootstrap_ci,
        "recovered_missing_query_count": int(
            best_metrics["previously_impossible_recovered_count"]
        ),
        "recovered_missing_query_percentage": float(
            best_metrics["previously_impossible_recovered_percentage"]
        ),
        "frozen_receipt_hashes_before": receipt_hashes_before,
        "frozen_receipt_hashes_after": receipt_hashes_after,
        "frozen_candidate_cache_hashes_before": cache_hashes_before,
        "frozen_candidate_cache_hashes_after": cache_hashes_after,
        "test_remained_untouched": True,
        "historical_artifacts_byte_identical": True,
        "diagnostic_only": True,
        "can_promote_frozen_qi_method": False,
        "stage_a3_justified": False,
        "stage_b1_justified": verdict == "candidate_ceiling_recoverable",
    }
    _write_json(output_dir / "integrity_receipt.json", integrity)
    _write_json(output_dir / "candidate_ceiling_metrics.json", metrics)
    _write_json(output_dir / "candidate_ceiling_receipt.json", receipt)
    (output_dir / "candidate_ceiling_report.md").write_text(
        _render_report(metrics), encoding="utf-8"
    )
    return output_dir

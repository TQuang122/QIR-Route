from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer

from qir_route.baseline import resolve_device, sha256_file
from qir_route.stage_a.ablation import (
    AblationLane,
    paired_bootstrap_confidence_interval,
    train_ablation_lane,
)
from qir_route.stage_a.candidates import build_candidate_cache
from qir_route.stage_a.splits import audit_split_firewall, create_group_splits


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _load_per_query_metrics(
    run_dir: Path, lane: str, seed: int
) -> dict[str, np.ndarray]:
    path = run_dir / "training" / lane / f"seed_{seed}" / "best_per_query_metrics.npz"
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name].copy() for name in arrays.files}


def run_stage_a2_confirmation(config_path: Path) -> Path:
    config_path = config_path.resolve()
    repository_root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    dataset_config = config["dataset"]
    dataset_path = (repository_root / dataset_config["path"]).resolve()
    observed_hash = sha256_file(dataset_path)
    if observed_hash != dataset_config["sha256"]:
        raise RuntimeError(
            f"dataset hash mismatch: expected {dataset_config['sha256']}, got {observed_hash}"
        )
    splits = create_group_splits(
        pd.read_csv(dataset_path),
        sample_size=int(dataset_config["sample_size"]),
        split_seed=int(dataset_config["split_seed"]),
        train_fraction=float(dataset_config["train_fraction"]),
        validation_fraction=float(dataset_config["validation_fraction"]),
    )
    overlaps = audit_split_firewall(splits)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (
        repository_root / config["output_root"] / f"{timestamp}-{config_hash[:8]}"
    ).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)

    with (run_dir / "split_assignments.jsonl").open("w", encoding="utf-8") as handle:
        for split_name, rows in splits.items():
            for row in rows:
                handle.write(
                    json.dumps(
                        {
                            "qid": row.qid,
                            "context_sha256": row.context_sha256,
                            "split": split_name,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    split_receipt = {
        "schema_version": 1,
        "status": "verified",
        "dataset": dataset_config["name"],
        "dataset_sha256": observed_hash,
        "dataset_license": dataset_config["license"],
        "sample_size": sum(len(rows) for rows in splits.values()),
        "split_seed": dataset_config["split_seed"],
        "group_key_kind": "normalized_provided_context_sha256",
        "counts": {name: len(rows) for name, rows in splits.items()},
        "pairwise_overlaps": overlaps,
        "test_access_status": "assignment_only",
        "test_candidate_cache_created": False,
        "test_metrics_computed": False,
        "raw_text_exported": False,
    }
    _write_json(run_dir / "split_receipt.json", split_receipt)
    _write_json(run_dir / "resolved_config.json", config)

    model_config = config["model"]
    device = resolve_device(str(model_config["device"]))
    encoder = SentenceTransformer(
        model_config["id"], revision=model_config["revision"], device=device
    )
    encoder.max_seq_length = int(model_config["max_sequence_length"])
    candidate_config = config["candidates"]
    cache_manifests: dict[str, dict[str, Any]] = {}
    for split_name in ["train", "validation"]:
        cache_manifests[split_name] = build_candidate_cache(
            splits[split_name],
            split_name=split_name,
            encoder=encoder,
            model_config=model_config,
            candidate_config=candidate_config,
            output_path=run_dir / f"{split_name}_candidates.npz",
            force_positive=(
                split_name == "train" and bool(candidate_config["force_train_positive"])
            ),
        )
    del encoder

    training_config = config["training"]
    seeds = [int(seed) for seed in training_config["seeds"]]
    if len(seeds) != 5 or len(seeds) != len(set(seeds)):
        raise ValueError("Stage A.2 requires exactly five unique seeds")
    lanes: tuple[AblationLane, ...] = (
        "qi_only",
        "residual_qi",
        "residual_classical",
    )
    lane_receipts: dict[str, list[dict[str, Any]]] = {lane: [] for lane in lanes}
    for lane in lanes:
        for seed in seeds:
            lane_receipts[lane].append(
                train_ablation_lane(
                    run_dir / "train_candidates.npz",
                    run_dir / "validation_candidates.npz",
                    lane=lane,
                    seed=seed,
                    training_config=training_config,
                    output_dir=run_dir / "training" / lane / f"seed_{seed}",
                    requested_device=device,
                )
            )

    per_query = {
        lane: {seed: _load_per_query_metrics(run_dir, lane, seed) for seed in seeds}
        for lane in lanes
    }
    expected_qids = per_query["residual_qi"][seeds[0]]["qids"]
    for lane in lanes:
        for seed in seeds:
            if not np.array_equal(per_query[lane][seed]["qids"], expected_qids):
                raise RuntimeError("per-query metric exports are not aligned")
    bootstrap_config = config["bootstrap"]
    bootstrap_results: dict[str, dict[str, dict[str, float | int]]] = {
        "residual_qi_vs_baseline": {},
        "residual_qi_vs_residual_classical": {},
    }
    for metric_index, metric in enumerate(["ndcg_at_10", "mrr_at_10"]):
        residual_mean = np.stack(
            [per_query["residual_qi"][seed][f"model_{metric}"] for seed in seeds]
        ).mean(axis=0)
        classical_mean = np.stack(
            [per_query["residual_classical"][seed][f"model_{metric}"] for seed in seeds]
        ).mean(axis=0)
        baseline = per_query["residual_qi"][seeds[0]][f"baseline_{metric}"]
        bootstrap_results["residual_qi_vs_baseline"][metric] = (
            paired_bootstrap_confidence_interval(
                residual_mean,
                baseline,
                replicates=int(bootstrap_config["replicates"]),
                confidence=float(bootstrap_config["confidence"]),
                seed=int(bootstrap_config["seed"]) + metric_index,
            )
        )
        bootstrap_results["residual_qi_vs_residual_classical"][metric] = (
            paired_bootstrap_confidence_interval(
                residual_mean,
                classical_mean,
                replicates=int(bootstrap_config["replicates"]),
                confidence=float(bootstrap_config["confidence"]),
                seed=int(bootstrap_config["seed"]) + 100 + metric_index,
            )
        )

    baseline_ndcg = lane_receipts["residual_qi"][0]["baseline_validation_metrics"][
        "ndcg_at_10"
    ]
    residual_by_seed = {
        int(receipt["seed"]): receipt for receipt in lane_receipts["residual_qi"]
    }
    classical_by_seed = {
        int(receipt["seed"]): receipt for receipt in lane_receipts["residual_classical"]
    }
    promotion_config = config["promotion"]
    not_lower_count = sum(
        residual_by_seed[seed]["best_validation_metrics"]["ndcg_at_10"]
        >= baseline_ndcg - float(promotion_config["ndcg_tolerance"])
        for seed in seeds
    )
    comparator_win_count = sum(
        residual_by_seed[seed]["best_validation_metrics"]["ndcg_at_10"]
        > classical_by_seed[seed]["best_validation_metrics"]["ndcg_at_10"]
        for seed in seeds
    )
    gradient_finite_all = all(
        receipt["gradient_finite"] for lane in lanes for receipt in lane_receipts[lane]
    )
    ndcg_interval = bootstrap_results["residual_qi_vs_baseline"]["ndcg_at_10"]
    validation_count = len(splits["validation"])
    validation_candidate_count = cache_manifests["validation"]["candidate_count"]
    promotion_gate = {
        "required_not_lower_seed_count": promotion_config[
            "required_not_lower_seed_count"
        ],
        "required_comparator_win_seed_count": promotion_config[
            "required_comparator_win_seed_count"
        ],
        "baseline_validation_ndcg_at_10": baseline_ndcg,
        "residual_not_lower_baseline_seed_count": not_lower_count,
        "residual_beats_classical_seed_count": comparator_win_count,
        "validation_query_count": validation_count,
        "validation_candidate_count": validation_candidate_count,
        "gradient_finite_all": gradient_finite_all,
        "positive_mean_ndcg_delta": ndcg_interval["mean_delta"] > 0,
        "ndcg_confidence_interval_excludes_zero": ndcg_interval["lower"] > 0,
    }
    promotion_gate["passed"] = bool(
        not_lower_count >= int(promotion_config["required_not_lower_seed_count"])
        and comparator_win_count
        >= int(promotion_config["required_comparator_win_seed_count"])
        and validation_count >= int(promotion_config["minimum_validation_queries"])
        and validation_candidate_count
        >= int(promotion_config["minimum_candidate_count"])
        and gradient_finite_all
        and promotion_gate["positive_mean_ndcg_delta"]
        and promotion_gate["ndcg_confidence_interval_excludes_zero"]
    )
    baseline_lane = {
        "status": "verified",
        "seed_count": len(seeds),
        "validation_metrics": lane_receipts["residual_qi"][0][
            "baseline_validation_metrics"
        ],
        "trainable_parameter_count": 0,
        "test_cache_used": False,
    }
    final_receipt = {
        "schema_version": 1,
        "status": "verified",
        "scientific_status": (
            "promotion_passed" if promotion_gate["passed"] else "promotion_rejected"
        ),
        "experiment": config["experiment"],
        "config_sha256": config_hash,
        "split_receipt": split_receipt,
        "candidate_manifests": cache_manifests,
        "lanes": {"baseline": baseline_lane, **lane_receipts},
        "paired_bootstrap": bootstrap_results,
        "promotion_gate": promotion_gate,
        "test_firewall_intact": True,
    }
    _write_json(run_dir / "stage_a2_receipt.json", final_receipt)
    return run_dir

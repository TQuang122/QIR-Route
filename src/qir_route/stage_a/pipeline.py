from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer

from qir_route.baseline import resolve_device, sha256_file
from qir_route.stage_a.candidates import build_candidate_cache
from qir_route.stage_a.splits import audit_split_firewall, create_group_splits
from qir_route.stage_a.training import train_qi_head


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def run_stage_a_smoke(config_path: Path) -> Path:
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
    frame = pd.read_csv(dataset_path)
    splits = create_group_splits(
        frame,
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

    assignments_path = run_dir / "split_assignments.jsonl"
    with assignments_path.open("w", encoding="utf-8") as handle:
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
        "context_overlap": 0,
        "source_document_overlap": 0,
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
    cache_manifests: dict[str, dict[str, Any]] = {}
    for split_name in ["train", "validation"]:
        cache_manifests[split_name] = build_candidate_cache(
            splits[split_name],
            split_name=split_name,
            encoder=encoder,
            model_config=model_config,
            candidate_config=config["candidates"],
            output_path=run_dir / f"{split_name}_candidates.npz",
        )
    del encoder

    training_receipt = train_qi_head(
        run_dir / "train_candidates.npz",
        run_dir / "validation_candidates.npz",
        training_config=config["training"],
        output_dir=run_dir / "training",
        requested_device=device,
    )
    final_receipt = {
        "schema_version": 1,
        "status": "verified",
        "experiment": config["experiment"],
        "config_sha256": config_hash,
        "split_receipt": split_receipt,
        "candidate_manifests": cache_manifests,
        "training": training_receipt,
        "test_firewall_intact": True,
    }
    _write_json(run_dir / "stage_a_receipt.json", final_receipt)
    return run_dir

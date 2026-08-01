from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import resource
import stat
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from qir_route.baseline import resolve_device
from qir_route.stage_c0.core import (
    DataAcceptanceError,
    audit_structural_data,
    bm25_rankings,
    choose_candidate_lane,
    construct_document_text,
    evaluate_human_audit,
    evaluate_rankings,
    exact_dense_rankings,
    paired_lane_intervals,
    percentile_bootstrap_interval,
    reciprocal_rank_fusion,
    select_human_audit_query_ids,
    stage_c0_verdict,
)


FROZEN_DECISION_ID = "QIR-EVIRAL-C0-001"
FROZEN_DATASET_REVISION = "138308a5a1c647701b6f47bd7d14c919cd9c38fc"
FROZEN_MODEL_REVISION = "18b44161e041bf1d3a333ab5144b5b7b93f914d2"
FROZEN_PREREGISTRATION_SHA256 = (
    "c3d97fa856aec529fe0dff34839c4952f2ea760d11f7e6915458b9e75651361e"
)
FROZEN_PREREGISTRATION_COMMIT = "f49cdcb29255126ec628fc9efa5bf3446eeb358f"
FROZEN_IMPLEMENTATION_COMMIT_MESSAGE = "Implement preregistered EViRAL Stage C.0 runner"
FROZEN_ALLOWED_ASSETS = {
    "corpus/corpus-00000-of-00001.parquet": (
        "210359d579fec0f2a45bcb358fa83a4ef48081f3040f89934d8bc610e5a978d4"
    ),
    "queries/train-00000-of-00001.parquet": (
        "b536f8976901a9bd4136af853416ef14cb9b39e88209dabe71abe22254a3e334"
    ),
    "qrels/train-00000-of-00001.parquet": (
        "b50110e0798d24140ae421bf2ea665f0dc2ac5cb9e004a7443c47c594fd50d1c"
    ),
}
FROZEN_FORBIDDEN_ASSETS = {
    "queries/validation-00000-of-00001.parquet",
    "qrels/validation-00000-of-00001.parquet",
    "queries/test-00000-of-00001.parquet",
    "qrels/test-00000-of-00001.parquet",
}
FROZEN_CACHE_ROOT = "cache/stage_c0"
FROZEN_OUTPUT_ROOT = "artifacts/stage_c0"
FROZEN_RUNNER_PATHS = (
    "src/qir_route/cli.py",
    "src/qir_route/baseline.py",
    "src/qir_route/stage_c0",
    "configs/stage_c0_eviral.yaml",
    "tests/test_stage_c0.py",
)
ASSET_MAX_BYTES = {
    "corpus/corpus-00000-of-00001.parquet": 64 * 1024 * 1024,
    "queries/train-00000-of-00001.parquet": 64 * 1024 * 1024,
    "qrels/train-00000-of-00001.parquet": 16 * 1024 * 1024,
}
AUDIT_MAX_BYTES = 2 * 1024 * 1024
MODEL_MAX_BYTES = 8 * 1024 * 1024 * 1024
MODEL_MAX_FILE_BYTES = 4 * 1024 * 1024 * 1024
MODEL_MAX_FILES = 256
FROZEN_MODEL_PATTERNS = (
    "*.json",
    "*.txt",
    "*.model",
    "*.safetensors",
)
FROZEN_OUTPUT_ARTIFACTS = {
    "bootstrap_receipt.json",
    "candidate_metrics.json",
    "candidate_rankings.npz",
    "cost_receipt.json",
    "dataset_manifest.json",
    "firewall_receipt.json",
    "human_audit.jsonl",
    "partition_manifest.json",
    "preregistration_receipt.json",
    "stage_c0_receipt.json",
}
ARTIFACT_VERDICTS = {
    "benchmark_rejected",
    "candidate_ceiling_inadequate",
    "firewall_violation",
    "incomplete_technical_run",
    "stage_c1_authorized",
}
ARTIFACT_TOP_LEVEL_KEYS = {
    "dataset_manifest.json": (
        frozenset(
            {
                "schema_version",
                "status",
                "repository",
                "revision",
                "license",
                "observed_asset_sha256",
                "allowed_assets",
                "observed_schemas",
                "upstream_schema_fields",
                "materialized_query_fields",
                "prohibited_query_field_materialized",
                "corpus_count",
                "train_query_count",
                "train_qrel_count",
                "fit_count",
                "calibration_count",
                "query_qrel_sets_equal",
                "one_positive_qrel_per_query",
                "qrel_corpus_ids_exist",
                "normalized_cross_partition_duplicate_count",
                "raw_text_exported",
            }
        ),
        frozenset(
            {
                "schema_version",
                "status",
                "repository",
                "revision",
                "license",
                "observed_asset_sha256",
                "allowed_assets",
                "observed_schemas",
                "upstream_schema_fields",
                "materialized_query_fields",
                "prohibited_query_field_materialized",
                "corpus_count",
                "train_query_count",
                "train_qrel_count",
                "fit_count",
                "calibration_count",
                "query_qrel_sets_equal",
                "one_positive_qrel_per_query",
                "qrel_corpus_ids_exist",
                "normalized_cross_partition_duplicate_count",
                "raw_text_exported",
                "model_source",
            }
        ),
        frozenset(
            {
                "schema_version",
                "repository",
                "revision",
                "license",
                "allowed_assets",
                "status",
                "terminal_verdict",
                "error_type",
                "raw_text_exported",
            }
        ),
    ),
    "partition_manifest.json": (
        frozenset(
            {
                "schema_version",
                "decision_id",
                "salt",
                "partition_rule",
                "fit_count",
                "calibration_count",
                "fit_ordered_id_sha256",
                "calibration_ordered_id_sha256",
                "fit_calibration_overlap",
                "raw_text_exported",
            }
        ),
        frozenset(
            {
                "schema_version",
                "salt",
                "partition_rule",
                "status",
                "terminal_verdict",
                "error_type",
                "raw_text_exported",
            }
        ),
    ),
    "candidate_metrics.json": (
        frozenset(
            {
                "schema_version",
                "metrics",
                "selected_lane",
                "paired_lane_intervals",
                "deterministic_tie_break",
                "promotion_gate_observations",
                "promotion_thresholds",
                "secondary_metrics_have_promotion_authority",
            }
        ),
        frozenset(
            {
                "schema_version",
                "status",
                "reason",
                "selected_lane",
                "deterministic_tie_break",
                "promotion_gate_observations",
            }
        ),
        frozenset(
            {
                "schema_version",
                "status",
                "terminal_verdict",
                "error_type",
                "raw_text_exported",
            }
        ),
    ),
    "bootstrap_receipt.json": (
        frozenset(
            {
                "point_estimate",
                "lower",
                "upper",
                "confidence",
                "replicates",
                "seed",
                "query_count",
                "method",
                "per_query_value_sha256",
            }
        ),
        frozenset(
            {
                "schema_version",
                "status",
                "reason",
                "method",
                "seed",
                "replicates",
                "per_query_value_sha256",
            }
        ),
        frozenset(
            {
                "schema_version",
                "status",
                "terminal_verdict",
                "error_type",
                "raw_text_exported",
            }
        ),
    ),
    "cost_receipt.json": (
        frozenset(
            {
                "schema_version",
                "started_at_utc",
                "elapsed_seconds",
                "python",
                "platform",
                "packages",
                "peak_memory_bytes",
            }
        ),
        frozenset(
            {
                "schema_version",
                "started_at_utc",
                "elapsed_seconds",
                "python",
                "platform",
                "packages",
                "peak_memory_bytes",
                "failure_type",
            }
        ),
        frozenset(
            {
                "schema_version",
                "started_at_utc",
                "elapsed_seconds",
                "python",
                "platform",
                "packages",
                "peak_memory_bytes",
                "device",
                "bm25_seconds",
                "bm25_indexing_seconds",
                "bm25_query_seconds",
                "embedding_seconds",
                "document_embedding_seconds",
                "query_embedding_seconds",
                "exact_dense_seconds",
                "rrf_seconds",
                "query_latency_ms",
                "calibration_query_count",
                "model_source",
            }
        ),
    ),
    "firewall_receipt.json": (
        frozenset(
            {
                "schema_version",
                "allowed_accesses",
                "before",
                "after_candidate_evaluation",
                "firewall_intact",
                "validation_rows_accessed",
                "test_rows_accessed",
            }
        ),
        frozenset(
            {
                "schema_version",
                "allowed_accesses",
                "before",
                "after_candidate_evaluation",
                "firewall_intact",
                "validation_rows_accessed",
                "test_rows_accessed",
                "post_artifact_scan",
            }
        ),
        frozenset(
            {
                "schema_version",
                "allowed_accesses",
                "before",
                "after_candidate_evaluation",
                "firewall_intact",
                "validation_rows_accessed",
                "test_rows_accessed",
                "post_artifact_scan",
                "terminal_artifact_scan",
            }
        ),
    ),
    "preregistration_receipt.json": (
        frozenset(
            {
                "schema_version",
                "decision_id",
                "protocol_revision",
                "preregistration_sha256",
                "git_commit_sha",
                "config_sha256",
                "source_fingerprint",
                "frozen_runner_paths",
                "implementation_lineage_frozen",
            }
        ),
    ),
    "stage_c0_receipt.json": (
        frozenset(
            {
                "schema_version",
                "status",
                "decision_id",
                "verdict",
                "gate_checks",
                "selected_lane",
                "stage_c1_authorized",
                "firewall_intact",
                "validation_rows_accessed",
                "test_rows_accessed",
                "qi_or_classical_reranker_trained",
                "post_artifact_firewall_verified",
                "artifact_sha256",
            }
        ),
    ),
}


class StageC0FirewallViolation(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    path.chmod(0o600)


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    path.chmod(0o600)


def _read_jsonl(path: Path, *, expected_count: int) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise DataAcceptanceError(
            "human audit labels must be a non-symlink regular file"
        ) from error
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise DataAcceptanceError("human audit labels must be a regular file")
    if metadata.st_size > AUDIT_MAX_BYTES:
        os.close(descriptor)
        raise DataAcceptanceError("human audit labels exceed the fixed size limit")
    with os.fdopen(descriptor, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise DataAcceptanceError(
                    f"human audit row {line_number} is invalid JSON"
                ) from error
            if not isinstance(value, dict):
                raise DataAcceptanceError(
                    f"human audit row {line_number} is not a JSON object"
                )
            values.append(value)
            if len(values) > expected_count:
                raise DataAcceptanceError("human audit record count exceeds 200")
    return values


def _secure_mkdir(path: Path, *, parents: bool, exist_ok: bool) -> None:
    path.mkdir(parents=parents, exist_ok=exist_ok, mode=0o700)
    path.chmod(0o700)


def _secure_npz(path: Path, **arrays: np.ndarray) -> None:
    np.savez_compressed(path, **arrays)
    path.chmod(0o600)


def _hashed_identifiers(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.str_)
    return np.asarray(
        [
            hashlib.sha256(str(value).encode()).hexdigest()
            for value in array.reshape(-1)
        ],
        dtype=np.str_,
    ).reshape(array.shape)


def _resolve_frozen_root(repository_root: Path, relative: str, expected: str) -> Path:
    if relative != expected or Path(relative).is_absolute():
        raise ValueError(f"Stage C.0 root must remain {expected}")
    repository = repository_root.resolve()
    root = repository / relative
    current = repository
    for component in Path(relative).parts:
        current = current / component
        if current.is_symlink():
            raise StageC0FirewallViolation("Stage C.0 root contains a symlink")
    _secure_mkdir(root, parents=True, exist_ok=True)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(root, flags)
    try:
        metadata = os.fstat(descriptor)
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
            raise StageC0FirewallViolation(
                "Stage C.0 root ownership or permissions are unsafe"
            )
    finally:
        os.close(descriptor)
    resolved = root.resolve(strict=True)
    if not resolved.is_relative_to(repository):
        raise StageC0FirewallViolation("Stage C.0 root escapes the repository")
    return resolved


def _validate_download_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").casefold()
    allowed_host = (
        host == "huggingface.co"
        or host.endswith(".huggingface.co")
        or host == "hf.co"
        or host.endswith(".hf.co")
    )
    if parsed.scheme != "https" or not allowed_host or parsed.username is not None:
        raise StageC0FirewallViolation(f"download redirect is not allowed: {url}")


class _RestrictedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> urllib.request.Request | None:
        _validate_download_url(new_url)
        return super().redirect_request(
            request, file_pointer, code, message, headers, new_url
        )


def _validate_model_download_plan(entries: list[Any]) -> list[dict[str, Any]]:
    if not entries or len(entries) > MODEL_MAX_FILES:
        raise StageC0FirewallViolation("model snapshot file count is not allowed")
    manifest: list[dict[str, Any]] = []
    total_bytes = 0
    for entry in entries:
        filename = str(entry.filename)
        relative = Path(filename)
        file_size = int(entry.file_size)
        if relative.is_absolute() or ".." in relative.parts:
            raise StageC0FirewallViolation("model snapshot path is not allowed")
        if file_size < 0 or file_size > MODEL_MAX_FILE_BYTES:
            raise StageC0FirewallViolation("model snapshot file size is not allowed")
        total_bytes += file_size
        manifest.append({"filename": relative.as_posix(), "bytes": file_size})
    if total_bytes > MODEL_MAX_BYTES:
        raise StageC0FirewallViolation("model snapshot exceeds the byte limit")
    return sorted(manifest, key=lambda item: str(item["filename"]))


def _download_frozen_model(
    config: dict[str, Any], cache_root: Path
) -> tuple[Path, dict[str, Any], set[str]]:
    dense = config["retrieval"]["dense"]
    repository = str(dense["model_id"])
    revision = str(dense["revision"])
    model_dir = cache_root / "model"
    hub_cache = cache_root / "model_hub_cache"
    _secure_mkdir(model_dir, parents=False, exist_ok=False)
    _secure_mkdir(hub_cache, parents=False, exist_ok=False)
    plan = snapshot_download(
        repo_id=repository,
        repo_type="model",
        revision=revision,
        cache_dir=hub_cache,
        allow_patterns=list(FROZEN_MODEL_PATTERNS),
        max_workers=1,
        dry_run=True,
    )
    if not isinstance(plan, list):
        raise StageC0FirewallViolation("model snapshot plan is unavailable")
    planned_manifest = _validate_model_download_plan(plan)
    previous_umask = os.umask(0o077)
    try:
        downloaded = snapshot_download(
            repo_id=repository,
            repo_type="model",
            revision=revision,
            cache_dir=hub_cache,
            local_dir=model_dir,
            allow_patterns=list(FROZEN_MODEL_PATTERNS),
            max_workers=1,
        )
    finally:
        os.umask(previous_umask)
    if Path(downloaded).resolve() != model_dir.resolve():
        raise StageC0FirewallViolation("model snapshot escaped the isolated cache")
    for record in planned_manifest:
        planned_path = model_dir / str(record["filename"])
        if (
            planned_path.is_symlink()
            or not planned_path.is_file()
            or planned_path.stat().st_size != int(record["bytes"])
        ):
            raise StageC0FirewallViolation(
                "model snapshot differs from its bounded download plan"
            )
    planned_names = {str(record["filename"]) for record in planned_manifest}
    observed_payload_names = {
        path.relative_to(model_dir).as_posix()
        for path in model_dir.rglob("*")
        if path.is_file() and path.relative_to(model_dir).parts[0] != ".cache"
    }
    if observed_payload_names != planned_names:
        raise StageC0FirewallViolation(
            "model snapshot contains unplanned payload files"
        )
    observed_files: set[str] = set()
    observed_manifest: list[dict[str, Any]] = []
    for path in sorted(cache_root.rglob("*")):
        if path.is_symlink():
            raise StageC0FirewallViolation("model snapshot contains a symlink")
        if path.is_dir():
            path.chmod(0o700)
            continue
        path.chmod(0o600)
        relative = path.relative_to(cache_root).as_posix()
        if relative in FROZEN_ALLOWED_ASSETS:
            continue
        observed_files.add(relative)
        observed_manifest.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return (
        model_dir,
        {
            "repository": repository,
            "revision": revision,
            "planned_files": planned_manifest,
            "observed_files": observed_manifest,
            "trust_remote_code": False,
        },
        observed_files,
    )


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def load_stage_c0_config(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("Stage C.0 configuration must be a YAML mapping")
    validate_frozen_config(config)
    return config


def validate_frozen_config(config: dict[str, Any]) -> None:
    _require(config.get("schema_version") == 1, "schema version must be 1")
    _require(
        config.get("decision_id") == FROZEN_DECISION_ID,
        "decision ID differs from the preregistration",
    )
    _require(
        config.get("status") == "PREREGISTERED_NOT_RUN",
        "configuration status must remain PREREGISTERED_NOT_RUN",
    )
    preregistration = dict(config.get("preregistration", {}))
    _require(
        preregistration.get("path") == "outputs/stage_c0_preregistration.md",
        "preregistration path differs from the frozen contract",
    )
    _require(
        preregistration.get("sha256") == FROZEN_PREREGISTRATION_SHA256,
        "preregistration hash differs from the frozen contract",
    )
    dataset = dict(config.get("dataset", {}))
    _require(dataset.get("repository") == "NIRVLab/EViRAL", "dataset differs")
    _require(
        dataset.get("revision") == FROZEN_DATASET_REVISION,
        "dataset revision differs",
    )
    _require(dataset.get("license") == "CC-BY-4.0", "dataset license differs")
    _require(dataset.get("query_field") == "query_vi", "query field must be query_vi")
    _require(
        dataset.get("prohibited_query_field") == "query_ede",
        "query_ede must remain prohibited",
    )
    _require(
        dict(dataset.get("allowed_assets", {})) == FROZEN_ALLOWED_ASSETS,
        "allowed assets differ from the preregistration",
    )
    _require(
        set(dataset.get("forbidden_assets", [])) == FROZEN_FORBIDDEN_ASSETS,
        "forbidden assets differ from the preregistration",
    )
    _require(
        dict(dataset.get("expected_counts", {}))
        == {
            "corpus": 123972,
            "train_queries": 79700,
            "train_qrels": 79700,
            "validation_queries": 17078,
            "validation_qrels": 17078,
            "test_queries": 17079,
            "test_qrels": 17079,
        },
        "dataset counts differ from the preregistration",
    )
    partition = dict(config.get("partition", {}))
    _require(partition.get("calibration_count") == 15940, "calibration count differs")
    _require(partition.get("fit_count") == 63760, "fit count differs")
    audit = dict(config.get("human_audit", {}))
    _require(audit.get("samples_per_quartile") == 50, "audit sample differs")
    _require(audit.get("minimum_supported") == 180, "audit support gate differs")
    _require(
        audit.get("maximum_non_vietnamese") == 10,
        "audit language gate differs",
    )
    retrieval = dict(config.get("retrieval", {}))
    _require(
        dict(retrieval.get("bm25", {})) == {"k1": 1.5, "b": 0.75, "input_top_k": 1000},
        "BM25 contract differs",
    )
    dense = dict(retrieval.get("dense", {}))
    _require(
        dense.get("model_id") == "AITeamVN/Vietnamese_Embedding_v2", "model differs"
    )
    _require(dense.get("revision") == FROZEN_MODEL_REVISION, "model revision differs")
    _require(dense.get("dimension") == 1024, "embedding dimension differs")
    _require(dense.get("max_sequence_length") == 2048, "sequence length differs")
    _require(dense.get("input_top_k") == 1000, "dense top-k differs")
    _require(dense.get("exact") is True, "dense retrieval must be exact")
    rrf = dict(retrieval.get("rrf", {}))
    _require(rrf == {"k": 60, "output_top_k": 100}, "RRF contract differs")
    _require(
        retrieval.get("lane_tie_break") == ["rrf", "dense", "bm25"],
        "lane tie-break differs",
    )
    bootstrap = dict(config.get("bootstrap", {}))
    _require(
        bootstrap == {"replicates": 10000, "confidence": 0.95, "seed": 20260801},
        "bootstrap contract differs",
    )
    promotion = dict(config.get("promotion", {}))
    _require(promotion.get("minimum_recall_at_100") == 0.90, "recall gate differs")
    _require(
        promotion.get("minimum_bootstrap_lower") == 0.88,
        "bootstrap gate differs",
    )
    _require(config.get("cache_root") == FROZEN_CACHE_ROOT, "cache root differs")
    _require(config.get("output_root") == FROZEN_OUTPUT_ROOT, "output root differs")


def assert_asset_allowed(asset_name: str, config: dict[str, Any]) -> None:
    normalized = Path(asset_name).as_posix()
    forbidden = set(config["dataset"]["forbidden_assets"])
    if normalized in forbidden or any(
        marker in normalized.casefold()
        for marker in (
            "queries/validation",
            "qrels/validation",
            "queries/test",
            "qrels/test",
        )
    ):
        raise StageC0FirewallViolation(f"Stage C.0 may not access {normalized}")
    if normalized not in config["dataset"]["allowed_assets"]:
        raise StageC0FirewallViolation(f"asset is not whitelisted: {normalized}")


def _artifact_key_allowed(
    artifact_name: str, ancestors: tuple[str, ...], key: str
) -> bool:
    parent = ancestors[-1]
    metric_names = {
        "mrr_at_10",
        "ndcg_at_10",
        "recall_at_10",
        "recall_at_100",
        "recall_at_20",
        "recall_at_50",
    }
    gate_names = {
        "bootstrap_lower_at_least_threshold",
        "firewall_intact",
        "human_audit_gates_passed",
        "recall_at_100_at_least_threshold",
        "run_complete",
        "structural_data_gates_passed",
    }
    model_source_keys = {
        "observed_files",
        "planned_files",
        "repository",
        "revision",
        "trust_remote_code",
    }
    if artifact_name == "dataset_manifest.json":
        if parent in {"observed_schemas", "upstream_schema_fields"}:
            return key in {"corpus", "queries", "qrels"}
        if parent in {"corpus", "queries", "qrels"}:
            return key in {
                "corpus_id",
                "passage",
                "query_ede",
                "query_id",
                "query_vi",
                "score",
                "title",
            }
        if parent == "observed_asset_sha256":
            return key in FROZEN_ALLOWED_ASSETS
        if parent == "model_source":
            return key in model_source_keys
    if artifact_name == "candidate_metrics.json":
        if parent == "metrics":
            return key in {"bm25", "dense", "rrf"}
        if parent in {"bm25", "dense", "rrf"} and "metrics" in ancestors:
            return key in metric_names
        if parent == "paired_lane_intervals":
            return bool(re.fullmatch(r"(?:bm25|dense|rrf)_vs_(?:bm25|dense|rrf)", key))
        if re.fullmatch(r"(?:bm25|dense|rrf)_vs_(?:bm25|dense|rrf)", parent):
            return key in metric_names
        if parent in metric_names and "paired_lane_intervals" in ancestors:
            return key in {
                "confidence",
                "lower",
                "point_estimate",
                "query_count",
                "replicates",
                "seed",
                "upper",
            }
        if parent == "deterministic_tie_break":
            return key in {"document_ties", "lane_metric_ties"}
        if parent == "promotion_gate_observations":
            return key in gate_names
        if parent == "promotion_thresholds":
            return key in {"minimum_bootstrap_lower", "minimum_recall_at_100"}
    if artifact_name == "cost_receipt.json":
        if parent == "packages":
            return key in {
                "numpy",
                "pandas",
                "pyarrow",
                "rank-bm25",
                "sentence-transformers",
                "torch",
            }
        if parent == "query_latency_ms":
            return key in {"bm25", "dense_exact", "rrf"}
        if parent == "model_source":
            return key in model_source_keys
    if artifact_name == "firewall_receipt.json" and parent in {
        "after_candidate_evaluation",
        "before",
        "post_artifact_scan",
        "terminal_artifact_scan",
    }:
        return key in {
            "allowed_observed_files",
            "error_type",
            "firewall_intact",
            "forbidden_artifacts",
            "status",
            "test_rows_accessed",
            "unexpected_files",
            "validation_rows_accessed",
        }
    if artifact_name == "stage_c0_receipt.json":
        if parent == "gate_checks":
            return key in gate_names
        if parent == "artifact_sha256":
            return key in FROZEN_OUTPUT_ARTIFACTS - {"stage_c0_receipt.json"}
    if parent in {"planned_files", "observed_files"}:
        return key in {"bytes", "filename", "path", "sha256"}
    return False


def _assert_artifact_nested_schema(
    artifact_name: str, value: Any, ancestors: tuple[str, ...] = ()
) -> None:
    if isinstance(value, dict):
        for raw_key, nested in value.items():
            key = str(raw_key)
            if not _artifact_key_allowed(artifact_name, ancestors, key):
                raise StageC0FirewallViolation(
                    f"artifact schema contains an unapproved key: {key}"
                )
            _assert_artifact_nested_schema(
                artifact_name, nested, ancestors=(*ancestors, key)
            )
    elif isinstance(value, list):
        for nested in value:
            _assert_artifact_nested_schema(artifact_name, nested, ancestors=ancestors)


def _assert_artifact_string_value(
    artifact_name: str, path: tuple[str, ...], value: str
) -> None:
    key = path[-1]
    parent = path[-2] if len(path) > 1 else ""
    hex_digest = r"[0-9a-f]{64}"
    if key == "status":
        if value not in {
            "failed",
            "incomplete",
            "invalidated",
            "not_available",
            "not_run",
            "verified",
        }:
            raise StageC0FirewallViolation("artifact status value is invalid")
        return
    if key in {"terminal_verdict", "verdict"}:
        if value not in ARTIFACT_VERDICTS:
            raise StageC0FirewallViolation(f"artifact {key} is invalid")
        return
    if key in {"error_type", "failure_type"}:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", value):
            raise StageC0FirewallViolation(f"artifact {key} is invalid")
        return
    if key == "selected_lane":
        if value not in {"bm25", "dense", "rrf"}:
            raise StageC0FirewallViolation("selected lane is invalid")
        return
    fixed_values = {
        "decision_id": FROZEN_DECISION_ID,
        "document_ties": "corpus_id_ascending",
        "lane_metric_ties": "rrf_then_dense_then_bm25",
        "license": "CC-BY-4.0",
        "method": "nonparametric_percentile",
        "partition_rule": 'SHA256("QIR-EVIRAL-C0-001\\0" + query_id)',
        "preregistration_sha256": FROZEN_PREREGISTRATION_SHA256,
        "reason": "human_audit_failed",
        "salt": FROZEN_DECISION_ID,
    }
    if key in fixed_values:
        if value != fixed_values[key]:
            raise StageC0FirewallViolation(f"artifact {key} value is invalid")
        return
    if key == "repository":
        expected = (
            "AITeamVN/Vietnamese_Embedding_v2"
            if "model_source" in path
            else "NIRVLab/EViRAL"
        )
        if value != expected:
            raise StageC0FirewallViolation("artifact repository is invalid")
        return
    if key == "revision":
        expected = (
            FROZEN_MODEL_REVISION if "model_source" in path else FROZEN_DATASET_REVISION
        )
        if value != expected:
            raise StageC0FirewallViolation("artifact revision is invalid")
        return
    if key in {
        "config_sha256",
        "per_query_value_sha256",
        "sha256",
        "source_fingerprint",
    } or key in {"fit_ordered_id_sha256", "calibration_ordered_id_sha256"}:
        if not re.fullmatch(hex_digest, value):
            raise StageC0FirewallViolation(f"artifact {key} is not a SHA-256")
        return
    if key == "git_commit_sha":
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise StageC0FirewallViolation("artifact git commit is invalid")
        return
    if parent == "observed_asset_sha256":
        if value != FROZEN_ALLOWED_ASSETS.get(key):
            raise StageC0FirewallViolation("observed asset hash is invalid")
        return
    if parent == "artifact_sha256":
        if not re.fullmatch(hex_digest, value):
            raise StageC0FirewallViolation("artifact hash is invalid")
        return
    if parent == "packages":
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+_-]{0,127}", value):
            raise StageC0FirewallViolation("package version is invalid")
        return
    if artifact_name == "dataset_manifest.json" and parent in {
        "corpus",
        "queries",
        "qrels",
    }:
        if not re.fullmatch(r"[A-Za-z0-9_\[\], .<>-]{1,128}", value):
            raise StageC0FirewallViolation("dataset dtype is invalid")
        return
    if key in {"filename", "path"}:
        relative = Path(value)
        if (
            not value
            or len(value) > 512
            or relative.is_absolute()
            or ".." in relative.parts
            or any(character.isspace() for character in value)
        ):
            raise StageC0FirewallViolation("model artifact path is invalid")
        return
    if key == "started_at_utc":
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise StageC0FirewallViolation("cost timestamp is invalid") from error
        return
    if key == "device":
        if not re.fullmatch(r"(?:cpu|mps|cuda(?::[0-9]+)?)", value):
            raise StageC0FirewallViolation("device value is invalid")
        return
    if key in {"platform", "python"}:
        if (
            not value
            or len(value) > 1024
            or any(ord(character) < 32 and character not in "\t" for character in value)
        ):
            raise StageC0FirewallViolation(f"artifact {key} value is invalid")
        return
    if key in {"allowed_assets", "allowed_accesses"}:
        if value not in FROZEN_ALLOWED_ASSETS:
            raise StageC0FirewallViolation("artifact asset name is invalid")
        return
    if key == "materialized_query_fields":
        if value not in {"query_id", "query_vi"}:
            raise StageC0FirewallViolation("materialized query field is invalid")
        return
    if "upstream_schema_fields" in path:
        expected_fields = {
            "corpus": {"corpus_id", "passage", "title"},
            "queries": {"query_ede", "query_id", "query_vi"},
            "qrels": {"corpus_id", "query_id", "score"},
        }
        if value not in expected_fields.get(key, set()):
            raise StageC0FirewallViolation("upstream schema field is invalid")
        return
    if key == "frozen_runner_paths":
        if value not in FROZEN_RUNNER_PATHS:
            raise StageC0FirewallViolation("frozen runner path is invalid")
        return
    if key == "allowed_observed_files":
        if not re.fullmatch(r"(?:cache|output):[^\s]{1,512}", value):
            raise StageC0FirewallViolation("observed artifact path is invalid")
        return
    if key in {"forbidden_artifacts", "unexpected_files"}:
        raise StageC0FirewallViolation(f"artifact {key} must be empty")
    raise StageC0FirewallViolation(
        f"artifact string value is not permitted at {'.'.join(path)}"
    )


def _assert_artifact_value_schema(
    artifact_name: str, value: Any, path: tuple[str, ...] = ()
) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _assert_artifact_value_schema(artifact_name, nested, path=(*path, str(key)))
    elif isinstance(value, list):
        for nested in value:
            _assert_artifact_value_schema(artifact_name, nested, path=path)
    elif isinstance(value, str):
        _assert_artifact_string_value(artifact_name, path, value)


def _validate_json_artifact(path: Path, value: Any) -> None:
    if not isinstance(value, dict):
        raise StageC0FirewallViolation("artifact root must be a JSON object")
    schemas = ARTIFACT_TOP_LEVEL_KEYS.get(path.name)
    if schemas is None or frozenset(value) not in schemas:
        raise StageC0FirewallViolation(
            f"artifact top-level schema is invalid: {path.name}"
        )
    for key, nested in value.items():
        _assert_artifact_nested_schema(path.name, nested, ancestors=(key,))
    _assert_artifact_value_schema(path.name, value)


def _validate_human_audit_record(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {
        "query_id",
        "corpus_id",
        "label",
        "reason_codes",
        "review_timestamp",
        "rubric_version",
        "reviewer_pseudonym",
    }:
        raise StageC0FirewallViolation("human audit artifact schema is invalid")
    safe_identifier = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}")
    if not safe_identifier.fullmatch(
        str(value["query_id"])
    ) or not safe_identifier.fullmatch(str(value["corpus_id"])):
        raise StageC0FirewallViolation("human audit identifiers are invalid")
    if value["label"] not in {"supported", "not_supported"}:
        raise StageC0FirewallViolation("human audit label is invalid")
    reasons = value["reason_codes"]
    if not isinstance(reasons, list) or not set(reasons).issubset(
        {
            "ambiguous",
            "empty_or_corrupt",
            "insufficient_answer",
            "non_vietnamese_query",
            "other",
            "topic_mismatch",
        }
    ):
        raise StageC0FirewallViolation("human audit reason codes are invalid")
    for key in ("rubric_version", "reviewer_pseudonym"):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", str(value[key])):
            raise StageC0FirewallViolation(f"human audit {key} is invalid")
    try:
        datetime.fromisoformat(str(value["review_timestamp"]).replace("Z", "+00:00"))
    except ValueError as error:
        raise StageC0FirewallViolation("human audit timestamp is invalid") from error


def _verify_output_artifact_content(path: Path) -> None:
    try:
        if path.suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            _validate_json_artifact(path, value)
        elif path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if path.name != "human_audit.jsonl":
                        raise StageC0FirewallViolation("unexpected JSONL artifact")
                    _validate_human_audit_record(value)
        elif path.name == "candidate_rankings.npz":
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {
                    "query_id_sha256",
                    "bm25_document_id_sha256",
                    "dense_document_id_sha256",
                    "rrf_document_id_sha256",
                }:
                    raise StageC0FirewallViolation(
                        "candidate ranking artifact schema is invalid"
                    )
                if archive["query_id_sha256"].ndim != 1 or any(
                    archive[name].ndim != 2
                    for name in (
                        "bm25_document_id_sha256",
                        "dense_document_id_sha256",
                        "rrf_document_id_sha256",
                    )
                ):
                    raise StageC0FirewallViolation(
                        "candidate ranking artifact dimensions are invalid"
                    )
                query_count = len(archive["query_id_sha256"])
                arrays = [archive[name] for name in archive.files]
                if any(array.dtype.kind not in {"S", "U"} for array in arrays) or any(
                    array.ndim == 2
                    and (array.shape[0] != query_count or array.shape[1] > 100)
                    for array in arrays
                ):
                    raise StageC0FirewallViolation(
                        "candidate ranking artifact shape or dtype is invalid"
                    )
                digest_pattern = re.compile(r"[0-9a-f]{64}")
                if any(
                    not digest_pattern.fullmatch(str(item))
                    for array in arrays
                    for item in array.reshape(-1)
                ):
                    raise StageC0FirewallViolation(
                        "candidate ranking artifact contains non-hashed identifiers"
                    )
    except (json.JSONDecodeError, OSError, ValueError, TypeError) as error:
        raise StageC0FirewallViolation(
            f"artifact content could not be verified: {path.name}"
        ) from error


def verify_stage_c0_firewall(
    cache_root: Path,
    output_dir: Path | None = None,
    *,
    allowed_cache_files: set[str] | None = None,
    expected_cache_sha256: dict[str, str] | None = None,
) -> dict[str, Any]:
    forbidden_markers = (
        "queries/validation",
        "qrels/validation",
        "queries/test",
        "qrels/test",
        "validation_candidates",
        "test_candidates",
        "validation_metrics",
        "test_metrics",
    )
    violations: list[str] = []
    observed_allowed: list[str] = []
    additional_cache_files = {
        path.casefold() for path in (allowed_cache_files or set())
    }
    additional_cache_hashes = {
        path.casefold(): digest
        for path, digest in (expected_cache_sha256 or {}).items()
    }
    if additional_cache_files != set(additional_cache_hashes):
        raise StageC0FirewallViolation(
            "every additional cache file requires an expected SHA-256"
        )
    cache_allowlist = {
        path.casefold()
        for path in set(FROZEN_ALLOWED_ASSETS).union(additional_cache_files)
    }
    cache_hashes = {
        path.casefold(): digest
        for path, digest in {
            **FROZEN_ALLOWED_ASSETS,
            **additional_cache_hashes,
        }.items()
    }
    for root_kind, root in (("cache", cache_root), ("output", output_dir)):
        if root is None or not root.exists():
            continue
        for path in root.rglob("*"):
            relative = path.relative_to(root).as_posix().casefold()
            unexpected_file = path.is_file() and (
                (root_kind == "cache" and relative not in cache_allowlist)
                or (root_kind == "output" and relative not in FROZEN_OUTPUT_ARTIFACTS)
            )
            hash_mismatch = (
                root_kind == "cache"
                and path.is_file()
                and relative in cache_hashes
                and _sha256_file(path) != cache_hashes[relative]
            )
            if (
                path.is_symlink()
                or unexpected_file
                or hash_mismatch
                or any(marker in relative for marker in forbidden_markers)
            ):
                violations.append(str(path))
            elif path.is_file():
                if root_kind == "output":
                    try:
                        _verify_output_artifact_content(path)
                    except StageC0FirewallViolation:
                        violations.append(str(path))
                        continue
                observed_allowed.append(f"{root_kind}:{relative}")
    if violations:
        raise StageC0FirewallViolation(
            f"forbidden Stage C.0 artifacts exist: {sorted(violations)}"
        )
    return {
        "status": "verified",
        "firewall_intact": True,
        "validation_rows_accessed": False,
        "test_rows_accessed": False,
        "forbidden_artifacts": [],
        "allowed_observed_files": sorted(observed_allowed),
        "unexpected_files": [],
    }


def _git_path_is_frozen(repository_root: Path, relative_path: str) -> bool:
    tracked = subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "ls-files",
            "--error-unmatch",
            relative_path,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if tracked.returncode != 0:
        return False
    changed = subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            relative_path,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return changed.returncode == 0 and not changed.stdout.strip()


def _git_implementation_lineage_frozen(repository_root: Path) -> bool:
    expected_paths = {
        "configs/stage_c0_eviral.yaml",
        "src/qir_route/cli.py",
        "src/qir_route/stage_c0/__init__.py",
        "src/qir_route/stage_c0/core.py",
        "src/qir_route/stage_c0/pipeline.py",
        "tests/test_stage_c0.py",
    }
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "show",
            "-s",
            "--format=%P%n%s",
            "HEAD",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    lines = result.stdout.splitlines()
    if len(lines) < 2:
        return False
    parents = lines[0].split()
    if parents != [FROZEN_PREREGISTRATION_COMMIT]:
        return False
    if lines[1] != FROZEN_IMPLEMENTATION_COMMIT_MESSAGE:
        return False
    paths = subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            "HEAD",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return paths.returncode == 0 and set(paths.stdout.splitlines()) == expected_paths


def dry_run_stage_c0(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_stage_c0_config(config_path)
    repository_root = config_path.parent.parent
    preregistration_path = repository_root / config["preregistration"]["path"]
    observed_hash = _sha256_file(preregistration_path)
    config_frozen = _git_path_is_frozen(
        repository_root, config_path.relative_to(repository_root).as_posix()
    )
    preregistration_frozen = _git_path_is_frozen(
        repository_root, config["preregistration"]["path"]
    )
    runner_frozen = {
        path: _git_path_is_frozen(repository_root, path) for path in FROZEN_RUNNER_PATHS
    }
    implementation_lineage_frozen = _git_implementation_lineage_frozen(repository_root)
    full_run_freeze_ready = (
        observed_hash == config["preregistration"]["sha256"]
        and preregistration_frozen
        and config_frozen
        and all(runner_frozen.values())
        and implementation_lineage_frozen
    )
    return {
        "schema_version": 1,
        "decision_id": config["decision_id"],
        "mode": "dry_run",
        "status": "verified" if full_run_freeze_ready else "blocked",
        "network_accessed": False,
        "data_downloaded": False,
        "scientific_metrics_computed": False,
        "validation_rows_accessed": False,
        "test_rows_accessed": False,
        "allowed_assets": sorted(config["dataset"]["allowed_assets"]),
        "forbidden_assets": sorted(config["dataset"]["forbidden_assets"]),
        "preregistration_hash_matches": observed_hash
        == config["preregistration"]["sha256"],
        "preregistration_git_frozen": preregistration_frozen,
        "config_git_frozen": config_frozen,
        "runner_git_frozen": runner_frozen,
        "scientific_code_git_frozen": all(
            runner_frozen[path]
            for path in (
                "src/qir_route/cli.py",
                "src/qir_route/baseline.py",
                "src/qir_route/stage_c0",
            )
        ),
        "tests_git_frozen": runner_frozen["tests/test_stage_c0.py"],
        "implementation_lineage_frozen": implementation_lineage_frozen,
        "full_run_freeze_ready": full_run_freeze_ready,
    }


def _download_allowed_assets(
    config: dict[str, Any], cache_root: Path
) -> dict[str, Path]:
    _secure_mkdir(cache_root, parents=False, exist_ok=False)
    repository = config["dataset"]["repository"]
    revision = config["dataset"]["revision"]
    paths: dict[str, Path] = {}
    opener = urllib.request.build_opener(_RestrictedRedirectHandler())
    for asset_name, expected_hash in config["dataset"]["allowed_assets"].items():
        assert_asset_allowed(asset_name, config)
        output = cache_root / asset_name
        _secure_mkdir(output.parent, parents=True, exist_ok=True)
        url = (
            f"https://huggingface.co/datasets/{repository}/resolve/{revision}/"
            f"{asset_name}"
        )
        request = urllib.request.Request(
            url, headers={"User-Agent": "qir-route-stage-c0/1"}
        )
        _validate_download_url(url)
        partial = output.with_suffix(output.suffix + ".part")
        byte_limit = ASSET_MAX_BYTES[asset_name]
        bytes_written = 0
        try:
            partial.touch(mode=0o600, exist_ok=False)
            with (
                opener.open(request, timeout=60) as response,
                partial.open("wb") as handle,
            ):
                _validate_download_url(response.geturl())
                content_length = response.headers.get("Content-Length")
                if content_length is not None and int(content_length) > byte_limit:
                    raise RuntimeError(f"asset exceeds byte limit: {asset_name}")
                while chunk := response.read(1024 * 1024):
                    bytes_written += len(chunk)
                    if bytes_written > byte_limit:
                        raise RuntimeError(f"asset exceeds byte limit: {asset_name}")
                    handle.write(chunk)
            observed = _sha256_file(partial)
            if observed != expected_hash:
                raise RuntimeError(
                    f"asset hash mismatch for {asset_name}: expected {expected_hash}, got {observed}"
                )
            partial.replace(output)
            output.chmod(0o600)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        paths[asset_name] = output
    return paths


def _package_versions() -> dict[str, str]:
    names = (
        "numpy",
        "pandas",
        "pyarrow",
        "rank-bm25",
        "sentence-transformers",
        "torch",
    )
    return {name: version(name) for name in names}


def _git_head(repository_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_frames(
    paths: dict[str, Path], config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for asset_name, path in paths.items():
        if _sha256_file(path) != config["dataset"]["allowed_assets"][asset_name]:
            raise StageC0FirewallViolation(
                f"cached asset changed after verification: {asset_name}"
            )
    corpus_path = paths["corpus/corpus-00000-of-00001.parquet"]
    queries_path = paths["queries/train-00000-of-00001.parquet"]
    qrels_path = paths["qrels/train-00000-of-00001.parquet"]
    upstream_schemas = {
        "corpus": set(pq.ParquetFile(corpus_path).schema_arrow.names),
        "queries": set(pq.ParquetFile(queries_path).schema_arrow.names),
        "qrels": set(pq.ParquetFile(qrels_path).schema_arrow.names),
    }
    if upstream_schemas != {
        "corpus": {"corpus_id", "title", "passage"},
        "queries": {"query_id", "query_vi", "query_ede"},
        "qrels": {"query_id", "corpus_id", "score"},
    }:
        raise DataAcceptanceError("upstream parquet schemas differ from the contract")
    corpus = pd.read_parquet(corpus_path, columns=["corpus_id", "title", "passage"])
    queries = pd.read_parquet(queries_path, columns=["query_id", "query_vi"])
    qrels = pd.read_parquet(qrels_path, columns=["query_id", "corpus_id", "score"])
    expected = config["dataset"]["expected_counts"]
    if len(corpus) != expected["corpus"]:
        raise DataAcceptanceError("corpus row count differs from the frozen contract")
    if len(queries) != expected["train_queries"]:
        raise DataAcceptanceError("train query count differs from the frozen contract")
    if len(qrels) != expected["train_qrels"]:
        raise DataAcceptanceError("train qrel count differs from the frozen contract")
    for asset_name, path in paths.items():
        if _sha256_file(path) != config["dataset"]["allowed_assets"][asset_name]:
            raise StageC0FirewallViolation(
                f"cached asset changed while loading: {asset_name}"
            )
    return corpus, queries, qrels


def _peak_memory_bytes() -> int:
    maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return maximum if sys.platform == "darwin" else maximum * 1024


def _artifact_hashes(output_dir: Path, excluded: set[str]) -> dict[str, str]:
    return {
        path.name: _sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name not in excluded
    }


def _ensure_failure_artifacts(
    output_dir: Path,
    config: dict[str, Any],
    *,
    verdict: str,
    error: Exception,
) -> None:
    reason = {
        "status": "not_available",
        "terminal_verdict": verdict,
        "error_type": type(error).__name__,
        "raw_text_exported": False,
    }
    dataset_manifest = output_dir / "dataset_manifest.json"
    if not dataset_manifest.exists():
        _write_json(
            dataset_manifest,
            {
                "schema_version": 1,
                "repository": config["dataset"]["repository"],
                "revision": config["dataset"]["revision"],
                "license": config["dataset"]["license"],
                "allowed_assets": sorted(config["dataset"]["allowed_assets"]),
                **reason,
            },
        )
    partition_manifest = output_dir / "partition_manifest.json"
    if not partition_manifest.exists():
        _write_json(
            partition_manifest,
            {
                "schema_version": 1,
                "salt": config["decision_id"],
                "partition_rule": 'SHA256("QIR-EVIRAL-C0-001\\0" + query_id)',
                **reason,
            },
        )
    audit_path = output_dir / "human_audit.jsonl"
    if not audit_path.exists():
        _write_jsonl(audit_path, [])
    for name in ("candidate_metrics.json", "bootstrap_receipt.json"):
        path = output_dir / name
        if not path.exists():
            _write_json(path, {"schema_version": 1, **reason})
    rankings = output_dir / "candidate_rankings.npz"
    if not rankings.exists():
        _secure_npz(
            rankings,
            query_id_sha256=np.asarray([], dtype=np.str_),
            bm25_document_id_sha256=np.empty((0, 0), dtype=np.str_),
            dense_document_id_sha256=np.empty((0, 0), dtype=np.str_),
            rrf_document_id_sha256=np.empty((0, 0), dtype=np.str_),
        )


def _finalize_failed_run(
    output_dir: Path,
    repository_root: Path,
    config_path: Path,
    config: dict[str, Any],
    started_at: datetime,
    wall_start: float,
    *,
    verdict: str,
    error: Exception,
    cache_root: Path,
) -> None:
    _ensure_failure_artifacts(output_dir, config, verdict=verdict, error=error)
    firewall_intact = verdict != "firewall_violation"
    model_cache_hashes: dict[str, str] = {}
    dataset_manifest_path = output_dir / "dataset_manifest.json"
    if dataset_manifest_path.is_file():
        try:
            manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
            for record in manifest.get("model_source", {}).get("observed_files", []):
                model_cache_hashes[str(record["path"])] = str(record["sha256"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            model_cache_hashes = {}
    try:
        scan = verify_stage_c0_firewall(
            cache_root,
            output_dir,
            allowed_cache_files=set(model_cache_hashes),
            expected_cache_sha256=model_cache_hashes,
        )
    except StageC0FirewallViolation as scan_error:
        firewall_intact = False
        verdict = "firewall_violation"
        scan = {
            "status": "failed",
            "firewall_intact": False,
            "error_type": type(scan_error).__name__,
            "validation_rows_accessed": False,
            "test_rows_accessed": False,
        }
    complete = verdict == "benchmark_rejected"
    _, checks = stage_c0_verdict(
        structural_passed=False,
        human_audit_passed=False,
        firewall_intact=firewall_intact,
        complete=complete,
        recall_at_100=0.0,
        bootstrap_lower=0.0,
        minimum_recall=float(config["promotion"]["minimum_recall_at_100"]),
        minimum_bootstrap_lower=float(config["promotion"]["minimum_bootstrap_lower"]),
    )
    _finalize_receipts(
        output_dir,
        repository_root,
        config_path,
        config,
        verdict,
        checks,
        scan,
        scan,
        started_at,
        wall_start,
        selected_lane=None,
        extra_cost={
            "failure_type": type(error).__name__,
            "peak_memory_bytes": _peak_memory_bytes(),
        },
        firewall_intact=firewall_intact,
    )


def run_stage_c0(config_path: Path, audit_labels_path: Path) -> Path:
    config_path = config_path.resolve()
    audit_labels_path = audit_labels_path.absolute()
    config = load_stage_c0_config(config_path)
    expected_audit_count = int(config["human_audit"]["samples_per_quartile"]) * 4
    audit_records = _read_jsonl(audit_labels_path, expected_count=expected_audit_count)
    repository_root = config_path.parent.parent
    preregistration_relative = str(config["preregistration"]["path"])
    if not _git_path_is_frozen(repository_root, preregistration_relative):
        raise RuntimeError("preregistration must be tracked and clean before Stage C.0")
    config_relative = config_path.relative_to(repository_root).as_posix()
    frozen_paths = (
        preregistration_relative,
        config_relative,
        *FROZEN_RUNNER_PATHS,
    )
    if not all(_git_path_is_frozen(repository_root, path) for path in frozen_paths):
        raise RuntimeError(
            "Stage C.0 protocol, config, runner, and tests must be tracked and clean"
        )
    if not _git_implementation_lineage_frozen(repository_root):
        raise RuntimeError("Stage C.0 implementation commit lineage is not frozen")
    if (
        _sha256_file(repository_root / preregistration_relative)
        != config["preregistration"]["sha256"]
    ):
        raise RuntimeError("preregistration hash mismatch")
    started_at = datetime.now(UTC)
    config_hash = _sha256_file(config_path)
    run_id = f"{started_at.strftime('%Y%m%dT%H%M%SZ')}-{config_hash[:8]}"
    output_root = _resolve_frozen_root(
        repository_root, str(config["output_root"]), FROZEN_OUTPUT_ROOT
    )
    cache_base = _resolve_frozen_root(
        repository_root, str(config["cache_root"]), FROZEN_CACHE_ROOT
    )
    output_dir = output_root / run_id
    cache_root = cache_base / run_id
    _secure_mkdir(output_dir, parents=False, exist_ok=False)
    wall_start = time.perf_counter()
    try:
        return _execute_stage_c0(
            config_path=config_path,
            audit_records=audit_records,
            config=config,
            repository_root=repository_root,
            output_dir=output_dir,
            cache_root=cache_root,
            started_at=started_at,
            wall_start=wall_start,
        )
    except DataAcceptanceError as error:
        _finalize_failed_run(
            output_dir,
            repository_root,
            config_path,
            config,
            started_at,
            wall_start,
            verdict="benchmark_rejected",
            error=error,
            cache_root=cache_root,
        )
        return output_dir
    except StageC0FirewallViolation as error:
        _finalize_failed_run(
            output_dir,
            repository_root,
            config_path,
            config,
            started_at,
            wall_start,
            verdict="firewall_violation",
            error=error,
            cache_root=cache_root,
        )
        raise RuntimeError(
            f"Stage C.0 firewall failure receipt: {output_dir}"
        ) from error
    except Exception as error:
        _finalize_failed_run(
            output_dir,
            repository_root,
            config_path,
            config,
            started_at,
            wall_start,
            verdict="incomplete_technical_run",
            error=error,
            cache_root=cache_root,
        )
        raise RuntimeError(
            f"Stage C.0 technical failure receipt: {output_dir}"
        ) from error


def _execute_stage_c0(
    *,
    config_path: Path,
    audit_records: list[dict[str, Any]],
    config: dict[str, Any],
    repository_root: Path,
    output_dir: Path,
    cache_root: Path,
    started_at: datetime,
    wall_start: float,
) -> Path:
    firewall_before = verify_stage_c0_firewall(cache_root, output_dir)
    paths = _download_allowed_assets(config, cache_root)
    corpus, queries, qrels = _load_frames(paths, config)
    partitions, structural = audit_structural_data(
        corpus,
        queries,
        qrels,
        decision_id=config["decision_id"],
        calibration_count=int(config["partition"]["calibration_count"]),
    )
    audit_query_ids = select_human_audit_query_ids(
        queries,
        partitions["calibration"],
        decision_id=config["decision_id"],
        samples_per_quartile=int(config["human_audit"]["samples_per_quartile"]),
    )
    gold_by_query = dict(
        zip(qrels["query_id"].astype(str), qrels["corpus_id"].astype(str), strict=True)
    )
    human_audit = evaluate_human_audit(
        audit_records,
        expected_query_ids=audit_query_ids,
        gold_corpus_by_query=gold_by_query,
        minimum_supported=int(config["human_audit"]["minimum_supported"]),
        maximum_non_vietnamese=int(config["human_audit"]["maximum_non_vietnamese"]),
    )

    dataset_manifest = {
        "schema_version": 1,
        "status": "verified",
        "repository": config["dataset"]["repository"],
        "revision": config["dataset"]["revision"],
        "license": config["dataset"]["license"],
        "observed_asset_sha256": {
            name: _sha256_file(path) for name, path in paths.items()
        },
        "allowed_assets": sorted(config["dataset"]["allowed_assets"]),
        "observed_schemas": {
            "corpus": {column: str(dtype) for column, dtype in corpus.dtypes.items()},
            "queries": {column: str(dtype) for column, dtype in queries.dtypes.items()},
            "qrels": {column: str(dtype) for column, dtype in qrels.dtypes.items()},
        },
        "upstream_schema_fields": {
            "corpus": pq.ParquetFile(
                paths["corpus/corpus-00000-of-00001.parquet"]
            ).schema_arrow.names,
            "queries": pq.ParquetFile(
                paths["queries/train-00000-of-00001.parquet"]
            ).schema_arrow.names,
            "qrels": pq.ParquetFile(
                paths["qrels/train-00000-of-00001.parquet"]
            ).schema_arrow.names,
        },
        "materialized_query_fields": ["query_id", "query_vi"],
        "prohibited_query_field_materialized": False,
        **structural,
    }
    partition_manifest = {
        "schema_version": 1,
        "decision_id": config["decision_id"],
        "salt": config["decision_id"],
        "partition_rule": 'SHA256("QIR-EVIRAL-C0-001\\0" + query_id)',
        "fit_count": len(partitions["fit"]),
        "calibration_count": len(partitions["calibration"]),
        "fit_ordered_id_sha256": [
            hashlib.sha256(value.encode()).hexdigest() for value in partitions["fit"]
        ],
        "calibration_ordered_id_sha256": [
            hashlib.sha256(value.encode()).hexdigest()
            for value in partitions["calibration"]
        ],
        "fit_calibration_overlap": 0,
        "raw_text_exported": False,
    }
    _write_json(output_dir / "dataset_manifest.json", dataset_manifest)
    _write_json(output_dir / "partition_manifest.json", partition_manifest)
    _write_jsonl(output_dir / "human_audit.jsonl", audit_records)

    if not human_audit["passed"]:
        _write_json(
            output_dir / "candidate_metrics.json",
            {
                "schema_version": 1,
                "status": "not_run",
                "reason": "human_audit_failed",
                "selected_lane": None,
                "deterministic_tie_break": {
                    "document_ties": "corpus_id_ascending",
                    "lane_metric_ties": "rrf_then_dense_then_bm25",
                },
                "promotion_gate_observations": {"human_audit_gates_passed": False},
            },
        )
        _write_json(
            output_dir / "bootstrap_receipt.json",
            {
                "schema_version": 1,
                "status": "not_run",
                "reason": "human_audit_failed",
                "method": "nonparametric_percentile",
                "seed": config["bootstrap"]["seed"],
                "replicates": config["bootstrap"]["replicates"],
                "per_query_value_sha256": None,
            },
        )
        _secure_npz(
            output_dir / "candidate_rankings.npz",
            query_id_sha256=np.asarray([], dtype=np.str_),
            bm25_document_id_sha256=np.empty((0, 0), dtype=np.str_),
            dense_document_id_sha256=np.empty((0, 0), dtype=np.str_),
            rrf_document_id_sha256=np.empty((0, 0), dtype=np.str_),
        )
        firewall_after = verify_stage_c0_firewall(cache_root, output_dir)
        verdict, checks = stage_c0_verdict(
            structural_passed=True,
            human_audit_passed=False,
            firewall_intact=True,
            complete=True,
            recall_at_100=0.0,
            bootstrap_lower=0.0,
            minimum_recall=float(config["promotion"]["minimum_recall_at_100"]),
            minimum_bootstrap_lower=float(
                config["promotion"]["minimum_bootstrap_lower"]
            ),
        )
        _finalize_receipts(
            output_dir,
            repository_root,
            config_path,
            config,
            verdict,
            checks,
            firewall_before,
            firewall_after,
            started_at,
            wall_start,
            selected_lane=None,
        )
        return output_dir

    corpus_ids = corpus["corpus_id"].astype(str).tolist()
    document_texts = [
        construct_document_text(title, passage)
        for title, passage in zip(corpus["title"], corpus["passage"], strict=True)
    ]
    query_text_by_id = dict(
        zip(
            queries["query_id"].astype(str),
            queries["query_vi"].astype(str),
            strict=True,
        )
    )
    calibration_ids = partitions["calibration"]
    calibration_queries = [query_text_by_id[query_id] for query_id in calibration_ids]
    calibration_gold = [gold_by_query[query_id] for query_id in calibration_ids]

    retrieval_started = time.perf_counter()
    bm25_config = config["retrieval"]["bm25"]
    bm25_timing: dict[str, float] = {}
    bm25_indices, _ = bm25_rankings(
        document_texts,
        calibration_queries,
        corpus_ids,
        top_k=int(bm25_config["input_top_k"]),
        k1=float(bm25_config["k1"]),
        b=float(bm25_config["b"]),
        timing=bm25_timing,
    )
    bm25_seconds = time.perf_counter() - retrieval_started

    dense_config = config["retrieval"]["dense"]
    device = resolve_device(str(dense_config["device"]))
    model_dir, model_source, model_cache_files = _download_frozen_model(
        config, cache_root
    )
    dataset_manifest["model_source"] = model_source
    _write_json(output_dir / "dataset_manifest.json", dataset_manifest)
    model = SentenceTransformer(
        str(model_dir),
        device=device,
        trust_remote_code=False,
        local_files_only=True,
    )
    model.max_seq_length = int(dense_config["max_sequence_length"])
    document_embedding_started = time.perf_counter()
    document_embeddings = np.asarray(
        model.encode(
            document_texts,
            batch_size=int(dense_config["batch_size"]),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ),
        dtype=np.float32,
    )
    document_embedding_seconds = time.perf_counter() - document_embedding_started
    query_embedding_started = time.perf_counter()
    query_embeddings = np.asarray(
        model.encode(
            calibration_queries,
            batch_size=int(dense_config["batch_size"]),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ),
        dtype=np.float32,
    )
    query_embedding_seconds = time.perf_counter() - query_embedding_started
    embedding_seconds = document_embedding_seconds + query_embedding_seconds
    expected_dimension = int(dense_config["dimension"])
    if (
        document_embeddings.shape[1] != expected_dimension
        or query_embeddings.shape[1] != expected_dimension
    ):
        raise RuntimeError("dense embedding dimension differs from the frozen contract")
    if (
        not np.isfinite(document_embeddings).all()
        or not np.isfinite(query_embeddings).all()
    ):
        raise RuntimeError("dense embeddings contain non-finite values")
    if not np.allclose(
        np.linalg.norm(document_embeddings, axis=1), 1.0, atol=1e-4
    ) or not np.allclose(np.linalg.norm(query_embeddings, axis=1), 1.0, atol=1e-4):
        raise RuntimeError("dense embeddings are not L2-normalized")
    dense_started = time.perf_counter()
    dense_indices, _ = exact_dense_rankings(
        query_embeddings,
        document_embeddings,
        corpus_ids,
        top_k=int(dense_config["input_top_k"]),
        query_block_size=int(dense_config["query_block_size"]),
        document_block_size=int(dense_config["document_block_size"]),
    )
    dense_seconds = time.perf_counter() - dense_started
    rrf_config = config["retrieval"]["rrf"]
    rrf_started = time.perf_counter()
    rrf = reciprocal_rank_fusion(
        bm25_indices,
        dense_indices,
        corpus_ids,
        rrf_k=int(rrf_config["k"]),
        output_top_k=int(rrf_config["output_top_k"]),
    )
    rrf_seconds = time.perf_counter() - rrf_started
    top_100 = int(rrf_config["output_top_k"])
    rankings = {
        "bm25": [list(map(int, row[:top_100])) for row in bm25_indices],
        "dense": [list(map(int, row[:top_100])) for row in dense_indices],
        "rrf": rrf,
    }
    metrics: dict[str, dict[str, float]] = {}
    vectors: dict[str, dict[str, np.ndarray]] = {}
    for lane, lane_rankings in rankings.items():
        metrics[lane], vectors[lane] = evaluate_rankings(
            lane_rankings,
            document_ids=corpus_ids,
            gold_document_ids=calibration_gold,
        )
    selected_lane = choose_candidate_lane(metrics)
    bootstrap_config = config["bootstrap"]
    bootstrap = percentile_bootstrap_interval(
        vectors[selected_lane]["recall_at_100"],
        replicates=int(bootstrap_config["replicates"]),
        confidence=float(bootstrap_config["confidence"]),
        seed=int(bootstrap_config["seed"]),
    )
    paired = paired_lane_intervals(
        vectors,
        selected_lane=selected_lane,
        replicates=int(bootstrap_config["replicates"]),
        confidence=float(bootstrap_config["confidence"]),
        seed=int(bootstrap_config["seed"]),
    )
    firewall_after = verify_stage_c0_firewall(
        cache_root,
        output_dir,
        allowed_cache_files=model_cache_files,
        expected_cache_sha256={
            str(record["path"]): str(record["sha256"])
            for record in model_source["observed_files"]
        },
    )
    verdict, checks = stage_c0_verdict(
        structural_passed=True,
        human_audit_passed=True,
        firewall_intact=True,
        complete=True,
        recall_at_100=float(metrics[selected_lane]["recall_at_100"]),
        bootstrap_lower=float(bootstrap["lower"]),
        minimum_recall=float(config["promotion"]["minimum_recall_at_100"]),
        minimum_bootstrap_lower=float(config["promotion"]["minimum_bootstrap_lower"]),
    )
    _write_json(
        output_dir / "candidate_metrics.json",
        {
            "schema_version": 1,
            "metrics": metrics,
            "selected_lane": selected_lane,
            "paired_lane_intervals": paired,
            "deterministic_tie_break": {
                "document_ties": "corpus_id_ascending",
                "lane_metric_ties": "rrf_then_dense_then_bm25",
            },
            "promotion_gate_observations": checks,
            "promotion_thresholds": {
                "minimum_recall_at_100": config["promotion"]["minimum_recall_at_100"],
                "minimum_bootstrap_lower": config["promotion"][
                    "minimum_bootstrap_lower"
                ],
            },
            "secondary_metrics_have_promotion_authority": False,
        },
    )
    selected_values = np.asarray(
        vectors[selected_lane]["recall_at_100"], dtype=np.float64
    )
    _write_json(
        output_dir / "bootstrap_receipt.json",
        {
            **bootstrap,
            "method": "nonparametric_percentile",
            "per_query_value_sha256": hashlib.sha256(
                selected_values.tobytes(order="C")
            ).hexdigest(),
        },
    )
    document_id_array = np.asarray(corpus_ids, dtype=np.str_)
    _secure_npz(
        output_dir / "candidate_rankings.npz",
        query_id_sha256=_hashed_identifiers(calibration_ids),
        bm25_document_id_sha256=_hashed_identifiers(
            document_id_array[bm25_indices[:, :top_100]]
        ),
        dense_document_id_sha256=_hashed_identifiers(
            document_id_array[dense_indices[:, :top_100]]
        ),
        rrf_document_id_sha256=_hashed_identifiers(
            np.asarray(
                [[corpus_ids[index] for index in row] for row in rrf], dtype=np.str_
            )
        ),
    )
    _finalize_receipts(
        output_dir,
        repository_root,
        config_path,
        config,
        verdict,
        checks,
        firewall_before,
        firewall_after,
        started_at,
        wall_start,
        selected_lane=selected_lane,
        extra_cost={
            "device": device,
            "bm25_seconds": bm25_seconds,
            "bm25_indexing_seconds": bm25_timing["indexing_seconds"],
            "bm25_query_seconds": bm25_timing["query_seconds"],
            "embedding_seconds": embedding_seconds,
            "document_embedding_seconds": document_embedding_seconds,
            "query_embedding_seconds": query_embedding_seconds,
            "exact_dense_seconds": dense_seconds,
            "rrf_seconds": rrf_seconds,
            "query_latency_ms": {
                "bm25": bm25_timing["query_seconds"] / len(calibration_ids) * 1000.0,
                "dense_exact": (query_embedding_seconds + dense_seconds)
                / len(calibration_ids)
                * 1000.0,
                "rrf": rrf_seconds / len(calibration_ids) * 1000.0,
            },
            "peak_memory_bytes": _peak_memory_bytes(),
            "calibration_query_count": len(calibration_ids),
            "model_source": model_source,
        },
    )
    return output_dir


def _finalize_receipts(
    output_dir: Path,
    repository_root: Path,
    config_path: Path,
    config: dict[str, Any],
    verdict: str,
    checks: dict[str, bool],
    firewall_before: dict[str, Any],
    firewall_after: dict[str, Any],
    started_at: datetime,
    wall_start: float,
    *,
    selected_lane: str | None,
    extra_cost: dict[str, Any] | None = None,
    firewall_intact: bool = True,
) -> None:
    firewall_receipt = {
        "schema_version": 1,
        "allowed_accesses": sorted(config["dataset"]["allowed_assets"]),
        "before": firewall_before,
        "after_candidate_evaluation": firewall_after,
        "firewall_intact": firewall_intact,
        "validation_rows_accessed": False,
        "test_rows_accessed": False,
    }
    _write_json(output_dir / "firewall_receipt.json", firewall_receipt)
    _write_json(
        output_dir / "cost_receipt.json",
        {
            "schema_version": 1,
            "started_at_utc": started_at.isoformat(),
            "elapsed_seconds": time.perf_counter() - wall_start,
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
            "peak_memory_bytes": _peak_memory_bytes(),
            **(extra_cost or {}),
        },
    )
    _write_json(
        output_dir / "preregistration_receipt.json",
        {
            "schema_version": 1,
            "decision_id": config["decision_id"],
            "protocol_revision": 1,
            "preregistration_sha256": config["preregistration"]["sha256"],
            "git_commit_sha": _git_head(repository_root),
            "config_sha256": _sha256_file(config_path),
            "source_fingerprint": _source_fingerprint(),
            "frozen_runner_paths": list(FROZEN_RUNNER_PATHS),
            "implementation_lineage_frozen": _git_implementation_lineage_frozen(
                repository_root
            ),
        },
    )
    model_cache_hashes = {
        str(record["path"]): str(record["sha256"])
        for record in (extra_cost or {})
        .get("model_source", {})
        .get("observed_files", [])
    }
    if firewall_intact:
        post_artifact_scan = verify_stage_c0_firewall(
            output_dir=output_dir,
            cache_root=(repository_root / config["cache_root"] / output_dir.name),
            allowed_cache_files=set(model_cache_hashes),
            expected_cache_sha256=model_cache_hashes,
        )
        firewall_receipt["post_artifact_scan"] = post_artifact_scan
        _write_json(output_dir / "firewall_receipt.json", firewall_receipt)
        verify_stage_c0_firewall(
            output_dir=output_dir,
            cache_root=(repository_root / config["cache_root"] / output_dir.name),
            allowed_cache_files=set(model_cache_hashes),
            expected_cache_sha256=model_cache_hashes,
        )
    else:
        firewall_receipt["post_artifact_scan"] = {
            "status": "failed",
            "firewall_intact": False,
        }
        _write_json(output_dir / "firewall_receipt.json", firewall_receipt)
    stage_receipt = {
        "schema_version": 1,
        "status": (
            "incomplete"
            if verdict == "incomplete_technical_run"
            else "invalidated"
            if verdict == "firewall_violation"
            else "verified"
        ),
        "decision_id": config["decision_id"],
        "verdict": verdict,
        "gate_checks": checks,
        "selected_lane": selected_lane,
        "stage_c1_authorized": verdict == "stage_c1_authorized",
        "firewall_intact": firewall_intact,
        "validation_rows_accessed": False,
        "test_rows_accessed": False,
        "qi_or_classical_reranker_trained": False,
        "post_artifact_firewall_verified": False,
        "artifact_sha256": _artifact_hashes(output_dir, {"stage_c0_receipt.json"}),
    }
    _write_json(output_dir / "stage_c0_receipt.json", stage_receipt)
    if firewall_intact:
        terminal_scan = verify_stage_c0_firewall(
            output_dir=output_dir,
            cache_root=(repository_root / config["cache_root"] / output_dir.name),
            allowed_cache_files=set(model_cache_hashes),
            expected_cache_sha256=model_cache_hashes,
        )
        firewall_receipt["terminal_artifact_scan"] = terminal_scan
        _write_json(output_dir / "firewall_receipt.json", firewall_receipt)
        stage_receipt["post_artifact_firewall_verified"] = True
        stage_receipt["artifact_sha256"] = _artifact_hashes(
            output_dir, {"stage_c0_receipt.json"}
        )
        _write_json(output_dir / "stage_c0_receipt.json", stage_receipt)
        verify_stage_c0_firewall(
            output_dir=output_dir,
            cache_root=(repository_root / config["cache_root"] / output_dir.name),
            allowed_cache_files=set(model_cache_hashes),
            expected_cache_sha256=model_cache_hashes,
        )


def run_synthetic_stage_c0_smoke(config_path: Path | None = None) -> dict[str, Any]:
    config = load_stage_c0_config(config_path.resolve()) if config_path else None
    document_ids = [f"d-{index}" for index in range(8)]
    document_texts = [f"topic{index} answer" for index in range(8)]
    query_texts = [f"topic{index}" for index in range(8)]
    gold = document_ids.copy()
    bm25_indices, _ = bm25_rankings(
        document_texts,
        query_texts,
        document_ids,
        top_k=8,
        k1=1.5,
        b=0.75,
    )
    embeddings = np.eye(8, dtype=np.float32)
    dense_indices, _ = exact_dense_rankings(
        embeddings,
        embeddings,
        document_ids,
        top_k=8,
        query_block_size=3,
        document_block_size=3,
    )
    rrf = reciprocal_rank_fusion(
        bm25_indices, dense_indices, document_ids, rrf_k=60, output_top_k=8
    )
    rankings = {
        "bm25": [list(map(int, row)) for row in bm25_indices],
        "dense": [list(map(int, row)) for row in dense_indices],
        "rrf": rrf,
    }
    metrics: dict[str, dict[str, float]] = {}
    vectors: dict[str, dict[str, np.ndarray]] = {}
    for lane, lane_rankings in rankings.items():
        metrics[lane], vectors[lane] = evaluate_rankings(
            lane_rankings, document_ids=document_ids, gold_document_ids=gold
        )
    selected = choose_candidate_lane(metrics)
    bootstrap = percentile_bootstrap_interval(
        vectors[selected]["recall_at_100"],
        replicates=500,
        confidence=0.95,
        seed=20260801,
    )
    checks = {
        "three_candidate_lanes_exercised": set(metrics) == {"bm25", "dense", "rrf"},
        "deterministic_lane_tie_break_exercised": selected == "rrf",
        "bootstrap_exercised": bootstrap["replicates"] == 500,
    }
    return {
        "schema_version": 1,
        "decision_id": config["decision_id"] if config else FROZEN_DECISION_ID,
        "mode": "synthetic_smoke",
        "status": "verified",
        "network_accessed": False,
        "data_downloaded": False,
        "validation_rows_accessed": False,
        "test_rows_accessed": False,
        "config_validated": config is not None,
        "synthetic_query_count": 8,
        "selected_lane": selected,
        "metrics": metrics,
        "bootstrap": bootstrap,
        "smoke_checks": checks,
        "smoke_passed": all(checks.values()),
        "scientific_verdict": None,
        "stage_c1_authorized": False,
    }

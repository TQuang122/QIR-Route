from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from qir_route.metrics import evaluate_single_positive, stable_order

TOKEN_PATTERN = re.compile(r"[0-9A-Za-zÀ-ỹ_]+")


@dataclass(frozen=True)
class SmokeData:
    qids: list[str]
    questions: list[str]
    contexts: list[str]
    gold_indices: list[int]
    context_hashes: list[str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    normalized = "".join(
        char for char in normalized if unicodedata.category(char) not in {"Cc", "Cf"}
    )
    return " ".join(normalized.split())


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


def prefer_unique_sample(
    frame: pd.DataFrame, sample_size: int, seed: int
) -> pd.DataFrame:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    keyed = frame.copy()
    keyed["_context_key"] = keyed["context"].map(normalize_key)
    shuffled = keyed.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    unique = shuffled.drop_duplicates(subset=["_context_key"], keep="first")
    if len(unique) < sample_size:
        raise ValueError(
            f"requested {sample_size} unique contexts, but only {len(unique)} are available"
        )
    return unique.head(sample_size).drop(columns="_context_key").reset_index(drop=True)


def load_smoke_data(dataset_path: Path, sample_size: int, seed: int) -> SmokeData:
    frame = pd.read_csv(dataset_path)
    required = {"qid", "question", "context"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")
    frame = frame.dropna(subset=["question", "context"]).reset_index(drop=True)
    sampled = prefer_unique_sample(frame, sample_size=sample_size, seed=seed)

    contexts = sampled["context"].astype(str).tolist()
    context_hashes = [
        hashlib.sha256(normalize_key(text).encode()).hexdigest() for text in contexts
    ]
    return SmokeData(
        qids=sampled["qid"].astype(str).tolist(),
        questions=sampled["question"].astype(str).tolist(),
        contexts=contexts,
        gold_indices=list(range(len(sampled))),
        context_hashes=context_hashes,
    )


def minmax_rowwise(scores: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    minimum = scores.min(axis=1, keepdims=True)
    maximum = scores.max(axis=1, keepdims=True)
    denominator = np.maximum(maximum - minimum, epsilon)
    return (scores - minimum) / denominator


def bm25_scores(data: SmokeData, *, k1: float, b: float) -> np.ndarray:
    index = BM25Okapi([tokenize(text) for text in data.contexts], k1=k1, b=b)
    return np.stack(
        [index.get_scores(tokenize(question)) for question in data.questions]
    ).astype(np.float32)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def dense_scores(
    data: SmokeData,
    *,
    model_id: str,
    revision: str,
    expected_dimension: int,
    max_sequence_length: int,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    resolved_device = resolve_device(device)
    model = SentenceTransformer(model_id, revision=revision, device=resolved_device)
    model.max_seq_length = max_sequence_length
    started = time.perf_counter()
    document_embeddings = model.encode(
        data.contexts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    query_embeddings = model.encode(
        data.questions,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    elapsed = time.perf_counter() - started
    if document_embeddings.shape[1] != expected_dimension:
        raise RuntimeError(
            f"expected {expected_dimension} embedding dimensions, got {document_embeddings.shape[1]}"
        )
    scores = np.asarray(query_embeddings @ document_embeddings.T, dtype=np.float32)
    return scores, {
        "device": resolved_device,
        "embedding_seconds": elapsed,
        "embedding_dimension": int(document_embeddings.shape[1]),
    }


def _git_head(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _tree_fingerprint(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def run_smoke(config_path: Path) -> Path:
    config_path = config_path.resolve()
    repository_root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_config = config["dataset"]
    model_config = config["model"]
    retrieval_config = config["retrieval"]
    dataset_path = (repository_root / dataset_config["path"]).resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")
    observed_dataset_hash = sha256_file(dataset_path)
    if observed_dataset_hash != dataset_config["sha256"]:
        raise RuntimeError(
            f"dataset hash mismatch: expected {dataset_config['sha256']}, got {observed_dataset_hash}"
        )

    started_at = datetime.now(UTC)
    wall_start = time.perf_counter()
    data = load_smoke_data(
        dataset_path,
        sample_size=int(dataset_config["sample_size"]),
        seed=int(dataset_config["sample_seed"]),
    )
    bm25_start = time.perf_counter()
    lexical_scores = bm25_scores(
        data,
        k1=float(retrieval_config["bm25_k1"]),
        b=float(retrieval_config["bm25_b"]),
    )
    bm25_seconds = time.perf_counter() - bm25_start
    semantic_scores, dense_cost = dense_scores(
        data,
        model_id=model_config["id"],
        revision=model_config["revision"],
        expected_dimension=int(model_config["expected_dimension"]),
        max_sequence_length=int(model_config["max_sequence_length"]),
        batch_size=int(model_config["batch_size"]),
        device=model_config["device"],
    )
    dense_weight = float(retrieval_config["dense_weight"])
    hybrid_scores = dense_weight * minmax_rowwise(semantic_scores) + (
        1.0 - dense_weight
    ) * minmax_rowwise(lexical_scores)

    methods = {
        "bm25": lexical_scores,
        "dense": semantic_scores,
        "hybrid_minmax_alpha": hybrid_scores,
    }
    metrics: dict[str, dict[str, float]] = {}
    ranks_by_method: dict[str, list[int]] = {}
    for name, scores in methods.items():
        metrics[name], ranks_by_method[name] = evaluate_single_positive(
            scores,
            data.gold_indices,
            retrieval_config["ks"],
        )

    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    run_id = f"{started_at.strftime('%Y%m%dT%H%M%SZ')}-{config_hash[:8]}"
    run_dir = (repository_root / config["output_root"] / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)

    per_query_path = run_dir / "per_query.jsonl"
    with per_query_path.open("w", encoding="utf-8") as handle:
        for index, qid in enumerate(data.qids):
            record: dict[str, Any] = {
                "qid": qid,
                "query_sha256": hashlib.sha256(
                    data.questions[index].encode()
                ).hexdigest(),
                "gold_document_sha256": data.context_hashes[index],
                "gold_rank": {
                    name: ranks[index] for name, ranks in ranks_by_method.items()
                },
            }
            for name, scores in methods.items():
                top_indices = stable_order(scores[index])[:10]
                record[f"{name}_top10_document_sha256"] = [
                    data.context_hashes[int(doc_index)] for doc_index in top_indices
                ]
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    cost = {
        "bm25_seconds": bm25_seconds,
        **dense_cost,
        "total_seconds": time.perf_counter() - wall_start,
        "query_count": len(data.questions),
        "document_count": len(data.contexts),
    }
    source_files = list((repository_root / "src" / "qir_route").rglob("*.py"))
    manifest = {
        "schema_version": 1,
        "status": "verified",
        "experiment": config["experiment"],
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "command": f"qir smoke-baseline --config {config_path.relative_to(repository_root)}",
        "config_sha256": config_hash,
        "source_tree_sha256": _tree_fingerprint(source_files, repository_root),
        "dataset": {
            "name": dataset_config["name"],
            "sha256": observed_dataset_hash,
            "license": dataset_config["license"],
            "sample_seed": dataset_config["sample_seed"],
            "sample_size": len(data.questions),
            "raw_text_exported": False,
        },
        "model": {
            "id": model_config["id"],
            "revision": model_config["revision"],
            "frozen": True,
        },
        "upstream": {
            "vire_commit": _git_head(repository_root / "upstream" / "ViRE"),
            "qiepsm_commit": _git_head(repository_root / "upstream" / "qiepsm"),
            "source_copied": False,
            "software_license_detected": False,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                name: version(name)
                for name in [
                    "numpy",
                    "pandas",
                    "rank-bm25",
                    "sentence-transformers",
                    "torch",
                ]
            },
        },
        "artifacts": ["manifest.json", "metrics.json", "cost.json", "per_query.jsonl"],
    }
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "cost.json", cost)
    _write_json(run_dir / "manifest.json", manifest)
    return run_dir

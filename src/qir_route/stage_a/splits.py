from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from qir_route.baseline import normalize_key, prefer_unique_sample


@dataclass(frozen=True)
class SplitRow:
    qid: str
    question: str
    context: str
    context_sha256: str


def _context_hash(context: str) -> str:
    return hashlib.sha256(normalize_key(context).encode()).hexdigest()


def create_group_splits(
    frame: pd.DataFrame,
    *,
    sample_size: int,
    split_seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, list[SplitRow]]:
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("train and validation fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test partition")
    required = {"qid", "question", "context"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")
    clean = frame.dropna(subset=["question", "context"]).reset_index(drop=True)
    sampled = prefer_unique_sample(clean, sample_size=sample_size, seed=split_seed)
    rows = [
        SplitRow(
            qid=str(row["qid"]),
            question=str(row["question"]),
            context=str(row["context"]),
            context_sha256=_context_hash(str(row["context"])),
        )
        for _, row in sampled.iterrows()
    ]
    group_hashes = [row.context_sha256 for row in rows]
    if len(group_hashes) != len(set(group_hashes)):
        raise RuntimeError("unique-context sampling produced duplicate source groups")

    generator = np.random.default_rng(split_seed)
    order = generator.permutation(len(rows)).tolist()
    train_count = max(1, int(len(rows) * train_fraction))
    validation_count = max(1, int(len(rows) * validation_fraction))
    if train_count + validation_count >= len(rows):
        raise ValueError("sample is too small for three non-empty partitions")
    partition_indices = {
        "train": order[:train_count],
        "validation": order[train_count : train_count + validation_count],
        "test": order[train_count + validation_count :],
    }
    splits = {
        name: [rows[index] for index in indices]
        for name, indices in partition_indices.items()
    }
    audit_split_firewall(splits)
    return splits


def audit_split_firewall(splits: dict[str, list[SplitRow]]) -> dict[str, int]:
    expected = {"train", "validation", "test"}
    if set(splits) != expected:
        raise ValueError(f"split names must be {sorted(expected)}")
    hashes = {
        name: {row.context_sha256 for row in rows} for name, rows in splits.items()
    }
    overlaps = {
        "train_validation_overlap": len(hashes["train"] & hashes["validation"]),
        "train_test_overlap": len(hashes["train"] & hashes["test"]),
        "validation_test_overlap": len(hashes["validation"] & hashes["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"split firewall violation: {overlaps}")
    return overlaps

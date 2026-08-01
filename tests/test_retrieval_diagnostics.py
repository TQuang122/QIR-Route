from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from qir_route.diagnostics.firewall import FirewallViolation
from qir_route.retrieval_diagnostics.analysis import (
    build_per_query_recovery,
    build_slice_analysis,
    evaluate_strategies,
    full_dense_ranking,
    positive_ranks,
    stable_union,
    verdict_from_gate,
)
from qir_route.retrieval_diagnostics.pipeline import (
    _assert_config_paths_allowed,
    _verify_cache_hashes,
)


BOOTSTRAP = {"replicates": 300, "confidence": 0.95, "seed": 20260801}


def test_forbidden_test_and_split_paths_fail_before_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[Path] = []

    def tracked_read_bytes(path: Path) -> bytes:
        opened.append(path)
        return b"sentinel"

    monkeypatch.setattr(Path, "read_bytes", tracked_read_bytes)
    base = {
        "stage_a2_run": "artifacts/stage_a2/run",
        "output_root": "artifacts/candidate_ceiling_b0",
        "frozen_receipts": {},
        "candidate_manifests": {
            "train": "train_candidates.manifest.json",
            "validation": "validation_candidates.manifest.json",
        },
    }
    with pytest.raises(FirewallViolation):
        _assert_config_paths_allowed(
            {**base, "candidate_caches": {"test_candidates.npz": "hash"}}
        )
    with pytest.raises(RuntimeError, match="split_assignments"):
        _assert_config_paths_allowed(
            {**base, "candidate_caches": {"split_assignments.jsonl": "hash"}}
        )
    with pytest.raises(RuntimeError, match="may not access"):
        _assert_config_paths_allowed(
            {**base, "candidate_caches": {"original_csconda.csv": "hash"}}
        )
    assert opened == []


def test_full_dense_ranking_uses_supplied_stored_embeddings() -> None:
    query = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    documents = np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32)
    scores, order = full_dense_ranking(query, documents)
    np.testing.assert_allclose(scores, query @ documents.T)
    np.testing.assert_array_equal(order, [[0, 1], [1, 0]])


def test_stable_union_preserves_frozen_order_and_removes_duplicates() -> None:
    assert stable_union([4, 2, 1], [2, 5, 4, 3], 6) == [4, 2, 1, 5, 3]
    assert stable_union([4, 2, 1], [5, 3], 4) == [4, 2, 1, 5]


def test_positive_rank_calculation_uses_aligned_document_identity() -> None:
    ranks = positive_ranks([[2, 0, 1], [0, 2, 1], [1, 0]])
    np.testing.assert_equal(ranks[:2], [2.0, 3.0])
    assert np.isnan(ranks[2])


def test_recovery_metrics_are_paired_by_query() -> None:
    frozen = np.asarray([[0, 1], [0, 2], [0, 1], [0, 1]], dtype=np.int32)
    dense = np.asarray(
        [[0, 1, 2, 3], [1, 0, 2, 3], [2, 0, 1, 3], [3, 0, 1, 2]],
        dtype=np.int32,
    )
    _, metrics = evaluate_strategies(frozen, dense, bootstrap=BOOTSTRAP)
    union = metrics["fused50_union_dense50"]
    assert union["previously_impossible_recovered_count"] == 3
    assert union["already_retrievable_lost_count"] == 0
    assert union["paired_candidate_recall_delta_vs_frozen"] == pytest.approx(0.75)


def test_bootstrap_ci_is_deterministic() -> None:
    frozen = np.asarray([[1], [0], [0], [0]], dtype=np.int32)
    dense = np.asarray(
        [[0, 1, 2, 3], [1, 0, 2, 3], [2, 0, 1, 3], [3, 0, 1, 2]],
        dtype=np.int32,
    )
    _, first = evaluate_strategies(frozen, dense, bootstrap=BOOTSTRAP)
    _, second = evaluate_strategies(frozen, dense, bootstrap=BOOTSTRAP)
    assert (
        first["fused50_union_dense100"]["paired_bootstrap_ci"]
        == second["fused50_union_dense100"]["paired_bootstrap_ci"]
    )


def test_slice_with_fewer_than_100_queries_cannot_define_recoverable_regime() -> None:
    count = 99
    scores = np.eye(count, dtype=np.float32)
    dense_order = np.argsort(-scores, axis=1, kind="stable")
    frozen = np.tile(np.arange(50, dtype=np.int32), (count, 1))
    rankings, _ = evaluate_strategies(frozen, dense_order, bootstrap=BOOTSTRAP)
    frame = build_per_query_recovery(
        np.asarray([f"q-{index}" for index in range(count)]),
        np.asarray([f"g-{index}" for index in range(count)]),
        scores,
        rankings,
    )
    slices = build_slice_analysis(
        frame,
        rankings,
        best_strategy="fused50_union_dense100",
        bootstrap=BOOTSTRAP,
        minimum_support=100,
    )
    assert slices
    assert all(item["can_define_recoverable_slice"] is False for item in slices)
    assert all(item["support_warning"] is True for item in slices)


def test_cache_hash_verification_is_read_only(tmp_path: Path) -> None:
    cache = tmp_path / "train_candidates.npz"
    cache.write_bytes(b"frozen-cache")
    before = cache.read_bytes()
    digest = hashlib.sha256(before).hexdigest()
    observed = _verify_cache_hashes(tmp_path, {cache.name: digest})
    assert observed == {cache.name: digest}
    assert cache.read_bytes() == before


@pytest.mark.parametrize(
    ("improvement", "lower", "recovery", "lost", "integrity", "count", "expected"),
    [
        (0.05, 0.0001, 0.20, 0, True, 100, "candidate_ceiling_recoverable"),
        (0.0499, 0.0001, 0.20, 0, True, 100, "candidate_ceiling_not_recoverable"),
        (0.05, 0.0, 0.20, 0, True, 100, "candidate_ceiling_not_recoverable"),
        (0.05, 0.0001, 0.1999, 0, True, 100, "candidate_ceiling_not_recoverable"),
        (0.05, 0.0001, 0.20, 1, True, 100, "candidate_ceiling_not_recoverable"),
        (0.05, 0.0001, 0.20, 0, False, 100, "blocked"),
        (0.05, 0.0001, 0.20, 0, True, 99, "insufficient_evidence"),
    ],
)
def test_verdict_gate_uses_frozen_thresholds_exactly(
    improvement: float,
    lower: float,
    recovery: float,
    lost: int,
    integrity: bool,
    count: int,
    expected: str,
) -> None:
    verdict, _ = verdict_from_gate(
        absolute_recall_improvement=improvement,
        bootstrap_lower=lower,
        recovered_missing_percentage=recovery,
        already_retrievable_lost_count=lost,
        integrity_passed=integrity,
        query_count=count,
        minimum_support=100,
    )
    assert verdict == expected

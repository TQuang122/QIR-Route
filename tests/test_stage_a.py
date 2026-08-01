import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest
import torch

from qir_route.stage_a.ablation import (
    combine_residual_scores,
    paired_bootstrap_confidence_interval,
    ranking_metrics,
    train_ablation_lane,
)
from qir_route.stage_a.candidates import (
    build_candidate_cache,
    inject_missing_positives,
    select_top_candidates,
)
from qir_route.stage_a.models import MatchedClassicalHead
from qir_route.stage_a.splits import audit_split_firewall, create_group_splits
from qir_route.stage_a.training import (
    mean_ndcg_at_k,
    multi_positive_listwise_loss,
    train_qi_head,
)


class FakeEncoder:
    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
        del kwargs
        values = []
        for sentence in sentences:
            code = sum(ord(character) for character in sentence)
            vector = np.asarray(
                [code % 7 + 1, code % 11 + 1, code % 13 + 1, code % 17 + 1],
                dtype=np.float32,
            )
            values.append(vector / np.linalg.norm(vector))
        return np.stack(values)


def synthetic_frame(size: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "qid": [f"q-{index}" for index in range(size)],
            "question": [f"question {index}" for index in range(size)],
            "context": [f"unique context {index}" for index in range(size)],
        }
    )


def test_group_split_is_deterministic_and_firewalled() -> None:
    frame = synthetic_frame(30)
    first = create_group_splits(
        frame,
        sample_size=30,
        split_seed=20260731,
        train_fraction=0.7,
        validation_fraction=0.15,
    )
    second = create_group_splits(
        frame,
        sample_size=30,
        split_seed=20260731,
        train_fraction=0.7,
        validation_fraction=0.15,
    )
    assert {name: len(rows) for name, rows in first.items()} == {
        "train": 21,
        "validation": 4,
        "test": 5,
    }
    assert {name: [row.qid for row in rows] for name, rows in first.items()} == {
        name: [row.qid for row in rows] for name, rows in second.items()
    }
    assert audit_split_firewall(first) == {
        "train_validation_overlap": 0,
        "train_test_overlap": 0,
        "validation_test_overlap": 0,
    }


def test_firewall_rejects_source_overlap() -> None:
    splits = create_group_splits(
        synthetic_frame(12),
        sample_size=12,
        split_seed=7,
        train_fraction=0.5,
        validation_fraction=0.25,
    )
    splits["test"].append(splits["train"][0])
    with pytest.raises(RuntimeError, match="firewall violation"):
        audit_split_firewall(splits)


def test_candidate_fusion_returns_stable_topk() -> None:
    bm25 = np.asarray([[0.0, 1.0, 0.5], [1.0, 0.0, 0.5]], dtype=np.float32)
    dense = np.asarray([[0.0, 0.5, 1.0], [0.5, 1.0, 0.0]], dtype=np.float32)
    indices, scores = select_top_candidates(bm25, dense, top_k=2, dense_weight=0.7)
    assert indices.tolist() == [[2, 1], [1, 0]]
    assert scores.shape == (2, 2)


def test_train_positive_injection_replaces_only_missing_tail() -> None:
    indices = np.asarray([[1, 2], [1, 0], [0, 1]], dtype=np.int32)
    scores = np.asarray([[0.9, 0.8], [0.9, 0.8], [0.9, 0.8]], dtype=np.float32)
    full_scores = np.asarray(
        [[0.1, 0.9, 0.8], [0.8, 0.9, 0.1], [0.9, 0.8, 0.2]],
        dtype=np.float32,
    )
    injected = inject_missing_positives(indices, scores, full_scores)
    assert injected == 2
    assert indices.tolist() == [[1, 0], [1, 0], [0, 2]]
    assert np.allclose(scores[:, -1], [0.1, 0.8, 0.2])


def test_matched_classical_head_is_symmetric_and_parameter_matched() -> None:
    head = MatchedClassicalHead(embedding_dim=1024)
    queries = torch.randn(2, 1024)
    documents = torch.randn(2, 3, 1024)
    forward = head.score(queries, documents)
    reverse = torch.stack(
        [
            head.score(documents[:, index], queries.unsqueeze(1)).squeeze(1)
            for index in range(3)
        ],
        dim=1,
    )
    assert head.head_parameter_count == 6912
    assert torch.allclose(forward, reverse)


def test_zero_residual_weight_preserves_baseline_exactly() -> None:
    baseline = torch.randn(3, 5)
    correction = torch.randn(3, 5)
    combined = combine_residual_scores(baseline, correction, torch.tensor(0.0))
    assert torch.equal(combined, baseline)


def test_ranking_metrics_report_all_screening_metrics() -> None:
    scores = torch.tensor([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    positives = torch.tensor(
        [[False, True, False], [False, False, False]], dtype=torch.bool
    )
    metrics = ranking_metrics(scores, positives)
    assert np.isclose(metrics["ndcg_at_10"], 0.3154648768)
    assert metrics["mrr_at_10"] == 0.25
    assert metrics["recall_at_50"] == 0.5


def test_paired_bootstrap_is_deterministic_and_paired() -> None:
    reference = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    model = reference + np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    first = paired_bootstrap_confidence_interval(
        model, reference, replicates=500, confidence=0.95, seed=20260801
    )
    second = paired_bootstrap_confidence_interval(
        model, reference, replicates=500, confidence=0.95, seed=20260801
    )
    assert first == second
    assert np.isclose(first["mean_delta"], 0.25)
    assert first["lower"] > 0
    assert first["query_count"] == 4


def test_candidate_cache_contains_no_raw_text(tmp_path: Path) -> None:
    splits = create_group_splits(
        synthetic_frame(12),
        sample_size=12,
        split_seed=7,
        train_fraction=0.5,
        validation_fraction=0.25,
    )
    output = tmp_path / "train.npz"
    manifest = build_candidate_cache(
        splits["train"],
        split_name="train",
        encoder=FakeEncoder(),
        model_config={
            "id": "fake",
            "revision": "fixed",
            "batch_size": 4,
            "expected_dimension": 4,
        },
        candidate_config={
            "bm25_k1": 1.5,
            "bm25_b": 0.75,
            "top_k": 4,
            "dense_weight": 0.7,
        },
        output_path=output,
    )
    with np.load(output, allow_pickle=False) as arrays:
        assert set(arrays.files) == {
            "qids",
            "context_sha256",
            "query_embeddings",
            "document_embeddings",
            "candidate_indices",
            "candidate_scores",
            "positive_mask",
        }
        assert arrays["candidate_indices"].shape == (6, 4)
    assert manifest["raw_text_exported"] is False
    persisted = json.loads(output.with_suffix(".manifest.json").read_text())
    assert persisted["cache_sha256"] == manifest["cache_sha256"]


def test_multi_positive_listwise_loss_matches_definition() -> None:
    scores = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 0.0]])
    positives = torch.tensor(
        [[True, False, True], [False, True, False]], dtype=torch.bool
    )
    loss = multi_positive_listwise_loss(scores, positives, temperature=1.0)
    expected = torch.stack(
        [
            torch.logsumexp(scores[0], dim=0)
            - torch.logsumexp(scores[0, [0, 2]], dim=0),
            torch.logsumexp(scores[1], dim=0) - scores[1, 1],
        ]
    ).mean()
    assert torch.allclose(loss, expected)


def test_ndcg_counts_missing_candidates_as_zero() -> None:
    scores = torch.tensor([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    positives = torch.tensor(
        [[False, True, False], [False, False, False]], dtype=torch.bool
    )
    expected = (1 / torch.log2(torch.tensor(3.0))).item() / 2
    assert np.isclose(mean_ndcg_at_k(scores, positives, 3), expected)


def test_tiny_listwise_training_writes_checkpoint_without_test_cache(
    tmp_path: Path,
) -> None:
    generator = np.random.default_rng(20260801)
    for split, count in [("train", 12), ("validation", 6)]:
        embeddings = generator.normal(size=(count, 8)).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        candidate_indices = np.tile(np.arange(count, dtype=np.int32), (count, 1))
        positive_mask = candidate_indices == np.arange(count)[:, None]
        np.savez_compressed(
            tmp_path / f"{split}.npz",
            query_embeddings=embeddings,
            document_embeddings=embeddings,
            candidate_indices=candidate_indices,
            candidate_scores=np.linspace(1, 0, count, dtype=np.float32)[None].repeat(
                count, axis=0
            ),
            positive_mask=positive_mask,
        )
    receipt = train_qi_head(
        tmp_path / "train.npz",
        tmp_path / "validation.npz",
        training_config={
            "seed": 20260801,
            "aggregation": "mean",
            "temperature": 0.1,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 4,
            "max_epochs": 2,
            "patience": 1,
        },
        output_dir=tmp_path / "training",
        requested_device="cpu",
    )
    assert receipt["status"] == "verified"
    assert receipt["test_cache_used"] is False
    assert receipt["gradient_finite"] is True
    assert (tmp_path / "training" / "best_checkpoint.pt").is_file()


@pytest.mark.parametrize("lane", ["residual_qi", "residual_classical"])
def test_tiny_residual_ablation_is_parameter_matched_and_firewalled(
    tmp_path: Path, lane: Literal["residual_qi", "residual_classical"]
) -> None:
    generator = np.random.default_rng(20260801)
    for split, count in [("train", 8), ("validation", 4)]:
        queries = generator.normal(size=(count, 8)).astype(np.float32)
        documents = generator.normal(size=(count, 8)).astype(np.float32)
        candidate_indices = np.tile(np.arange(count, dtype=np.int32), (count, 1))
        positive_mask = candidate_indices == np.arange(count)[:, None]
        np.savez_compressed(
            tmp_path / f"{split}.npz",
            query_embeddings=queries,
            document_embeddings=documents,
            candidate_indices=candidate_indices,
            candidate_scores=generator.normal(size=(count, count)).astype(np.float32),
            positive_mask=positive_mask,
        )
    receipt = train_ablation_lane(
        tmp_path / "train.npz",
        tmp_path / "validation.npz",
        lane=lane,
        seed=20260801,
        training_config={
            "aggregation": "mean",
            "temperature": 0.1,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 4,
            "max_epochs": 2,
            "patience": 1,
            "max_grad_norm": 1e-8,
        },
        output_dir=tmp_path / lane,
        requested_device="cpu",
    )
    assert receipt["head_parameter_count"] == 54
    assert receipt["trainable_parameter_count"] == 55
    assert receipt["test_cache_used"] is False
    assert receipt["gradient_finite"] is True
    assert receipt["max_head_gradient_norm"] > 0
    assert receipt["clipped_step_rate"] > 0
    assert receipt["per_query_raw_text_exported"] is False
    assert (tmp_path / lane / "best_per_query_metrics.npz").is_file()

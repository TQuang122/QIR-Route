from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import Tensor, nn

from qir_route.baseline import resolve_device
from qir_route.quantum import AggregationMode, QuantumInspiredHead
from qir_route.stage_a.models import MatchedClassicalHead, standardize_candidate_scores
from qir_route.stage_a.training import (
    CandidateCache,
    load_candidate_cache,
    multi_positive_listwise_loss,
)

AblationLane = Literal["qi_only", "residual_qi", "residual_classical"]
ScoringHead = QuantumInspiredHead | MatchedClassicalHead


def _source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def per_query_ranking_metrics(
    scores: Tensor, positive_mask: Tensor
) -> dict[str, Tensor]:
    if scores.shape != positive_mask.shape or scores.ndim != 2:
        raise ValueError("scores and positive mask must have the same 2D shape")
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    ranked_positives = positive_mask.gather(1, order)
    cutoff_10 = min(10, scores.shape[1])
    cutoff_50 = min(50, scores.shape[1])
    top_10 = ranked_positives[:, :cutoff_10]
    ranks = torch.arange(1, cutoff_10 + 1, dtype=scores.dtype, device=scores.device)
    reciprocal = (
        torch.where(
            top_10,
            1 / ranks.unsqueeze(0),
            torch.zeros((), dtype=scores.dtype, device=scores.device),
        )
        .max(dim=1)
        .values
    )
    discounts = 1 / torch.log2(ranks + 1)
    dcg = (top_10.to(scores.dtype) * discounts).sum(dim=1)
    positive_counts = positive_mask.sum(dim=1).clamp(max=cutoff_10)
    ideal = discounts.cumsum(dim=0)[(positive_counts - 1).clamp_min(0)]
    ndcg = torch.where(positive_counts > 0, dcg / ideal, torch.zeros_like(dcg))
    recall = ranked_positives[:, :cutoff_50].any(dim=1).to(scores.dtype)
    return {"ndcg_at_10": ndcg, "mrr_at_10": reciprocal, "recall_at_50": recall}


def ranking_metrics(scores: Tensor, positive_mask: Tensor) -> dict[str, float]:
    per_query = per_query_ranking_metrics(scores, positive_mask)
    return {
        name: float(values.mean().detach().cpu()) for name, values in per_query.items()
    }


def paired_bootstrap_confidence_interval(
    model_values: np.ndarray,
    reference_values: np.ndarray,
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict[str, float | int]:
    model_values = np.asarray(model_values, dtype=np.float64)
    reference_values = np.asarray(reference_values, dtype=np.float64)
    if model_values.shape != reference_values.shape or model_values.ndim != 1:
        raise ValueError(
            "paired metric arrays must have the same one-dimensional shape"
        )
    if model_values.size == 0 or replicates <= 0:
        raise ValueError("paired bootstrap requires values and positive replicates")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    differences = model_values - reference_values
    generator = np.random.default_rng(seed)
    bootstrap_means = np.empty(replicates, dtype=np.float64)
    chunk_size = min(1000, replicates)
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        indices = generator.integers(
            0, differences.size, size=(stop - start, differences.size)
        )
        bootstrap_means[start:stop] = differences[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(bootstrap_means, [alpha, 1 - alpha])
    return {
        "mean_delta": float(differences.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "confidence": confidence,
        "replicates": replicates,
        "seed": seed,
        "query_count": int(differences.size),
    }


def combine_residual_scores(
    base_scores: Tensor,
    correction_scores: Tensor,
    residual_weight: Tensor,
) -> Tensor:
    if base_scores.shape != correction_scores.shape:
        raise ValueError("base and correction scores must have the same shape")
    return base_scores + residual_weight * standardize_candidate_scores(
        correction_scores
    )


def _make_head(lane: AblationLane, embedding_dim: int, device: str) -> ScoringHead:
    if lane in {"qi_only", "residual_qi"}:
        return QuantumInspiredHead(embedding_dim=embedding_dim).to(device)
    return MatchedClassicalHead(embedding_dim=embedding_dim).to(device)


def _head_parameter_count(head: ScoringHead) -> int:
    return sum(parameter.numel() for parameter in head.parameters())


def _raw_scores(
    head: ScoringHead,
    queries: Tensor,
    documents: Tensor,
    mode: AggregationMode,
) -> Tensor:
    if isinstance(head, QuantumInspiredHead):
        return head.score(queries, documents, mode=mode)
    return head.score(queries, documents)


def _score_cache(
    head: ScoringHead,
    cache: CandidateCache,
    *,
    lane: AblationLane,
    residual_weight: Tensor | None,
    batch_size: int,
    mode: AggregationMode,
) -> tuple[Tensor, float]:
    batches: list[Tensor] = []
    variances: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, cache.query_embeddings.shape[0], batch_size):
            stop = min(start + batch_size, cache.query_embeddings.shape[0])
            indices = cache.candidate_indices[start:stop]
            documents = cache.document_embeddings[indices]
            raw = _raw_scores(head, cache.query_embeddings[start:stop], documents, mode)
            variances.append(raw.var(dim=1, unbiased=False))
            if lane == "qi_only":
                batches.append(raw)
            else:
                if residual_weight is None:
                    raise RuntimeError("residual lane requires a residual weight")
                batches.append(
                    combine_residual_scores(
                        cache.candidate_scores[start:stop], raw, residual_weight
                    )
                )
    return (
        torch.cat(batches, dim=0),
        float(torch.cat(variances).mean().detach().cpu()),
    )


def train_ablation_lane(
    train_cache_path: Path,
    validation_cache_path: Path,
    *,
    lane: AblationLane,
    seed: int,
    training_config: dict[str, Any],
    output_dir: Path,
    requested_device: str,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device(requested_device)
    train_cache = load_candidate_cache(train_cache_path, device)
    validation_cache = load_candidate_cache(validation_cache_path, device)
    embedding_dim = int(train_cache.query_embeddings.shape[1])
    if validation_cache.query_embeddings.shape[1] != embedding_dim:
        raise ValueError("train and validation embedding dimensions differ")
    if not bool(train_cache.positive_mask.any(dim=1).all()):
        raise RuntimeError("every training query must contain a positive candidate")
    mode = cast(AggregationMode, str(training_config["aggregation"]))
    head = _make_head(lane, embedding_dim, device)
    residual_weight = (
        nn.Parameter(torch.zeros((), device=device)) if lane != "qi_only" else None
    )
    parameters = list(head.parameters())
    if residual_weight is not None:
        parameters.append(residual_weight)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    batch_size = int(training_config["batch_size"])
    temperature = float(training_config["temperature"])
    max_epochs = int(training_config["max_epochs"])
    patience = int(training_config["patience"])
    max_grad_norm = float(training_config.get("max_grad_norm", math.inf))
    if min(batch_size, max_epochs, patience, max_grad_norm) <= 0:
        raise ValueError(
            "batch size, epochs, patience, and gradient norm must be positive"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_dir / "best_checkpoint.pt"
    history: list[dict[str, Any]] = []
    best_ndcg = -math.inf
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    best_residual_weight: float | None = None
    best_score_variance = 0.0
    best_per_query: dict[str, np.ndarray] = {}
    stale_epochs = 0
    max_gradient_norm = 0.0
    max_head_gradient_norm = 0.0
    clipped_step_count = 0
    optimizer_step_count = 0
    gradient_finite = True
    started = time.perf_counter()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train_count = int(train_cache.query_embeddings.shape[0])
    baseline_per_query = per_query_ranking_metrics(
        validation_cache.candidate_scores, validation_cache.positive_mask
    )
    baseline_metrics = {
        name: float(values.mean().detach().cpu())
        for name, values in baseline_per_query.items()
    }

    for epoch in range(1, max_epochs + 1):
        head.train()
        permutation = torch.randperm(train_count, generator=generator)
        epoch_losses: list[float] = []
        epoch_clipped_steps = 0
        epoch_optimizer_steps = 0
        for start in range(0, train_count, batch_size):
            indices = permutation[start : start + batch_size].to(device)
            candidate_indices = train_cache.candidate_indices[indices]
            documents = train_cache.document_embeddings[candidate_indices]
            optimizer.zero_grad(set_to_none=True)
            raw = _raw_scores(
                head, train_cache.query_embeddings[indices], documents, mode
            )
            if lane == "qi_only":
                scores = raw
            else:
                if residual_weight is None:
                    raise RuntimeError("residual lane requires a residual weight")
                scores = combine_residual_scores(
                    train_cache.candidate_scores[indices], raw, residual_weight
                )
            loss = multi_positive_listwise_loss(
                scores, train_cache.positive_mask[indices], temperature
            )
            loss.backward()
            squared_norm = 0.0
            squared_head_norm = 0.0
            for parameter in parameters:
                if parameter.grad is not None:
                    gradient_finite = gradient_finite and bool(
                        torch.isfinite(parameter.grad).all().cpu()
                    )
                    value = float(parameter.grad.detach().norm().cpu()) ** 2
                    squared_norm += value
            for parameter in head.parameters():
                if parameter.grad is not None:
                    squared_head_norm += (
                        float(parameter.grad.detach().norm().cpu()) ** 2
                    )
            if not gradient_finite:
                raise RuntimeError("non-finite gradient detected")
            pre_clip_norm = float(
                nn.utils.clip_grad_norm_(parameters, max_grad_norm).detach().cpu()
            )
            max_gradient_norm = max(max_gradient_norm, pre_clip_norm)
            max_head_gradient_norm = max(
                max_head_gradient_norm, math.sqrt(squared_head_norm)
            )
            was_clipped = pre_clip_norm > max_grad_norm
            clipped_step_count += int(was_clipped)
            epoch_clipped_steps += int(was_clipped)
            optimizer_step_count += 1
            epoch_optimizer_steps += 1
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        if not epoch_losses:
            raise RuntimeError("training epoch produced no losses")

        head.eval()
        validation_scores, score_variance = _score_cache(
            head,
            validation_cache,
            lane=lane,
            residual_weight=residual_weight,
            batch_size=batch_size,
            mode=mode,
        )
        validation_per_query = per_query_ranking_metrics(
            validation_scores, validation_cache.positive_mask
        )
        metrics = {
            name: float(values.mean().detach().cpu())
            for name, values in validation_per_query.items()
        }
        learned_weight = (
            float(residual_weight.detach().cpu())
            if residual_weight is not None
            else None
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "validation_metrics": metrics,
                "residual_weight": learned_weight,
                "correction_score_variance": score_variance,
                "clipped_step_count": epoch_clipped_steps,
                "optimizer_step_count": epoch_optimizer_steps,
                "clipped_step_rate": epoch_clipped_steps / epoch_optimizer_steps,
            }
        )
        if metrics["ndcg_at_10"] > best_ndcg + 1e-12:
            best_ndcg = metrics["ndcg_at_10"]
            best_epoch = epoch
            best_metrics = metrics
            best_residual_weight = learned_weight
            best_score_variance = score_variance
            best_per_query = {
                name: values.detach().cpu().numpy()
                for name, values in validation_per_query.items()
            }
            stale_epochs = 0
            torch.save(
                {
                    "schema_version": 1,
                    "lane": lane,
                    "seed": seed,
                    "epoch": epoch,
                    "embedding_dimension": embedding_dim,
                    "head_parameter_count": _head_parameter_count(head),
                    "head_state_dict": head.state_dict(),
                    "residual_weight": learned_weight,
                    "training_config": training_config,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    per_query_path = output_dir / "best_per_query_metrics.npz"
    np.savez_compressed(
        per_query_path,
        qids=np.asarray(validation_cache.qids, dtype=np.str_),
        baseline_ndcg_at_10=baseline_per_query["ndcg_at_10"].detach().cpu().numpy(),
        baseline_mrr_at_10=baseline_per_query["mrr_at_10"].detach().cpu().numpy(),
        baseline_recall_at_50=baseline_per_query["recall_at_50"].detach().cpu().numpy(),
        model_ndcg_at_10=best_per_query["ndcg_at_10"],
        model_mrr_at_10=best_per_query["mrr_at_10"],
        model_recall_at_50=best_per_query["recall_at_50"],
    )
    receipt = {
        "schema_version": 1,
        "status": "verified",
        "lane": lane,
        "seed": seed,
        "device": device,
        "source_tree_sha256": _source_fingerprint(),
        "embedding_dimension": embedding_dim,
        "head_parameter_count": _head_parameter_count(head),
        "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
        "train_cache_sha256": train_cache.sha256,
        "validation_cache_sha256": validation_cache.sha256,
        "test_cache_used": False,
        "train_positive_candidate_rate": float(
            train_cache.positive_mask.any(dim=1).float().mean().cpu()
        ),
        "validation_positive_candidate_rate": float(
            validation_cache.positive_mask.any(dim=1).float().mean().cpu()
        ),
        "baseline_validation_metrics": baseline_metrics,
        "best_validation_metrics": best_metrics,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_residual_weight": best_residual_weight,
        "best_correction_score_variance": best_score_variance,
        "gradient_finite": gradient_finite,
        "max_gradient_norm": max_gradient_norm,
        "max_head_gradient_norm": max_head_gradient_norm,
        "max_grad_norm": max_grad_norm,
        "clipped_step_count": clipped_step_count,
        "optimizer_step_count": optimizer_step_count,
        "clipped_step_rate": clipped_step_count / optimizer_step_count,
        "per_query_metrics_sha256": hashlib.sha256(
            per_query_path.read_bytes()
        ).hexdigest(),
        "per_query_raw_text_exported": False,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "history": history,
    }
    (output_dir / "training_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return receipt

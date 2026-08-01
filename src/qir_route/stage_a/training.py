from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from qir_route.baseline import resolve_device
from qir_route.quantum import AggregationMode, QuantumInspiredHead


@dataclass(frozen=True)
class CandidateCache:
    qids: tuple[str, ...]
    query_embeddings: Tensor
    document_embeddings: Tensor
    candidate_indices: Tensor
    candidate_scores: Tensor
    positive_mask: Tensor
    sha256: str


def load_candidate_cache(path: Path, device: str) -> CandidateCache:
    with np.load(path, allow_pickle=False) as arrays:
        if "qids" in arrays.files:
            qids = tuple(str(value) for value in arrays["qids"].tolist())
        else:
            qids = tuple(
                str(index) for index in range(arrays["query_embeddings"].shape[0])
            )
        query_embeddings = torch.from_numpy(arrays["query_embeddings"].copy()).to(
            device
        )
        document_embeddings = torch.from_numpy(arrays["document_embeddings"].copy()).to(
            device
        )
        candidate_indices = torch.from_numpy(arrays["candidate_indices"].copy()).to(
            device=device, dtype=torch.long
        )
        candidate_scores = torch.from_numpy(arrays["candidate_scores"].copy()).to(
            device
        )
        positive_mask = torch.from_numpy(arrays["positive_mask"].copy()).to(
            device=device, dtype=torch.bool
        )
    if candidate_indices.shape != positive_mask.shape:
        raise ValueError("candidate indices and positive mask must have the same shape")
    if query_embeddings.shape[0] != candidate_indices.shape[0]:
        raise ValueError("candidate cache has inconsistent query counts")
    return CandidateCache(
        qids=qids,
        query_embeddings=query_embeddings,
        document_embeddings=document_embeddings,
        candidate_indices=candidate_indices,
        candidate_scores=candidate_scores,
        positive_mask=positive_mask,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def multi_positive_listwise_loss(
    scores: Tensor,
    positive_mask: Tensor,
    temperature: float,
) -> Tensor:
    if scores.shape != positive_mask.shape or scores.ndim != 2:
        raise ValueError("scores and positive_mask must have the same 2D shape")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    valid = positive_mask.any(dim=1)
    if not bool(valid.any()):
        raise ValueError("the batch contains no positive candidates")
    logits = scores[valid] / temperature
    mask = positive_mask[valid]
    positive_logits = logits.masked_fill(~mask, -torch.inf)
    losses = torch.logsumexp(logits, dim=1) - torch.logsumexp(positive_logits, dim=1)
    return losses.mean()


def mean_ndcg_at_k(scores: Tensor, positive_mask: Tensor, k: int) -> float:
    if scores.shape != positive_mask.shape or scores.ndim != 2:
        raise ValueError("scores and positive_mask must have the same 2D shape")
    if k <= 0:
        raise ValueError("k must be positive")
    cutoff = min(k, scores.shape[1])
    order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :cutoff]
    gains = positive_mask.gather(1, order).to(scores.dtype)
    discounts = 1 / torch.log2(
        torch.arange(2, cutoff + 2, dtype=scores.dtype, device=scores.device)
    )
    dcg = (gains * discounts).sum(dim=1)
    positive_counts = positive_mask.sum(dim=1).clamp(max=cutoff)
    cumulative_discounts = discounts.cumsum(dim=0)
    safe_indices = (positive_counts - 1).clamp_min(0)
    ideal = cumulative_discounts[safe_indices]
    ideal = torch.where(positive_counts > 0, ideal, torch.ones_like(ideal))
    ndcg = torch.where(positive_counts > 0, dcg / ideal, torch.zeros_like(dcg))
    return float(ndcg.mean().detach().cpu())


def _score_cache(
    head: QuantumInspiredHead,
    cache: CandidateCache,
    *,
    batch_size: int,
    mode: AggregationMode,
) -> Tensor:
    score_batches: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, cache.query_embeddings.shape[0], batch_size):
            stop = min(start + batch_size, cache.query_embeddings.shape[0])
            indices = cache.candidate_indices[start:stop]
            documents = cache.document_embeddings[indices]
            score_batches.append(
                head.score(cache.query_embeddings[start:stop], documents, mode=mode)
            )
    return torch.cat(score_batches, dim=0)


def _source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def train_qi_head(
    train_cache_path: Path,
    validation_cache_path: Path,
    *,
    training_config: dict[str, Any],
    output_dir: Path,
    requested_device: str,
) -> dict[str, Any]:
    seed = int(training_config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device(requested_device)
    train_cache = load_candidate_cache(train_cache_path, device)
    validation_cache = load_candidate_cache(validation_cache_path, device)
    embedding_dim = int(train_cache.query_embeddings.shape[1])
    if validation_cache.query_embeddings.shape[1] != embedding_dim:
        raise ValueError("train and validation embedding dimensions differ")
    mode = cast(AggregationMode, str(training_config["aggregation"]))
    head = QuantumInspiredHead(embedding_dim=embedding_dim).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    batch_size = int(training_config["batch_size"])
    temperature = float(training_config["temperature"])
    max_epochs = int(training_config["max_epochs"])
    patience = int(training_config["patience"])
    if min(batch_size, max_epochs, patience) <= 0:
        raise ValueError("batch size, epochs, and patience must be positive")

    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_dir / "best_checkpoint.pt"
    history: list[dict[str, float | int]] = []
    best_ndcg = -math.inf
    best_epoch = 0
    stale_epochs = 0
    gradient_finite = True
    started = time.perf_counter()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train_count = train_cache.query_embeddings.shape[0]

    for epoch in range(1, max_epochs + 1):
        head.train()
        permutation = torch.randperm(train_count, generator=generator)
        epoch_losses: list[float] = []
        used_queries = 0
        for start in range(0, train_count, batch_size):
            cpu_indices = permutation[start : start + batch_size]
            indices = cpu_indices.to(device)
            positive_mask = train_cache.positive_mask[indices]
            valid = positive_mask.any(dim=1)
            if not bool(valid.any()):
                continue
            indices = indices[valid]
            positive_mask = positive_mask[valid]
            candidate_indices = train_cache.candidate_indices[indices]
            documents = train_cache.document_embeddings[candidate_indices]
            optimizer.zero_grad(set_to_none=True)
            scores = head.score(
                train_cache.query_embeddings[indices], documents, mode=mode
            )
            loss = multi_positive_listwise_loss(scores, positive_mask, temperature)
            loss.backward()
            for parameter in head.parameters():
                if parameter.grad is not None:
                    gradient_finite = gradient_finite and bool(
                        torch.isfinite(parameter.grad).all().cpu()
                    )
            if not gradient_finite:
                raise RuntimeError("non-finite gradient detected")
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            used_queries += int(indices.numel())

        head.eval()
        validation_scores = _score_cache(
            head,
            validation_cache,
            batch_size=batch_size,
            mode=mode,
        )
        validation_ndcg = mean_ndcg_at_k(
            validation_scores, validation_cache.positive_mask, 10
        )
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "train_queries_used": used_queries,
            "validation_ndcg_at_10": validation_ndcg,
        }
        history.append(record)
        if validation_ndcg > best_ndcg + 1e-12:
            best_ndcg = validation_ndcg
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "schema_version": 1,
                    "epoch": epoch,
                    "embedding_dimension": embedding_dim,
                    "parameter_count": head.quantum_parameter_count,
                    "model_state_dict": head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "training_config": training_config,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    base_validation_ndcg = mean_ndcg_at_k(
        validation_cache.candidate_scores,
        validation_cache.positive_mask,
        10,
    )
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    receipt = {
        "schema_version": 1,
        "status": "verified",
        "seed": seed,
        "device": device,
        "source_tree_sha256": _source_fingerprint(),
        "embedding_dimension": embedding_dim,
        "parameter_count": head.quantum_parameter_count,
        "aggregation": mode,
        "temperature": temperature,
        "train_cache_sha256": train_cache.sha256,
        "validation_cache_sha256": validation_cache.sha256,
        "test_cache_used": False,
        "train_query_count": int(train_count),
        "train_positive_candidate_rate": float(
            train_cache.positive_mask.any(dim=1).float().mean().cpu()
        ),
        "validation_query_count": int(validation_cache.query_embeddings.shape[0]),
        "validation_positive_candidate_rate": float(
            validation_cache.positive_mask.any(dim=1).float().mean().cpu()
        ),
        "base_validation_ndcg_at_10": base_validation_ndcg,
        "best_qi_validation_ndcg_at_10": best_ndcg,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "gradient_finite": gradient_finite,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_sha256": checkpoint_hash,
        "history": history,
    }
    (output_dir / "training_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt

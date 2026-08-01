from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qir_route.quantum.head import QuantumInspiredHead

BENCHMARK_SEED = 20260731


def _source_fingerprint() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _latency_summary(samples: list[float]) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def benchmark_head(
    *,
    device: str,
    candidates: int,
    batch_size: int,
    warmup: int,
    iterations: int,
    output: Path | None,
) -> dict[str, Any]:
    if min(candidates, batch_size, iterations) <= 0 or warmup < 0:
        raise ValueError("batch, candidates, and iterations must be positive")
    resolved_device = _resolve_device(device)
    torch.manual_seed(BENCHMARK_SEED)
    head = QuantumInspiredHead().to(resolved_device)
    queries = torch.randn(batch_size, 1024, device=resolved_device)
    documents = torch.randn(batch_size, candidates, 1024, device=resolved_device)

    for _ in range(warmup):
        with torch.no_grad():
            head.score(queries, documents, mode="mean")
    _synchronize(resolved_device)

    forward_samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        with torch.no_grad():
            head.score(queries, documents, mode="mean")
        _synchronize(resolved_device)
        forward_samples.append((time.perf_counter() - started) * 1000)

    training_samples: list[float] = []
    gradient_finite = True
    for _ in range(iterations):
        head.zero_grad(set_to_none=True)
        started = time.perf_counter()
        loss = -head.score(queries, documents, mode="clipped_mean_log").mean()
        loss.backward()
        _synchronize(resolved_device)
        training_samples.append((time.perf_counter() - started) * 1000)
        gradient_finite = gradient_finite and head.angles.grad is not None
        if head.angles.grad is not None:
            gradient_finite = gradient_finite and bool(
                torch.isfinite(head.angles.grad).all().cpu()
            )

    memory: dict[str, int | None] = {
        "allocated_bytes": None,
        "driver_allocated_bytes": None,
        "peak_allocated_bytes": None,
    }
    if resolved_device == "mps":
        memory["allocated_bytes"] = int(torch.mps.current_allocated_memory())
        memory["driver_allocated_bytes"] = int(torch.mps.driver_allocated_memory())
    elif resolved_device == "cuda":
        memory["allocated_bytes"] = int(torch.cuda.memory_allocated())
        memory["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "verified" if gradient_finite else "failed",
        "device": resolved_device,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "seed": BENCHMARK_SEED,
        "source_tree_sha256": _source_fingerprint(),
        "batch_size": batch_size,
        "candidates": candidates,
        "embedding_dimension": 1024,
        "group_count": head.group_count,
        "parameter_count": head.quantum_parameter_count,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "forward": _latency_summary(forward_samples),
        "forward_backward": _latency_summary(training_samples),
        "gradient_finite": gradient_finite,
        "memory": memory,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report

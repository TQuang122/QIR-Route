from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional

from qir_route.quantum.density import (
    density_matrix_reduce,
    squared_uhlmann_fidelity,
)

AggregationMode = Literal["mean", "mean_log", "clipped_mean_log"]


def scalar_to_density(values: Tensor, embedding_dim: int) -> Tensor:
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive")
    if values.dtype not in {torch.float32, torch.float64}:
        raise TypeError("values must use float32 or float64")
    bounded = torch.tanh(math.sqrt(embedding_dim) * values)
    polar_angle = 0.5 * math.pi * (bounded + 1)
    state = torch.stack(
        [torch.cos(0.5 * polar_angle), torch.sin(0.5 * polar_angle)], dim=-1
    )
    density = state.unsqueeze(-1) * state.unsqueeze(-2)
    complex_dtype = (
        torch.complex64 if values.dtype == torch.float32 else torch.complex128
    )
    return density.to(complex_dtype)


def aggregate_group_fidelities(
    fidelities: Tensor,
    mode: AggregationMode,
) -> Tensor:
    if fidelities.ndim == 0:
        raise ValueError("fidelities must include a group dimension")
    if mode == "mean":
        return fidelities.mean(dim=-1)
    if mode == "mean_log":
        return torch.log(fidelities.clamp_min(1e-12)).mean(dim=-1)
    if mode == "clipped_mean_log":
        return torch.log(fidelities.clamp_min(1e-4)).mean(dim=-1)
    raise ValueError(f"unsupported aggregation mode: {mode}")


class QuantumInspiredHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 1024,
        initialization_std: float = 0.02,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or embedding_dim % 4 != 0:
            raise ValueError("embedding_dim must be a positive multiple of 4")
        if initialization_std < 0:
            raise ValueError("initialization_std must be non-negative")
        self.embedding_dim = embedding_dim
        self.group_count = embedding_dim // 4
        self.initialization_std = initialization_std
        self.angles = nn.Parameter(torch.empty(self.group_count, 3, 9))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.angles, mean=0.0, std=self.initialization_std)

    @property
    def quantum_parameter_count(self) -> int:
        return int(self.angles.numel())

    def _validate_embeddings(self, embeddings: Tensor, name: str) -> None:
        if embeddings.ndim == 0 or embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(f"{name} must end with {self.embedding_dim} values")
        if not embeddings.is_floating_point():
            raise TypeError(f"{name} must be a floating-point tensor")
        if embeddings.device != self.angles.device:
            raise ValueError(f"{name} and the QI head must be on the same device")

    def forward(self, embeddings: Tensor) -> Tensor:
        self._validate_embeddings(embeddings, "embeddings")
        embeddings = embeddings.to(self.angles.dtype)
        normalized = functional.normalize(embeddings, p=2, dim=-1, eps=1e-12)
        scalar_densities = scalar_to_density(normalized, self.embedding_dim)
        grouped = scalar_densities.reshape(
            scalar_densities.shape[:-3] + (self.group_count, 4, 2, 2)
        )
        left = density_matrix_reduce(
            grouped[..., 0, :, :],
            grouped[..., 1, :, :],
            self.angles[:, 0, :],
        )
        right = density_matrix_reduce(
            grouped[..., 2, :, :],
            grouped[..., 3, :, :],
            self.angles[:, 1, :],
        )
        return density_matrix_reduce(left, right, self.angles[:, 2, :])

    def group_fidelities(
        self,
        query_embeddings: Tensor,
        document_embeddings: Tensor,
    ) -> Tensor:
        if document_embeddings.ndim == query_embeddings.ndim + 1:
            expected_prefix = query_embeddings.shape[:-1]
            if document_embeddings.shape[:-2] != expected_prefix:
                raise ValueError(
                    "document batch dimensions must match the query batch dimensions"
                )
        elif document_embeddings.ndim != query_embeddings.ndim:
            raise ValueError(
                "documents must be aligned with queries or add one candidate dimension"
            )
        query_densities = self(query_embeddings)
        document_densities = self(document_embeddings)
        if document_embeddings.ndim == query_embeddings.ndim + 1:
            query_densities = query_densities.unsqueeze(-4)
        return squared_uhlmann_fidelity(query_densities, document_densities)

    def score(
        self,
        query_embeddings: Tensor,
        document_embeddings: Tensor,
        mode: AggregationMode = "mean",
    ) -> Tensor:
        fidelities = self.group_fidelities(query_embeddings, document_embeddings)
        return aggregate_group_fidelities(fidelities, mode)

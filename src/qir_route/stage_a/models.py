from __future__ import annotations

import torch
from torch import Tensor, nn


class MatchedClassicalHead(nn.Module):
    def __init__(self, embedding_dim: int = 1024) -> None:
        super().__init__()
        if embedding_dim <= 0 or embedding_dim % 4 != 0:
            raise ValueError("embedding_dim must be a positive multiple of 4")
        self.embedding_dim = embedding_dim
        self.group_count = embedding_dim // 4
        self.weight = nn.Parameter(torch.empty(self.group_count, 3, 8))
        self.bias = nn.Parameter(torch.empty(self.group_count, 3))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.bias)

    @property
    def head_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def score(self, query_embeddings: Tensor, document_embeddings: Tensor) -> Tensor:
        if query_embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(f"queries must end with {self.embedding_dim} values")
        if document_embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(f"documents must end with {self.embedding_dim} values")
        if document_embeddings.ndim != query_embeddings.ndim + 1:
            raise ValueError("documents must add one candidate dimension")
        if document_embeddings.shape[:-2] != query_embeddings.shape[:-1]:
            raise ValueError("document batch dimensions must match queries")
        queries = query_embeddings.reshape(
            query_embeddings.shape[:-1] + (self.group_count, 4)
        ).unsqueeze(-3)
        documents = document_embeddings.reshape(
            document_embeddings.shape[:-1] + (self.group_count, 4)
        )
        features = torch.cat(
            [queries * documents, torch.abs(queries - documents)], dim=-1
        )
        hidden = torch.tanh(
            torch.einsum("...kgi,ghi->...kgh", features, self.weight) + self.bias
        )
        return hidden.mean(dim=(-1, -2))


def standardize_candidate_scores(scores: Tensor, epsilon: float = 1e-6) -> Tensor:
    if scores.ndim != 2:
        raise ValueError("candidate scores must be two-dimensional")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    centered = scores - scores.mean(dim=1, keepdim=True)
    scale = scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(epsilon)
    return centered / scale

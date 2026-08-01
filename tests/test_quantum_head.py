from typing import cast

import pytest
import torch

from qir_route.quantum import (
    AggregationMode,
    QuantumInspiredHead,
    aggregate_group_fidelities,
    scalar_to_density,
)


def assert_density_invariants(densities: torch.Tensor, tolerance: float) -> None:
    traces = torch.diagonal(densities, dim1=-2, dim2=-1).sum(dim=-1)
    eigenvalues = torch.linalg.eigvalsh(densities)
    assert torch.allclose(densities, densities.mH, atol=tolerance)
    assert torch.allclose(traces, torch.ones_like(traces), atol=tolerance)
    assert bool(torch.all(eigenvalues >= -tolerance))


def test_scalar_encoding_produces_pure_density_matrices() -> None:
    values = torch.linspace(-0.2, 0.2, 17, dtype=torch.float64)
    densities = scalar_to_density(values, embedding_dim=1024)
    assert densities.shape == (17, 2, 2)
    assert_density_invariants(densities, tolerance=1e-10)
    purity = torch.diagonal(densities @ densities, dim1=-2, dim2=-1).sum(-1).real
    assert torch.allclose(purity, torch.ones_like(purity), atol=1e-10)


def test_default_head_has_exact_parameter_ownership_and_count() -> None:
    head = QuantumInspiredHead()
    parameters = dict(head.named_parameters())
    assert set(parameters) == {"angles"}
    assert parameters["angles"].shape == (256, 3, 9)
    assert head.quantum_parameter_count == 6912
    assert sum(parameter.numel() for parameter in head.parameters()) == 6912


@pytest.mark.parametrize("leading_shape", [(), (3,), (2, 5)])
def test_head_output_shape_and_density_invariants(
    leading_shape: tuple[int, ...],
) -> None:
    torch.manual_seed(20260731)
    head = QuantumInspiredHead().double()
    embeddings = torch.randn(leading_shape + (1024,), dtype=torch.float64)
    densities = head(embeddings)
    assert densities.shape == leading_shape + (256, 2, 2)
    assert_density_invariants(densities, tolerance=1e-9)


def test_shared_head_scores_aligned_queries_and_topk_documents() -> None:
    torch.manual_seed(20260731)
    head = QuantumInspiredHead()
    queries = torch.randn(2, 1024)
    documents = torch.randn(2, 7, 1024)
    documents[:, 0] = queries
    group_scores = head.group_fidelities(queries, documents)
    assert group_scores.shape == (2, 7, 256)
    assert torch.allclose(group_scores[:, 0], torch.ones(2, 256), atol=1e-5)
    modes: tuple[AggregationMode, ...] = (
        "mean",
        "mean_log",
        "clipped_mean_log",
    )
    for mode in modes:
        scores = head.score(queries, documents, mode=mode)
        assert scores.shape == (2, 7)
        assert bool(torch.isfinite(scores).all())


def test_aggregation_modes_are_stable_at_zero_fidelity() -> None:
    fidelities = torch.tensor([[0.0, 0.5, 1.0]], requires_grad=True)
    outputs = [
        aggregate_group_fidelities(fidelities, "mean"),
        aggregate_group_fidelities(fidelities, "mean_log"),
        aggregate_group_fidelities(fidelities, "clipped_mean_log"),
    ]
    torch.stack([output.sum() for output in outputs]).sum().backward()
    assert all(bool(torch.isfinite(output).all()) for output in outputs)
    assert fidelities.grad is not None
    assert bool(torch.isfinite(fidelities.grad).all())


def test_head_forward_and_backward_are_finite() -> None:
    torch.manual_seed(20260731)
    head = QuantumInspiredHead()
    queries = torch.randn(2, 1024, requires_grad=True)
    documents = torch.randn(2, 3, 1024, requires_grad=True)
    loss = -head.score(queries, documents, mode="clipped_mean_log").mean()
    loss.backward()
    assert head.angles.grad is not None
    assert queries.grad is not None
    assert documents.grad is not None
    assert bool(torch.isfinite(head.angles.grad).all())
    assert bool(torch.isfinite(queries.grad).all())
    assert bool(torch.isfinite(documents.grad).all())


def test_head_rejects_invalid_shapes_and_aggregation() -> None:
    head = QuantumInspiredHead()
    with pytest.raises(ValueError, match="1024 values"):
        head(torch.randn(4, 768))
    with pytest.raises(ValueError, match="candidate dimension"):
        head.score(torch.randn(2, 1024), torch.randn(2, 3, 4, 1024))
    with pytest.raises(ValueError, match="unsupported aggregation"):
        head.score(
            torch.randn(2, 1024),
            torch.randn(2, 1024),
            mode=cast(AggregationMode, "median"),
        )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_head_forward_and_backward_on_mps() -> None:
    torch.manual_seed(20260731)
    head = QuantumInspiredHead().to("mps")
    queries = torch.randn(1, 1024, device="mps")
    documents = torch.randn(1, 2, 1024, device="mps")
    loss = head.score(queries, documents, mode="mean").sum()
    loss.backward()
    assert head.angles.grad is not None
    assert bool(torch.isfinite(head.angles.grad).all().cpu())

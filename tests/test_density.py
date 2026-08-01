import math

import pytest
import torch

from qir_route.quantum import (
    controlled_unitary,
    density_matrix_reduce,
    squared_uhlmann_fidelity,
    zyz_unitary,
)


def pure_density(state: torch.Tensor) -> torch.Tensor:
    normalized = state / torch.linalg.vector_norm(state, dim=-1, keepdim=True)
    return normalized.unsqueeze(-1) @ normalized.conj().unsqueeze(-2)


def random_density(batch_size: int, dtype: torch.dtype) -> torch.Tensor:
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    real = torch.randn(batch_size, 2, 2, dtype=real_dtype)
    imaginary = torch.randn(batch_size, 2, 2, dtype=real_dtype)
    matrix = torch.complex(real, imaginary).to(dtype)
    density = matrix @ matrix.mH
    trace = torch.diagonal(density, dim1=-2, dim2=-1).sum(dim=-1).real
    return density / trace[..., None, None]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zyz_and_controlled_matrices_are_unitary(dtype: torch.dtype) -> None:
    angles = torch.randn(7, 3, dtype=dtype)
    unitary = zyz_unitary(angles)
    controlled = controlled_unitary(unitary)
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    identity_two = torch.eye(2, dtype=complex_dtype).expand(7, 2, 2)
    identity_four = torch.eye(4, dtype=complex_dtype).expand(7, 4, 4)
    tolerance = 1e-5 if dtype == torch.float32 else 1e-10
    assert torch.allclose(unitary.mH @ unitary, identity_two, atol=tolerance)
    assert torch.allclose(controlled.mH @ controlled, identity_four, atol=tolerance)


def test_identity_reduction_returns_first_density() -> None:
    rho_a = random_density(5, torch.complex128)
    rho_b = random_density(5, torch.complex128)
    angles = torch.zeros(5, 9, dtype=torch.float64)
    reduced = density_matrix_reduce(rho_a, rho_b, angles)
    assert torch.allclose(reduced, rho_a, atol=1e-10)


def test_entangling_reduction_returns_mixed_density() -> None:
    complex_dtype = torch.complex128
    plus = torch.tensor([1.0, 1.0], dtype=complex_dtype)
    zero = torch.tensor([1.0, 0.0], dtype=complex_dtype)
    rho_plus = pure_density(plus)
    rho_zero = pure_density(zero)
    angles = torch.zeros(9, dtype=torch.float64)
    angles[7] = math.pi
    reduced = density_matrix_reduce(rho_plus, rho_zero, angles)
    expected = torch.eye(2, dtype=complex_dtype) / 2
    assert torch.allclose(reduced, expected, atol=1e-10)
    assert torch.allclose(torch.linalg.matrix_power(reduced, 2), expected / 2)


def test_reduction_preserves_density_invariants_for_batches() -> None:
    rho_a = random_density(11, torch.complex128)
    rho_b = random_density(11, torch.complex128)
    angles = torch.randn(11, 9, dtype=torch.float64)
    reduced = density_matrix_reduce(rho_a, rho_b, angles)
    traces = torch.diagonal(reduced, dim1=-2, dim2=-1).sum(dim=-1)
    eigenvalues = torch.linalg.eigvalsh(reduced)
    assert reduced.shape == (11, 2, 2)
    assert torch.allclose(reduced, reduced.mH, atol=1e-10)
    assert torch.allclose(traces, torch.ones_like(traces), atol=1e-10)
    assert bool(torch.all(eigenvalues >= -1e-10))


def test_squared_uhlmann_fidelity_matches_pure_state_overlap() -> None:
    complex_dtype = torch.complex128
    zero = torch.tensor([1.0, 0.0], dtype=complex_dtype)
    one = torch.tensor([0.0, 1.0], dtype=complex_dtype)
    plus = torch.tensor([1.0, 1.0], dtype=complex_dtype)
    rho_zero = pure_density(zero)
    rho_one = pure_density(one)
    rho_plus = pure_density(plus)
    one_scalar = torch.tensor(1.0, dtype=torch.float64)
    zero_scalar = torch.tensor(0.0, dtype=torch.float64)
    assert torch.allclose(squared_uhlmann_fidelity(rho_zero, rho_zero), one_scalar)
    assert torch.allclose(squared_uhlmann_fidelity(rho_zero, rho_one), zero_scalar)
    expected = torch.abs(torch.vdot(zero, plus / torch.linalg.vector_norm(plus))) ** 2
    observed = squared_uhlmann_fidelity(rho_zero, rho_plus)
    assert torch.allclose(observed, expected)
    assert torch.allclose(observed, squared_uhlmann_fidelity(rho_plus, rho_zero))


def test_orthogonal_pure_state_fidelity_has_finite_gradient() -> None:
    rho_zero = torch.tensor(
        [[1.0, 0.0], [0.0, 0.0]], dtype=torch.complex128, requires_grad=True
    )
    rho_one = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.complex128)
    fidelity = squared_uhlmann_fidelity(rho_zero, rho_one)
    fidelity.backward()
    assert fidelity == 0
    assert rho_zero.grad is not None
    assert bool(torch.isfinite(rho_zero.grad).all())


def test_reduction_and_fidelity_pass_gradcheck() -> None:
    rho_a = torch.diag(torch.tensor([0.7, 0.3], dtype=torch.complex128))
    rho_b = torch.diag(torch.tensor([0.4, 0.6], dtype=torch.complex128))
    target = torch.diag(torch.tensor([0.55, 0.45], dtype=torch.complex128))
    angles = torch.randn(9, dtype=torch.float64, requires_grad=True) * 0.1

    def score(parameters: torch.Tensor) -> torch.Tensor:
        reduced = density_matrix_reduce(rho_a, rho_b, parameters)
        return squared_uhlmann_fidelity(reduced, target)

    assert torch.autograd.gradcheck(score, (angles,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_invalid_shapes_fail_at_the_api_boundary() -> None:
    density = torch.eye(2, dtype=torch.complex64) / 2
    with pytest.raises(ValueError, match="9 values"):
        density_matrix_reduce(density, density, torch.zeros(8))
    with pytest.raises(ValueError, match=r"\[2, 2\]"):
        squared_uhlmann_fidelity(torch.eye(3), density)
    with pytest.raises(TypeError, match="float32 or float64"):
        zyz_unitary(torch.zeros(3, dtype=torch.float16))

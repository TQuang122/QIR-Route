from __future__ import annotations

import torch
from torch import Tensor


def _validate_angles(angles: Tensor, expected: int) -> None:
    if angles.ndim == 0 or angles.shape[-1] != expected:
        raise ValueError(f"angles must end with {expected} values")
    if angles.dtype not in {torch.float32, torch.float64}:
        raise TypeError("angles must use float32 or float64")


def _validate_qubit_matrix(matrix: Tensor, name: str) -> None:
    if matrix.ndim < 2 or matrix.shape[-2:] != (2, 2):
        raise ValueError(f"{name} must end with shape [2, 2]")


def _complex_dtype(real_dtype: torch.dtype) -> torch.dtype:
    return torch.complex64 if real_dtype == torch.float32 else torch.complex128


def _batched_kronecker(left: Tensor, right: Tensor) -> Tensor:
    product = torch.einsum("...ij,...kl->...ikjl", left, right)
    rows = left.shape[-2] * right.shape[-2]
    columns = left.shape[-1] * right.shape[-1]
    return product.reshape(product.shape[:-4] + (rows, columns))


def _qubit_determinant(matrix: Tensor) -> Tensor:
    return matrix[..., 0, 0] * matrix[..., 1, 1] - matrix[..., 0, 1] * matrix[..., 1, 0]


def zyz_unitary(angles: Tensor) -> Tensor:
    _validate_angles(angles, 3)
    alpha, beta, gamma = angles.unbind(dim=-1)
    complex_dtype = _complex_dtype(angles.dtype)

    alpha_phases = torch.stack(
        [torch.exp(-0.5j * alpha), torch.exp(0.5j * alpha)], dim=-1
    )
    gamma_phases = torch.stack(
        [torch.exp(-0.5j * gamma), torch.exp(0.5j * gamma)], dim=-1
    )
    rz_alpha = torch.diag_embed(alpha_phases).to(complex_dtype)
    rz_gamma = torch.diag_embed(gamma_phases).to(complex_dtype)

    cosine = torch.cos(0.5 * beta)
    sine = torch.sin(0.5 * beta)
    ry_beta = torch.stack(
        [
            torch.stack([cosine, -sine], dim=-1),
            torch.stack([sine, cosine], dim=-1),
        ],
        dim=-2,
    ).to(complex_dtype)
    return rz_gamma @ ry_beta @ rz_alpha


def controlled_unitary(target_unitary: Tensor) -> Tensor:
    _validate_qubit_matrix(target_unitary, "target_unitary")
    batch_shape = target_unitary.shape[:-2]
    identity = torch.eye(
        2, dtype=target_unitary.dtype, device=target_unitary.device
    ).expand(batch_shape + (2, 2))
    zeros = torch.zeros_like(identity)
    top = torch.cat([identity, zeros], dim=-1)
    bottom = torch.cat([zeros, target_unitary], dim=-1)
    return torch.cat([top, bottom], dim=-2)


def density_matrix_reduce(rho_a: Tensor, rho_b: Tensor, angles: Tensor) -> Tensor:
    _validate_qubit_matrix(rho_a, "rho_a")
    _validate_qubit_matrix(rho_b, "rho_b")
    _validate_angles(angles, 9)

    complex_dtype = _complex_dtype(angles.dtype)
    rho_a = rho_a.to(dtype=complex_dtype, device=angles.device)
    rho_b = rho_b.to(dtype=complex_dtype, device=angles.device)
    local_a = zyz_unitary(angles[..., 0:3])
    local_b = zyz_unitary(angles[..., 3:6])
    controlled = controlled_unitary(zyz_unitary(angles[..., 6:9]))
    joint_unitary = controlled @ _batched_kronecker(local_a, local_b)
    joint_density = _batched_kronecker(rho_a, rho_b)
    evolved = joint_unitary @ joint_density @ joint_unitary.mH
    tensor_form = evolved.reshape(evolved.shape[:-2] + (2, 2, 2, 2))
    return tensor_form.diagonal(dim1=-3, dim2=-1).sum(dim=-1)


def squared_uhlmann_fidelity(rho: Tensor, sigma: Tensor) -> Tensor:
    _validate_qubit_matrix(rho, "rho")
    _validate_qubit_matrix(sigma, "sigma")
    promoted_dtype = torch.promote_types(rho.dtype, sigma.dtype)
    if promoted_dtype not in {torch.complex64, torch.complex128}:
        promoted_dtype = (
            torch.complex64
            if promoted_dtype in {torch.float16, torch.float32}
            else torch.complex128
        )
    rho = rho.to(promoted_dtype)
    sigma = sigma.to(device=rho.device, dtype=promoted_dtype)
    overlap = torch.diagonal(rho @ sigma, dim1=-2, dim2=-1).sum(dim=-1).real
    det_rho = _qubit_determinant(rho).real.clamp_min(0)
    det_sigma = _qubit_determinant(sigma).real.clamp_min(0)
    determinant_product = det_rho * det_sigma
    threshold = 1e-12 if overlap.dtype == torch.float32 else 1e-24
    stable_root = torch.sqrt(determinant_product.clamp_min(threshold))
    stable_root = torch.where(
        determinant_product > threshold,
        stable_root,
        torch.zeros_like(stable_root),
    )
    fidelity = overlap + 2 * stable_root
    return fidelity.clamp(0, 1)

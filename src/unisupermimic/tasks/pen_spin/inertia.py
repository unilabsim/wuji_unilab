# SPDX-License-Identifier: Apache-2.0
# Pseudo-inertia helpers from mujocolab/mjlab v1.6.0,
# commit 0fb8a681136be94ffc636a3dd423cabb97d91f10, envs/mdp/dr/body.py.
# Original math is kept; only quaternion imports bind to the local wxyz API.
"""Source pseudo-inertia math, isolated from the UniLab model interface."""

import torch

from unisupermimic.rl.algos.dex_supermimic.rotations import (
  mat_to_quat as quat_from_matrix,
)
from unisupermimic.rl.algos.dex_supermimic.rotations import (
  quat_to_mat as matrix_from_quat,
)

_JACOBI_SWEEPS = 5

def _cholesky_4x4(A: torch.Tensor) -> torch.Tensor:
  """Analytical Cholesky for batched 4x4 SPD matrices.

  Avoids ``torch.linalg.cholesky`` (and the cuSOLVER library it loads), which allocates
  several GB of persistent GPU memory on first use.

  Args:
    A: ``(*batch, 4, 4)`` symmetric positive-definite matrix.

  Returns:
    L: ``(*batch, 4, 4)`` lower-triangular Cholesky factor.
  """
  L = torch.zeros_like(A)
  L[..., 0, 0] = torch.sqrt(A[..., 0, 0])
  L[..., 1, 0] = A[..., 1, 0] / L[..., 0, 0]
  L[..., 2, 0] = A[..., 2, 0] / L[..., 0, 0]
  L[..., 3, 0] = A[..., 3, 0] / L[..., 0, 0]
  L[..., 1, 1] = torch.sqrt(A[..., 1, 1] - L[..., 1, 0] ** 2)
  L[..., 2, 1] = (A[..., 2, 1] - L[..., 2, 0] * L[..., 1, 0]) / L[..., 1, 1]
  L[..., 3, 1] = (A[..., 3, 1] - L[..., 3, 0] * L[..., 1, 0]) / L[..., 1, 1]
  L[..., 2, 2] = torch.sqrt(A[..., 2, 2] - L[..., 2, 0] ** 2 - L[..., 2, 1] ** 2)
  L[..., 3, 2] = (
    A[..., 3, 2] - L[..., 3, 0] * L[..., 2, 0] - L[..., 3, 1] * L[..., 2, 1]
  ) / L[..., 2, 2]
  L[..., 3, 3] = torch.sqrt(
    A[..., 3, 3] - L[..., 3, 0] ** 2 - L[..., 3, 1] ** 2 - L[..., 3, 2] ** 2
  )
  return L

def _eigh_3x3_jacobi(
  A: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Batched eigendecomposition for 3x3 symmetric matrices via Jacobi.

  Avoids ``torch.linalg.eigh`` (and the cuSOLVER library it loads), which allocates
  several GB of persistent GPU memory on first use.

  Uses cyclic Jacobi rotations over the three off-diagonal pairs. For 3x3 matrices,
  ``_JACOBI_SWEEPS`` sweeps converge to machine precision.

  Args:
    A: ``(*batch, 3, 3)`` symmetric matrix.

  Returns:
    eigenvalues: ``(*batch, 3)`` in ascending order.
    V: ``(*batch, 3, 3)`` orthogonal eigenvectors (columns).
  """
  D = A.clone()
  V = torch.eye(3, device=A.device, dtype=A.dtype).expand_as(D).clone()

  for _ in range(_JACOBI_SWEEPS):
    for p, q in ((0, 1), (0, 2), (1, 2)):
      r = 3 - p - q
      apq = D[..., p, q]
      app = D[..., p, p]
      aqq = D[..., q, q]

      # Jacobi rotation: tau = (aqq - app) / (2*apq),
      # t = sign(tau) / (|tau| + sqrt(1 + tau^2)).
      diff = aqq - app
      denom = 2 * apq
      tau = diff / denom
      t = torch.sign(tau) / (torch.abs(tau) + torch.sqrt(1 + tau * tau))
      # When apq ≈ 0, skip rotation (t = 0).
      t = torch.where(torch.abs(denom) > 1e-30, t, torch.zeros_like(t))

      c = 1 / torch.sqrt(1 + t * t)
      s = t * c

      # Update diagonal and off-diagonal elements.
      D[..., p, p] = app - t * apq
      D[..., q, q] = aqq + t * apq
      D[..., p, q] = D[..., q, p] = 0

      arp = D[..., r, p].clone()
      arq = D[..., r, q].clone()
      D[..., r, p] = D[..., p, r] = c * arp - s * arq
      D[..., r, q] = D[..., q, r] = s * arp + c * arq

      # Accumulate eigenvectors: V[:, p/q] = c*Vp ∓ s*Vq.
      vp = V[..., :, p].clone()
      vq = V[..., :, q].clone()
      c_exp = c.unsqueeze(-1)
      s_exp = s.unsqueeze(-1)
      V[..., :, p] = c_exp * vp - s_exp * vq
      V[..., :, q] = s_exp * vp + c_exp * vq

  eigenvalues = torch.stack([D[..., 0, 0], D[..., 1, 1], D[..., 2, 2]], dim=-1)

  # Sort in ascending order to match torch.linalg.eigh convention.
  idx = eigenvalues.argsort(dim=-1)
  eigenvalues = eigenvalues.gather(-1, idx)
  V = V.gather(-1, idx.unsqueeze(-2).expand_as(V))

  return eigenvalues, V

def _reconstruct_pseudo_inertia_J(
  mass: torch.Tensor,
  ipos: torch.Tensor,
  inertia: torch.Tensor,
  iquat: torch.Tensor,
) -> torch.Tensor:
  """Build the 4x4 pseudo-inertia matrix J from MuJoCo body fields.

  1. Rotate principal moments into body frame via ``body_iquat``.
  2. Apply the parallel-axis theorem to shift inertia from COM to body origin.

  Args:
    mass: ``(*batch,)``.
    ipos: COM in body frame, ``(*batch, 3)``.
    inertia: Principal moments, ``(*batch, 3)``.
    iquat: Principal-to-body quaternion (wxyz), ``(*batch, 4)``.

  Returns:
    J: ``(*batch, 4, 4)``.
  """
  I3 = torch.eye(3, device=mass.device, dtype=mass.dtype)

  # Rotate principal moments into body frame (body_iquat maps principal->body).
  R = matrix_from_quat(iquat)  # (*batch, 3, 3)
  I_com = R @ torch.diag_embed(inertia) @ R.mT  # (*batch, 3, 3)

  # Parallel-axis theorem: shift inertia from COM to body origin.
  c = ipos
  c_sq = (c * c).sum(dim=-1)  # (*batch,)
  c_outer = c.unsqueeze(-1) * c.unsqueeze(-2)  # (*batch, 3, 3)
  m = mass.unsqueeze(-1).unsqueeze(-1)  # (*batch, 1, 1)
  I_origin = I_com + m * (
    c_sq.unsqueeze(-1).unsqueeze(-1) * I3 - c_outer
  )  # (*batch, 3, 3)

  # sigma = 0.5 * Tr(I_origin) * I3 - I_origin
  trace = I_origin.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # (*batch,)
  sigma = 0.5 * trace.unsqueeze(-1).unsqueeze(-1) * I3 - I_origin

  h = mass.unsqueeze(-1) * ipos  # first mass moment

  batch_shape = mass.shape
  J = torch.zeros(*batch_shape, 4, 4, device=mass.device, dtype=mass.dtype)
  J[..., :3, :3] = sigma
  J[..., :3, 3] = h
  J[..., 3, :3] = h
  J[..., 3, 3] = mass
  return J

def _decompose_pseudo_inertia_J(
  J: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Decompose pseudo-inertia matrix to MuJoCo body fields (exact).

  Extracts ``body_mass``, ``body_ipos``, ``body_inertia`` (principal moments), and
  ``body_iquat`` (principal-frame rotation) by diagonalizing the full inertia tensor
  via eigendecomposition. This is exact for any perturbation magnitude, including large
  shear.

  Args:
    J: 4x4 pseudo-inertia matrix, shape ``(*batch, 4, 4)``.

  Returns:
    Tuple of (mass, ipos, inertia, iquat) with shapes ``(*batch,)``,
    ``(*batch, 3)``, ``(*batch, 3)``, ``(*batch, 4)``.
  """
  mass = J[..., 3, 3]
  h = J[..., :3, 3]
  ipos = h / mass.unsqueeze(-1)

  sigma = J[..., :3, :3]
  trace_sigma = sigma.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # (*batch,)
  I3 = torch.eye(3, device=J.device, dtype=J.dtype)
  # Invert sigma = 0.5*Tr(I)*I3 - I  =>  I_origin = Tr(sigma)*I3 - sigma
  I_origin = trace_sigma.unsqueeze(-1).unsqueeze(-1) * I3 - sigma  # (*batch, 3, 3)

  # Inverse parallel-axis: shift inertia from body origin back to COM.
  c = ipos
  c_sq = (c * c).sum(dim=-1)  # (*batch,)
  c_outer = c.unsqueeze(-1) * c.unsqueeze(-2)  # (*batch, 3, 3)
  m = mass.unsqueeze(-1).unsqueeze(-1)  # (*batch, 1, 1)
  I_com = I_origin - m * (
    c_sq.unsqueeze(-1).unsqueeze(-1) * I3 - c_outer
  )  # (*batch, 3, 3)

  # Columns of V are principal axes in body frame; eigenvalues are principal moments.
  principal_moments, V = _eigh_3x3_jacobi(I_com)

  # Ensure V is a proper rotation (det = +1). eigh can return reflections.
  dets = torch.linalg.det(V)  # (*batch,)
  neg = dets < 0
  if torch.any(neg):
    V = V.clone()
    V[neg, :, 2] *= -1

  # MuJoCo body_iquat is principal->body, i.e. it represents R = V.
  iquat = quat_from_matrix(V)  # (*batch, 4), wxyz

  return mass, ipos, principal_moments, iquat

def _build_perturbation_U(
  alpha: torch.Tensor,
  d1: torch.Tensor,
  d2: torch.Tensor,
  d3: torch.Tensor,
  s12: torch.Tensor,
  s13: torch.Tensor,
  s23: torch.Tensor,
  t1: torch.Tensor,
  t2: torch.Tensor,
  t3: torch.Tensor,
) -> torch.Tensor:
  """Build the upper-triangular perturbation matrix U from 10 parameters.

  .. code-block::

      U = e^alpha * [[e^d1, s12, s13, t1],
                  [0,   e^d2, s23, t2],
                  [0,   0,   e^d3, t3],
                  [0,   0,   0,    1 ]]

  All arguments have shape ``(*batch,)``.

  Returns:
    U: shape ``(*batch, 4, 4)``.
  """
  scale = torch.exp(alpha)  # (*batch,)
  batch_shape = alpha.shape
  U = torch.zeros(*batch_shape, 4, 4, device=alpha.device, dtype=alpha.dtype)
  U[..., 0, 0] = scale * torch.exp(d1)
  U[..., 0, 1] = scale * s12
  U[..., 0, 2] = scale * s13
  U[..., 0, 3] = scale * t1
  U[..., 1, 1] = scale * torch.exp(d2)
  U[..., 1, 2] = scale * s23
  U[..., 1, 3] = scale * t2
  U[..., 2, 2] = scale * torch.exp(d3)
  U[..., 2, 3] = scale * t3
  U[..., 3, 3] = scale  # e^alpha × 1
  return U

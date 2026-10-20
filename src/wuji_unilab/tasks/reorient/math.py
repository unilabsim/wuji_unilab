"""Batched wxyz rotation math for Wuji observation and control frames."""

from __future__ import annotations

import numpy as np
import torch


def conjugate(q):
    return q * np.array([1, -1, -1, -1], dtype=np.float32)


def multiply(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def rotate(q, v):
    cross = np.cross(q[..., 1:], v)
    return v + 2 * (q[..., :1] * cross + np.cross(q[..., 1:], cross))


def random_quaternions(rng, count):
    return torch_random_quaternions(rng, count).cpu().numpy()


def torch_random_quaternions(rng, count: int, device=None) -> torch.Tensor:
    q = torch.as_tensor(
        rng.normal(0.0, 1.0, (count, 4)), dtype=torch.float32, device=device
    )
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)


def rotation6d(q):
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        (
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    )


def angle_error(a, b):
    return 2 * np.arccos(np.clip(np.abs(np.sum(a * b, axis=-1)), 0, 1))


def soft_limits(hard, factor):
    center = hard.mean(axis=-1)
    half = (hard[:, 1] - hard[:, 0]) * 0.5 * factor
    return np.stack((center - half, center + half), axis=-1)


def torch_conjugate(q: torch.Tensor) -> torch.Tensor:
    return q * torch.tensor((1.0, -1.0, -1.0, -1.0), device=q.device, dtype=q.dtype)


def torch_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def torch_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qvec = q[..., 1:]
    cross1 = torch.linalg.cross(qvec, v, dim=-1)
    cross2 = torch.linalg.cross(qvec, cross1, dim=-1)
    return v + 2 * (q[..., :1] * cross1 + cross2)


def torch_rotation6d(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        (
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    )


def torch_angle_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 2 * torch.arccos(torch.clamp(torch.abs(torch.sum(a * b, dim=-1)), 0.0, 1.0))


def torch_soft_limits(hard: torch.Tensor, factor: float) -> torch.Tensor:
    center = hard.mean(dim=-1)
    half = (hard[:, 1] - hard[:, 0]) * 0.5 * factor
    return torch.stack((center - half, center + half), dim=-1)

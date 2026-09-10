"""Batched wxyz rotation math for Wuji observation and control frames."""

from __future__ import annotations

import numpy as np


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
    q = rng.normal(size=(count, 4))
    return (q / np.linalg.norm(q, axis=-1, keepdims=True)).astype(np.float32)


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

"""Hand 2 analytic tensor contract used by the SuperMimic adapter.

The environment-side terms are intentionally kept separate from this module:
this file contains only shape-stable, differentiable operations that can also
be used by an imagined rollout.
"""

from __future__ import annotations

import torch

from unisupermimic.rl.algos.supermimic.geometry import matmul, so3_exp

NUM_JOINTS = 20
STATE_DIM = 58
CONTEXT_DIM = 46
OBS_DIM = 69
GOAL_FRAME_DIM = 55
CONTACT_DIM = 32


def pack_state(
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    position: torch.Tensor,
    rotation: torch.Tensor,
    linear_velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
) -> torch.Tensor:
    """Pack ``q, qd, p, R, v, w`` into the source Hand 2 state layout."""
    if joint_pos.shape[-1] != NUM_JOINTS or joint_vel.shape[-1] != NUM_JOINTS:
        raise ValueError("Hand2 state expects 20 joint positions and velocities")
    if rotation.shape[-2:] != (3, 3):
        raise ValueError("Hand2 rotation must have shape (..., 3, 3)")
    return torch.cat(
        (joint_pos, joint_vel, position, rotation.reshape(*rotation.shape[:-2], 9), linear_velocity, angular_velocity),
        dim=-1,
    )


def unpack_state(value: torch.Tensor):
    """Inverse of :func:`pack_state`, returning differentiable tensors."""
    if value.shape[-1] != STATE_DIM:
        raise ValueError(f"Hand2 state must have width {STATE_DIM}, got {value.shape[-1]}")
    return (
        value[..., :20],
        value[..., 20:40],
        value[..., 40:43],
        value[..., 43:52].reshape(*value.shape[:-1], 3, 3),
        value[..., 52:55],
        value[..., 55:58],
    )


def integrate(state: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Integrate the 52D residual predicted by the world model."""
    if delta.shape[-1] != 52:
        raise ValueError("Hand2 dynamics residual must have width 52")
    q, qd, p, rotation, v, w = unpack_state(state)
    dq, dqd, dp, dtheta, dv, dw = delta.split((20, 20, 3, 3, 3, 3), dim=-1)
    next_rotation = matmul(rotation, so3_exp(dtheta))
    return pack_state(q + dq, qd + dqd, p + dp, next_rotation, v + dv, w + dw)


def ema_target(
    default_joint_pos: torch.Tensor,
    action: torch.Tensor,
    previous_target: torch.Tensor,
    *,
    action_scale: float = 0.5,
    alpha: float = 0.5,
    lower: torch.Tensor | None = None,
    upper: torch.Tensor | None = None,
    age: torch.Tensor | None = None,
    warmup_steps: int = 0,
) -> torch.Tensor:
    """Apply the source position offset + EMA controller in batch form."""
    if action.shape[-1] != NUM_JOINTS:
        raise ValueError("Hand2 actions must have width 20")
    raw = default_joint_pos + action.clamp(-1.0, 1.0) * action_scale
    if lower is not None and upper is not None:
        raw = torch.clamp(raw, lower, upper)
    target = alpha * raw + (1.0 - alpha) * previous_target
    if age is not None and warmup_steps > 0:
        target = torch.where(age.reshape(-1, 1) < warmup_steps, default_joint_pos, target)
    return target


def observation(
    joint_angles: torch.Tensor,
    qpos_error: torch.Tensor,
    cube_position: torch.Tensor,
    cube_rotation_error: torch.Tensor,
    previous_action: torch.Tensor,
) -> torch.Tensor:
    """Build the 69D actor observation in source order."""
    values = (joint_angles, qpos_error, cube_position, cube_rotation_error, previous_action)
    if cube_rotation_error.shape[-1] != 6:
        raise ValueError("Hand2 rotation error must use a 6D representation")
    out = torch.cat(values, dim=-1)
    if out.shape[-1] != OBS_DIM:
        raise ValueError(f"Hand2 observation must have width {OBS_DIM}, got {out.shape[-1]}")
    return out


__all__ = [
    "CONTACT_DIM", "CONTEXT_DIM", "GOAL_FRAME_DIM", "NUM_JOINTS", "OBS_DIM", "STATE_DIM",
    "ema_target", "integrate", "observation", "pack_state", "unpack_state",
]

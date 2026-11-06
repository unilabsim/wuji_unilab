"""Measured source Pen Spin tracking and smoothness metrics."""

from __future__ import annotations

import torch
from unilab.managers import ManagerTermBase

from .commands import task
from .observations import _runtime_pose
from .rewards import (
    _axis_from_pose,
    _metadata,
    _sensor_values,
    current_actuator_limits,
    normalized_joint_torques_l2,
)


def object_pos_tracking_error_cm(env) -> torch.Tensor:
    position, _ = _runtime_pose(env, task(env))
    return torch.linalg.vector_norm(position - task(env).object_pose[:, :3], dim=-1) * 100.0


def object_ori_tracking_error_deg(env) -> torch.Tensor:
    _, current_quat = _runtime_pose(env, task(env))
    axis = torch.tensor((1.0, 0.0, 0.0), device=current_quat.device, dtype=current_quat.dtype)
    current = _axis_from_pose(current_quat, axis)
    target = _axis_from_pose(task(env).object_pose[:, 3:], axis)
    return torch.rad2deg(torch.acos((current * target).sum(-1).clamp(-1.0, 1.0)))


def action_rate_l2(env) -> torch.Tensor:
    action = env.action_manager
    return (action.action - action.prev_action).square().sum(-1)


def action_acc_l2(env) -> torch.Tensor:
    action = env.action_manager
    return (action.action - 2.0 * action.prev_action + action.prev_prev_action).square().sum(-1)


def joint_vel_rms(env) -> torch.Tensor:
    return env.scene.read_plan.joint_tensor_view("robot").joint_vel.square().mean(-1).sqrt()


class joint_acc_rms(ManagerTermBase):
    """Read native qacc through the explicit backend capability bridge."""

    def __init__(self, cfg, env):
        del cfg
        super().__init__(env)
        self._dof_ids = _metadata(env)["joint_dof_ids"]

    def __call__(self, env) -> torch.Tensor:
        from .reset_bridge import read_joint_acceleration

        return read_joint_acceleration(env, self._dof_ids).square().mean(-1).sqrt()


class actuator_saturation_share(normalized_joint_torques_l2):
    """Fraction of actuators above 98% of this world's active limit."""

    def __call__(self, env) -> torch.Tensor:
        limits = current_actuator_limits(env, self._actuator_ids, self._limits)
        return (_sensor_values(env, self._sensor_names).abs() >= 0.98 * limits).float().mean(-1)


__all__ = [
    "object_pos_tracking_error_cm", "object_ori_tracking_error_deg", "action_rate_l2",
    "action_acc_l2", "joint_vel_rms", "joint_acc_rms", "actuator_saturation_share",
]

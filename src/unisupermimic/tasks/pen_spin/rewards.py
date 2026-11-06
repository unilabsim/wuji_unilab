"""Measured Pen Spin rewards, ported from the source task's MDP terms.

The adaptation uses UniLab's named sensor read plan for actual actuator forces
and contacts. Torque normalization retains the nominal XML limits; overcurrent
uses the active world's limits, as in the source task.
"""

from __future__ import annotations

from functools import lru_cache

import mujoco
import numpy as np
import torch
from unilab.managers import ManagerTermBase

from .commands import JOINT_NAMES, task
from .observations import _quat_rotate, _runtime_link_pos, _runtime_pose
from .scene import materialize_scene


@lru_cache(maxsize=4)
def _model_metadata(scene_path: str) -> dict:
    """Bind immutable source metadata once, outside the simulation loop."""
    model = mujoco.MjModel.from_xml_path(scene_path)
    joint_ids = np.asarray([model.joint(name).id for name in JOINT_NAMES])
    actuator_ids = np.asarray([
        model.actuator(f"{name}_actuator").id for name in JOINT_NAMES
    ])
    ranges = model.actuator_forcerange[actuator_ids].copy()
    gears = model.actuator_gear[actuator_ids]
    if (
        not np.all(model.actuator_trntype[actuator_ids] == mujoco.mjtTrn.mjTRN_JOINT)
        or not np.allclose(gears[:, 0], 1.0)
        or not np.allclose(gears[:, 1:], 0.0)
        or not np.all(model.actuator_forcelimited[actuator_ids])
        or not np.isfinite(ranges).all()
        or np.any(ranges[:, 1] <= 0.0)
        or not np.allclose(ranges[:, 0], -ranges[:, 1])
    ):
        raise ValueError("Pen torque rewards require gear-1 actuators with symmetric positive limits")
    collision_sensors = {}
    for index in range(model.nsensor):
        sensor_name = model.sensor(index).name
        if sensor_name.startswith("finger_collision_"):
            geom_name = model.geom(int(model.sensor_objid[index])).name
            collision_sensors[geom_name] = sensor_name
    groups = tuple(
        tuple(collision_sensors[f"robot/right_finger{finger}_{part}_col"] for finger in range(1, 6))
        for part in ("link2", "link3", "link4", "tip_sensor")
    )
    return {
        "joint_limits": model.jnt_range[joint_ids].copy(),
        "joint_dof_ids": tuple(int(index) for index in model.jnt_dofadr[joint_ids]),
        "actuator_ids": actuator_ids,
        "nominal_limits": ranges[:, 1],
        "torque_sensors": tuple(f"actuator_force_{index}" for index in actuator_ids),
        "collision_sensors": groups,
    }


def _metadata(env) -> dict:
    metadata = getattr(env, "_pen_reward_metadata", None)
    if metadata is None:
        metadata = _model_metadata(str(materialize_scene()))
        env._pen_reward_metadata = metadata
    return metadata


def _sensor_values(env, names: tuple[str, ...]) -> torch.Tensor:
    views = env.scene.read_plan.sensor_tensor_views("robot", names)
    return torch.cat([
        views.values[name].reshape(env.num_envs, -1) for name in names
    ], dim=-1)


def current_actuator_limits(env, actuator_ids: torch.Tensor, nominal: torch.Tensor) -> torch.Tensor:
    """Read live limits published by startup DR, or fixed nominal physics."""
    fields = getattr(env, "_pen_model_fields", None)
    if fields is None:
        # Small synthetic task configurations omit the production DR event.
        # A registered DR table must contain the actual limit field.
        cfg = getattr(env, "cfg", getattr(env, "_cfg", None))
        events = getattr(cfg, "events", {}) or {}
        if events.get("source_physics") is not None:
            raise RuntimeError("Pen startup physics did not publish the active actuator force limits")
        return nominal.unsqueeze(0).expand(env.num_envs, -1)
    limits = fields["actuator_forcerange"]
    ids = actuator_ids.to(device=limits.device)
    return limits[:, ids, 1].to(device=nominal.device, dtype=nominal.dtype)


def reward(env, kind: str) -> torch.Tensor:
    motion = task(env)
    joints = env.scene.read_plan.joint_tensor_view("robot").joint_pos
    if kind == "joint_pos":
        limits = torch.as_tensor(
            _metadata(env)["joint_limits"], device=joints.device, dtype=joints.dtype
        )
        joint_range = (limits[:, 1] - limits[:, 0]).clamp_min(1.0e-6)
        error = (joints - motion.joint_pos).abs() / joint_range
        return torch.exp(-(error / 0.1).square()).mean(-1)
    rel_pos, rel_quat = _runtime_pose(env, motion)
    if kind == "object_pos":
        error = torch.linalg.vector_norm(rel_pos - motion.object_pose[:, :3], dim=-1)
        return torch.exp(-(error / 0.05).square())
    if kind == "object_rot":
        points = motion.axis_points.to(device=rel_pos.device, dtype=rel_pos.dtype)
        current = _quat_rotate(rel_quat[:, None, :], points[None, :, :])
        target = _quat_rotate(motion.object_pose[:, None, 3:], points[None, :, :])
        current_points = rel_pos[:, None, :] + current
        target_pos = motion.object_pose[:, :3]
        target_points = target_pos[:, None, :] + target
        error = torch.linalg.vector_norm(current_points - target_points, dim=-1).mean(-1)
        return torch.exp(-(error / 0.04).square())
    if kind == "link_pos":
        error = torch.linalg.vector_norm(
            _runtime_link_pos(env) - motion.link_pos, dim=-1
        )
        return torch.exp(-(error / 0.025).square()).mean(-1)
    if kind == "action_rate":
        action = env.action_manager
        return ((action.action - action.prev_action).square() + (action.action - 2 * action.prev_action + action.prev_prev_action).square()).sum(-1)
    if kind == "torque":
        metadata = _metadata(env)
        torque = _sensor_values(env, metadata["torque_sensors"])
        nominal = torch.as_tensor(metadata["nominal_limits"], device=torque.device, dtype=torque.dtype)
        return (torque / nominal).square().sum(-1)
    raise ValueError(f"Unknown pen reward kind: {kind}")


class normalized_joint_torques_l2(ManagerTermBase):
    """Actual actuator torque utilization, relative to nominal torque limits."""

    entity_name = "robot"

    def __init__(self, cfg, env):
        del cfg
        super().__init__(env)
        metadata = _metadata(env)
        self._sensor_names = metadata["torque_sensors"]
        self._limits = torch.as_tensor(
            metadata["nominal_limits"], device=env.device, dtype=torch.float32
        )
        self._actuator_ids = torch.as_tensor(
            metadata["actuator_ids"], device=env.device, dtype=torch.long
        )

    @property
    def tensor_sensor_names(self) -> tuple[str, ...]:
        return self._sensor_names

    def __call__(self, env) -> torch.Tensor:
        return (_sensor_values(env, self._sensor_names) / self._limits).square().sum(-1)


class sustained_overcurrent(normalized_joint_torques_l2):
    """Source leaky load integrator with row-scoped episode reset."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.charge = torch.zeros(env.num_envs, self._limits.numel(), device=env.device)
        # Keep the source spelling available to reward-phase readers.
        self._charge = self.charge

    def reset(self, env_ids=None) -> None:
        self.charge[env_ids if env_ids is not None else slice(None)] = 0.0

    def __call__(self, env, charge_s: float = 1.0, leak_s: float = 2.0, threshold: float = 0.8):
        limits = current_actuator_limits(env, self._actuator_ids, self._limits)
        load = (_sensor_values(env, self._sensor_names).abs() / limits).clamp(max=1.0)
        self.charge.add_(env.step_dt * (load / charge_s - self.charge / leak_s))
        self.charge.clamp_(0.0, 1.0)
        return (self.charge - threshold).clamp(min=0.0).sum(-1) / (1.0 - threshold)


class finger_self_collision_penalty(ManagerTermBase):
    """Count middle links and merge each distal/tip collider pair once."""

    entity_name = "robot"

    def __init__(self, cfg, env):
        del cfg
        super().__init__(env)
        groups = _metadata(env)["collision_sensors"]
        self._middle = groups[0] + groups[1]
        self._distal, self._tips = groups[2:]

    @property
    def tensor_sensor_names(self) -> tuple[str, ...]:
        return self._middle + self._distal + self._tips

    def __call__(self, env) -> torch.Tensor:
        middle = (_sensor_values(env, self._middle) > 0).float().sum(-1)
        distal = (_sensor_values(env, self._distal) > 0) | (_sensor_values(env, self._tips) > 0)
        return middle + distal.float().sum(-1)


def _axis_from_pose(quaternion: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    return _quat_rotate(quaternion, axis.expand(*quaternion.shape[:-1], 3))


def reference_end(env) -> torch.Tensor:
    return task(env).reference_end


def numerical_instability(env, max_pos: float = 3.0, max_vel: float = 80.0) -> torch.Tensor:
    from .reset_bridge import physics_nan_mask

    obj = env.scene.read_plan.body_tensor_view("object", ("object/cube",))
    joints = env.scene.read_plan.joint_tensor_view("robot")
    object_pos = obj.pos_w[:, 0]
    object_vel = obj.lin_vel_w[:, 0]
    return (
        physics_nan_mask(env)
        | ((object_pos - env.scene.env_origins(env.device)).abs().amax(-1) > max_pos)
        | ~torch.isfinite(obj.quat_w[:, 0]).all(-1)
        | ~torch.isfinite(object_vel).all(-1)
        | (object_vel.abs().amax(-1) > max_vel)
        | ~torch.isfinite(joints.joint_pos).all(-1)
        | ~torch.isfinite(joints.joint_vel).all(-1)
        | (joints.joint_vel.abs().amax(-1) > max_vel)
    )


def object_position_error_exceeded(env, threshold: float = 0.05) -> torch.Tensor:
    position, _ = _runtime_pose(env, task(env))
    return torch.linalg.vector_norm(position - task(env).object_pose[:, :3], dim=-1) > threshold


def object_orientation_error_exceeded(env, threshold_deg: float = 60.0) -> torch.Tensor:
    _, current_quat = _runtime_pose(env, task(env))
    axis = torch.tensor((1.0, 0.0, 0.0), device=env.device, dtype=current_quat.dtype)
    current = _axis_from_pose(current_quat, axis)
    target = _axis_from_pose(task(env).object_pose[:, 3:], axis)
    angle = torch.acos((current * target).sum(-1).clamp(-1.0, 1.0))
    return angle >= torch.deg2rad(torch.as_tensor(threshold_deg, device=env.device, dtype=angle.dtype))


__all__ = [
    "reward", "normalized_joint_torques_l2", "sustained_overcurrent",
    "finger_self_collision_penalty", "reference_end", "numerical_instability",
    "object_position_error_exceeded", "object_orientation_error_exceeded",
]

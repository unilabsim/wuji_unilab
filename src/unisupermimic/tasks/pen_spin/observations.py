# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Pen-spin observation terms matching the source 155D actor interface."""

from __future__ import annotations

import torch
from unilab.managers import ManagerTermBase

from .commands import LINK_NAMES, task


def _quat_conjugate(value: torch.Tensor) -> torch.Tensor:
    return torch.cat((value[..., :1], -value[..., 1:]), dim=-1)


def _quat_multiply(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = first.unbind(-1)
    w2, x2, y2, z2 = second.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _quat_rotate(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    pure = torch.cat((torch.zeros_like(vector[..., :1]), vector), dim=-1)
    return _quat_multiply(_quat_multiply(quat, pure), _quat_conjugate(quat))[..., 1:]


def _runtime_pose(env, motion):
    """Read pen pose in the palm frame from the packed plan or entity cache."""
    object_view = env.scene.read_plan.body_tensor_view("object", ("object/cube",))
    palm_view = env.scene.read_plan.body_tensor_view("robot", ("robot/right_palm_link",))
    object_pos = object_view.pos_w[:, 0]
    object_quat = object_view.quat_w[:, 0]
    palm_pos = palm_view.pos_w[:, 0]
    palm_quat = palm_view.quat_w[:, 0]
    palm_inv = _quat_conjugate(palm_quat)
    rel_pos = _quat_rotate(palm_inv, object_pos - palm_pos)
    rel_quat = _quat_multiply(palm_inv, object_quat)
    return rel_pos, rel_quat


def _runtime_link_pos(env) -> torch.Tensor:
    """Return all 21 hand-link positions in the palm frame."""
    links = env.scene.read_plan.body_tensor_view("robot", LINK_NAMES).pos_w
    palm_view = env.scene.read_plan.body_tensor_view("robot", (LINK_NAMES[0],))
    palm_pos = palm_view.pos_w[:, 0]
    palm_quat = palm_view.quat_w[:, 0]
    return _quat_rotate(
        _quat_conjugate(palm_quat)[:, None, :], links - palm_pos[:, None, :]
    )


class PenFeature(ManagerTermBase):
    entity_name = "robot"

    def __init__(self, cfg, env):
        super().__init__(env)
        params = getattr(cfg, "params", {}) or {}
        self.kind = str(params.get("kind", getattr(cfg, "kind", "motion")))
        self.biased = bool(params.get("biased", False))
        self.noise_bound = float(params.get("noise_bound", 0.0))

    @property
    def tensor_body_names(self):
        return LINK_NAMES

    def __call__(self, env, **kwargs):
        kind = str(kwargs.get("kind", self.kind))
        motion = task(env)
        if kind == "motion":
            return motion.command
        if getattr(env.scene, "read_plan", None) is None:
            widths = {
                "joint_pos": 20,
                "joint_vel": 20,
                "object_pose": 6,
                "last_action": 20,
                "link_pos": 63,
                "joint_position_error": 20,
                "link_position_error": 63,
            }
            if kind in widths:
                return torch.zeros((env.num_envs, widths[kind]), device=env.device)
            raise ValueError(f"Unknown pen feature kind: {kind}")
        joints = env.scene.read_plan.joint_tensor_view("robot")
        if kind == "joint_pos":
            default = motion.robot.data.default_joint_pos(env.device)
            default = default[0] if default.ndim >= 2 else default
            value = joints.joint_pos - default
            if bool(kwargs.get("biased", self.biased)):
                bias = getattr(motion.robot.data, "encoder_bias_tensor", None)
                if bias is not None:
                    value = value + bias
            noise = float(kwargs.get("noise_bound", self.noise_bound))
            if not env.cfg.observations["policy"].enable_corruption:
                noise = 0.0
            return value + torch.empty_like(value).uniform_(-noise, noise) if noise else value
        if kind == "joint_vel":
            value = joints.joint_vel
            noise = float(kwargs.get("noise_bound", self.noise_bound))
            if not env.cfg.observations["policy"].enable_corruption:
                noise = 0.0
            # Source mjlab applies observation noise before the term scale.
            # Its raw +/-0.05 rad/s noise therefore becomes +/-0.0125 in the
            # actor input. Keep the clean critic and play scale unchanged.
            if noise:
                value = value + torch.empty_like(value).uniform_(-noise, noise)
            return value * 0.25
        if kind == "object_pose":
            return object_axis_points(env, motion)
        if kind == "link_pos":
            return _runtime_link_pos(env).flatten(1)
        if kind == "joint_position_error":
            return joints.joint_pos - motion.joint_pos
        if kind == "link_position_error":
            return (_runtime_link_pos(env) - motion.link_pos).flatten(1)
        if kind == "last_action":
            return env.action_manager.action
        raise ValueError(f"Unknown pen feature kind: {kind}")


def object_axis_points(env, motion):
    rel_pos, rel_quat = _runtime_pose(env, motion)
    points = rel_pos[:, None] + _quat_rotate(rel_quat[:, None], motion.axis_points[None])
    scale = float(motion.cfg.object_position_tanh_scale_m)
    return (scale * torch.tanh(points / scale)).flatten(1)


class PenObjectPoseFeature(PenFeature):
    """Object-owned term that makes the pen body part of the read plan."""

    entity_name = "object"

    def __init__(self, cfg, env):
        ManagerTermBase.__init__(self, env)

    @property
    def tensor_body_names(self):
        return ("object/cube",)

    def __call__(self, env, **kwargs):
        if getattr(env.scene, "read_plan", None) is None:
            return torch.zeros((env.num_envs, 6), device=env.device)
        return object_axis_points(env, task(env))


class PenObjectPoseCameraModel(PenObjectPoseFeature):
    """Source camera noise, per-episode bias, latency and burst dropout.

    The source CircularBuffer is represented by a newest-first tensor here;
    its age lookup, selected-row backfill and append semantics are unchanged.
    The manager owns this state, so explicit partial reset never advances the
    other rows' camera stream in the same control tick.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        params = getattr(cfg, "params", {}) or {}
        self.bias_position = tuple(params.get("bias_position_m", (-.008, .008)))
        self.bias_orientation = tuple(params.get("bias_orientation_rad", (-.05, .05)))
        self.noise_position = tuple(params.get("noise_position_m", (-.002, .002)))
        self.noise_orientation = tuple(params.get("noise_orientation_rad", (-.015, .015)))
        self.lag_min, self.lag_max = map(int, params.get("lag_ticks", (1, 1)))
        self.dropout_enter_prob = float(params.get("dropout_enter_prob", .03))
        self.dropout_continue_prob = float(params.get("dropout_continue_prob", .78))
        self.dropout_max_ticks = int(params.get("dropout_max_ticks", 40))
        for value in (self.bias_position, self.bias_orientation, self.noise_position, self.noise_orientation):
            if len(value) != 2 or value[1] < value[0]:
                raise ValueError("Camera noise and bias ranges must be ordered pairs")
        if not 0 <= self.lag_min <= self.lag_max or self.dropout_max_ticks < 0:
            raise ValueError("Camera lag and dropout duration must be nonnegative")
        if not all(0 <= value <= 1 for value in (self.dropout_enter_prob, self.dropout_continue_prob)):
            raise ValueError("Camera dropout probabilities must be in [0,1]")
        n, device = env.num_envs, env.device
        self._history = torch.zeros((self.lag_max + self.dropout_max_ticks + 1, n, 7), device=device)
        self._length = torch.zeros(n, device=device, dtype=torch.long)
        self._bias_pos = torch.zeros((n, 3), device=device)
        self._bias_quat = torch.zeros((n, 4), device=device)
        self._bias_quat[:, 0] = 1.
        self._age = torch.zeros(n, device=device, dtype=torch.long)
        self._holding = torch.zeros(n, device=device, dtype=torch.bool)
        self._hold_ticks = torch.zeros(n, device=device, dtype=torch.long)
        self._reset_ids = None
        self._resample_bias(slice(None))

    @staticmethod
    def _random_rotation(count, device, bounds):
        angle = torch.empty(count, device=device).uniform_(*bounds)
        axis = torch.randn((count, 3), device=device)
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.cat((torch.cos(angle[:, None] / 2), axis * torch.sin(angle[:, None] / 2)), -1)

    def _resample_bias(self, ids):
        position = self._bias_pos[ids]
        if self.bias_position != (0., 0.):
            position.uniform_(*self.bias_position)
        else:
            position.zero_()
        self._bias_pos[ids] = position
        quaternion = self._bias_quat[ids]
        if self.bias_orientation != (0., 0.):
            quaternion = self._random_rotation(len(quaternion), quaternion.device, self.bias_orientation)
        else:
            quaternion.zero_()
            quaternion[:, 0] = 1.
        self._bias_quat[ids] = torch.where(quaternion[:, :1] < 0, -quaternion, quaternion)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._history[:, ids] = 0.
        self._length[ids] = 0
        self._age[ids] = 0
        self._holding[ids] = False
        self._hold_ticks[ids] = 0
        self._resample_bias(ids)
        self._reset_ids = (
            ids if not isinstance(ids, slice) and len(ids) < self.num_envs
            and not self._env.cfg.auto_reset else None
        )
        return {}

    def _points(self, pose):
        motion = task(self._env)
        points = pose[:, None, :3] + _quat_rotate(pose[:, None, 3:], motion.axis_points[None])
        scale = float(motion.cfg.object_position_tanh_scale_m)
        return (scale * torch.tanh(points / scale)).flatten(1)

    def memory_state(self):
        from unisupermimic.rl.algos.dex_supermimic.rotations import quat_to_mat

        rows = torch.arange(self.num_envs, device=self._age.device)
        return dict(
            camera_raw=self._points(self._history[0]).clone(),
            camera_obs=self._points(self._history[self._age, rows]).clone(),
            camera_holding=self._holding.float().unsqueeze(-1).clone(),
            camera_ticks=self._hold_ticks.float().unsqueeze(-1).clone(),
            camera_bias_pos=self._bias_pos.clone(),
            camera_bias_rot=quat_to_mat(self._bias_quat).flatten(1),
        )

    def __call__(self, env, **kwargs):
        del kwargs
        if getattr(env.scene, "read_plan", None) is None:
            return torch.zeros((env.num_envs, 6), device=env.device)
        position, quaternion = _runtime_pose(env, task(env))
        pose = torch.cat((position, quaternion), -1).clone()
        pose[:, :3] += self._bias_pos
        pose[:, 3:] = _quat_multiply(self._bias_quat, pose[:, 3:])
        if self.noise_position != (0., 0.):
            pose[:, :3] += torch.empty_like(pose[:, :3]).uniform_(*self.noise_position)
        if self.noise_orientation != (0., 0.):
            pose[:, 3:] = _quat_multiply(
                self._random_rotation(len(pose), pose.device, self.noise_orientation), pose[:, 3:]
            )
        pose[:, 3:] = torch.where(pose[:, 3:4] < 0, -pose[:, 3:], pose[:, 3:])
        reset_ids, self._reset_ids = self._reset_ids, None
        if reset_ids is not None and self._length.max() > 0:
            self._history[:, reset_ids] = pose[reset_ids][None]
            self._length[reset_ids] = 1
        else:
            self._history[1:] = self._history[:-1].clone()
            self._history[0] = pose
            self._length.add_(1).clamp_max_(len(self._history))
            if self.dropout_enter_prob > 0.:
                roll = torch.rand(env.num_envs, device=pose.device)
                holding = torch.where(
                    self._holding,
                    (roll < self.dropout_continue_prob) & (self._hold_ticks < self.dropout_max_ticks),
                    roll < self.dropout_enter_prob,
                )
                self._hold_ticks = torch.where(holding, self._hold_ticks + 1, 0)
                self._holding = holding
            else:
                holding = torch.zeros_like(self._holding)
            aged = self._age + 1
            fresh = (torch.randint(self.lag_min, self.lag_max + 1, (env.num_envs,), device=pose.device)
                     if self.lag_max else torch.zeros_like(self._age))
            age = torch.where(holding, aged, torch.minimum(fresh, aged))
            self._age = age.minimum(self._length - 1).clamp_min(0)
        rows = torch.arange(env.num_envs, device=pose.device)
        return self._points(self._history[self._age, rows])


def feature(env, kind: str) -> torch.Tensor:
    """Plain-function compatibility adapter for direct term tests."""
    cfg = type("Cfg", (), {"params": {"kind": kind}})()
    return PenFeature(cfg, env)(env, kind=kind)


__all__ = ["PenFeature", "PenObjectPoseFeature", "PenObjectPoseCameraModel", "feature"]

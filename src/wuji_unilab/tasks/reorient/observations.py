# pyright: reportAttributeAccessIssue=false
# Adapted from wuji-technology/wuji-mjlab 9410a3a, mdp/observations.py.
# Copyright 2026 Wuji Technology Co., Ltd. Apache-2.0.
"""Policy and privileged critic features under the target schema."""

import torch
from unilab.managers import ManagerTermBase

from .commands import task
from .math import (
    conjugate,
    multiply,
    random_quaternions,
    rotation6d,
)


def _require_read_plan(env):
    read_plan = env.scene.read_plan
    if read_plan is None:
        raise NotImplementedError("Wuji observations require the Manager packed tensor read plan")
    return read_plan


def feature(env, kind: str, injection_prob: float = 0.0):
    state = task(env)
    robot = state.robot
    read_plan = env.scene.read_plan
    if kind in ("joint", "target_error"):
        action = env.action_manager.get_term("joint_pos")
        limits = action.limits
        if limits.ndim == 3:
            limits = limits[0]
        center = limits.mean(dim=-1)
        half = torch.clamp((limits[:, 1] - limits[:, 0]) * 0.5, min=1e-6)
        if read_plan is None:
            return torch.zeros((env.num_envs, robot.num_joints), device=env.device)
        joints = read_plan.joint_tensor_view("robot").joint_pos
        joint = torch.clamp((joints - center) / half, -1.0, 1.0)
        if kind == "joint":
            return joint + robot.data.encoder_bias_tensor.to(joints.device)
        return joint - torch.clamp((action.target - center) / half, -1.0, 1.0)
    if kind == "position":
        value = (
            torch.zeros((env.num_envs, 3), device=env.device)
            if read_plan is None
            else state.cube_tag()[0]
        )
        if injection_prob:
            mask = env.rng.random(env.num_envs) < injection_prob
            value = torch.where(
                mask[:, None],
                env.rng.uniform(-0.5, 0.5, (env.num_envs, 3)),
                value,
            )
        return value
    if kind == "orientation":
        q = (
            torch.zeros((env.num_envs, 4), device=env.device)
            if read_plan is None
            else multiply(state.cube_tag()[1], conjugate(state.goal))
        )
        if injection_prob:
            mask = env.rng.random(env.num_envs) < injection_prob
            q = torch.where(
                mask[:, None], random_quaternions(env.rng, env.num_envs, device=env.device), q
            )
        return rotation6d(q)
    if kind == "action":
        return env.action_manager.prev_action
    if read_plan is None:
        if kind in ("qpos", "qvel"):
            return torch.zeros((env.num_envs, robot.num_joints), device=env.device)
        if kind == "tips":
            return torch.zeros((env.num_envs, 15), device=env.device)
        if kind in ("linvel", "angvel"):
            return torch.zeros((env.num_envs, 3), device=env.device)
    if kind == "qpos":
        joints = _require_read_plan(env).joint_tensor_view("robot").joint_pos
        default_joint_pos = robot.data.default_joint_pos(env.device)
        default_joint_pos = (
            default_joint_pos[0] if default_joint_pos.ndim >= 2 else default_joint_pos
        )
        return joints - default_joint_pos
    if kind == "qvel":
        return _require_read_plan(env).joint_tensor_view("robot").joint_vel
    if kind == "tips":
        bodies = _require_read_plan(env).body_tensor_view("robot", state.tip_names)
        return (bodies.pos_w - torch.tensor((0.0, 0.0, 0.5), device=env.device)).reshape(
            env.num_envs, 15
        )
    if kind == "linvel":
        return state.cube_velocity()
    if kind == "angvel":
        return (
            _require_read_plan(env).body_tensor_view("object", (state.cube_name,)).ang_vel_w[:, 0]
        )
    if kind == "perturbation":
        return state.perturbation
    if kind == "force_direction":
        return state.force_direction
    if kind == "progress":
        return torch.stack(
            (
                torch.where(state.window > 0, 0, state.hold).float() / state.cfg.success_hold_steps,
                state.window.float() / state.cfg.goal_switch_delay,
                env.episode_length_buf.float() / env.max_episode_length,
                state.goal_count.float(),
            ),
            dim=-1,
        )
    if kind == "cage":
        return torch.clamp(state.cage_counter / 10, max=1.0)[:, None]
    if kind == "dr":
        return state.dr_ratios
    if kind == "palm":
        if read_plan is None:
            return torch.zeros((env.num_envs, 6), device=env.device)
        return rotation6d(state.palm_pose()[1])
    raise ValueError(f"Unknown Wuji observation feature: {kind}")


class TipsFeature(ManagerTermBase):
    entity_name = "robot"

    def __init__(self, cfg, env):
        super().__init__(env)
        self.cfg = cfg

    @property
    def tensor_state_entity(self) -> str:
        return "robot"

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return task(self._env).tip_names

    def __call__(self, env) -> torch.Tensor:
        del env
        return feature(self._env, "tips")


class CubeAngularVelocityFeature(ManagerTermBase):
    entity_name = "object"

    def __init__(self, cfg, env):
        super().__init__(env)
        self.cfg = cfg

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return (task(self._env).cube_name,)

    def __call__(self, env) -> torch.Tensor:
        del env
        state = task(self._env)
        read_plan = getattr(self._env.scene, "read_plan", None)
        if read_plan is None:
            return torch.zeros((self._env.num_envs, 3), device=self._env.device)
        return read_plan.body_tensor_view("object", (state.cube_name,)).ang_vel_w[:, 0]


class PalmFeature(ManagerTermBase):
    entity_name = "robot"

    def __init__(self, cfg, env):
        super().__init__(env)
        self.cfg = cfg

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return (task(self._env).palm_name,)

    def __call__(self, env) -> torch.Tensor:
        del env
        state = task(self._env)
        if getattr(self._env.scene, "read_plan", None) is None:
            return torch.zeros((self._env.num_envs, 6), device=self._env.device)
        return rotation6d(state.palm_pose()[1])

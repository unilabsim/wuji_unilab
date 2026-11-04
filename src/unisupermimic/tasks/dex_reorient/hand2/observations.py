"""Observation terms for the Hand 2 cube reorientation task."""

from __future__ import annotations

import torch
from unilab.managers import ManagerTermBase

from .commands import task
from .math import conjugate, multiply, random_quaternions, rotation6d


def _read_plan(env):
    return getattr(env.scene, "read_plan", None)


def feature(
    env,
    kind: str,
    injection_prob: float = 0.0,
    max_outside_steps: int = 10,
) -> torch.Tensor:
    state = task(env)
    plan = _read_plan(env)
    if plan is None:
        # ObservationManager prepares term dimensions before the backend's
        # packed read plan is bound.  Return shape-stable placeholders during
        # that cold phase; runtime calls use the real plan below.
        joints = int(getattr(state.robot, "num_joints", len(state.tip_names) * 4))
        if kind in ("joint", "target_error", "qpos", "qvel", "joint_velocity"):
            return torch.zeros((env.num_envs, joints), device=env.device)
        if kind in ("position", "cube_position", "linvel", "force_direction"):
            return torch.zeros((env.num_envs, 3), device=env.device)
        if kind in ("perturbation", "dr"):
            return torch.zeros((env.num_envs, 6), device=env.device)
        if kind in ("orientation", "palm"):
            return torch.zeros((env.num_envs, 6), device=env.device)
        if kind == "angvel":
            return torch.zeros((env.num_envs, 3), device=env.device)
        if kind == "cage":
            return torch.zeros((env.num_envs, 1), device=env.device)
        if kind == "progress":
            return torch.zeros((env.num_envs, 4), device=env.device)
        if kind == "action":
            return torch.zeros((env.num_envs, 20), device=env.device)
    if kind in ("joint", "target_error"):
        # The policy contract follows the Wuji reference: joint angles and
        # target error are expressed in the same centred, half-range units.
        # Keeping this normalization here also makes the feature independent
        # of the physical radians used by the MJCF.
        action = env.action_manager.get_term("joint_pos")
        limits = action.limits
        if limits.ndim == 3:
            limits = limits[0]
        center = limits.mean(dim=-1)
        half = (limits[:, 1] - limits[:, 0]) * 0.5 + 1e-6
        joints = plan.joint_tensor_view("robot")
        current = torch.clamp((joints.joint_pos - center) / half, -1.0, 1.0)
        if kind == "joint":
            # Match Dex-SuperMimic's ``joint_pos_limit_normalized`` exactly:
            # the policy sees physical qpos in limit units.  Encoder bias is
            # handled only when forming the executed WM target ``u=c-bias``;
            # adding it here would silently change the actor observation
            # contract from the migration object.
            return current
        target = torch.clamp((action.target - center) / half, -1.0, 1.0)
        return current - target
    if kind in ("qpos", "qvel"):
        joints = plan.joint_tensor_view("robot")
        if kind == "qpos":
            default = state.robot.data.default_joint_pos(env.device)
            default = default[0] if default.ndim == 2 else default
            return joints.joint_pos - default
        return joints.joint_vel
    if kind == "joint_velocity":
        return plan.joint_tensor_view("robot").joint_vel
    if kind in ("position", "cube_position"):
        value = state.cube_tag()[0]
        if injection_prob:
            mask = env.rng.random(env.num_envs) < injection_prob
            noise = env.rng.uniform(-0.5, 0.5, value.shape)
            value = torch.where(mask[:, None], noise, value)
        return value
    if kind == "orientation":
        # Both the cube and goal must be expressed in the calibrated tag
        # frame.  ``state.goal`` is stored as a world-frame quaternion, while
        # ``cube_tag`` already applies tag^-1 to the cube orientation.
        cube_tag = state.cube_tag()
        _tag_pos, tag_quat = state.tag_pose()
        goal_tag = multiply(conjugate(tag_quat), state.goal)
        q = multiply(cube_tag[1], conjugate(goal_tag))
        if injection_prob:
            mask = env.rng.random(env.num_envs) < injection_prob
            q = torch.where(
                mask[:, None],
                random_quaternions(env.rng, env.num_envs, device=env.device),
                q,
            )
        return rotation6d(q)
    if kind == "palm":
        return rotation6d(state.palm_pose()[1])
    if kind == "action":
        return env.action_manager.prev_action
    if kind == "linvel":
        return state.cube_velocity()
    if kind == "angvel":
        return plan.body_tensor_view("object", (state.cube_name,)).ang_vel_w[:, 0]
    if kind == "progress":
        return torch.stack(
            (
                state.hold.float() / state.cfg.success_hold_steps,
                state.reward_window_timer.float() / state.cfg.goal_switch_delay,
                env.episode_length_buf.float() / env.max_episode_length,
                state.goal_count.float(),
            ), dim=-1,
        )
    if kind == "force_direction":
        return state.force_direction
    if kind == "perturbation":
        return state.perturbation
    if kind == "cage":
        return torch.clamp(
            state.cage_counter / max(float(max_outside_steps), 1.0), max=1.0
        )[:, None]
    if kind == "dr":
        return state.dr_ratios
    raise ValueError(f"Unknown Hand2 observation feature: {kind}")


class TipsFeature(ManagerTermBase):
    """Expose fingertip body positions through the packed read plan."""

    entity_name = "robot"

    def __init__(self, cfg, env):
        super().__init__(env)
        self.cfg = cfg

    @property
    def tensor_body_names(self):
        return task(self._env).tip_names

    def __call__(self, env):
        del env
        if _read_plan(self._env) is None:
            return torch.zeros((self._env.num_envs, len(self.tensor_body_names) * 3), device=self._env.device)
        bodies = _read_plan(self._env).body_tensor_view("robot", self.tensor_body_names)
        offset = torch.tensor((0.0, 0.0, 0.5), device=self._env.device)
        origins = self._env.scene.env_origins(self._env.device)
        # Dex-SuperMimic's body_pos_rel subtracts each vector environment
        # origin before applying the fixed critic offset.  Omitting this is
        # invisible for env 0 but injects the vectorized scene placement into
        # every subsequent critic feature.
        return (bodies.pos_w - origins[:, None, :] - offset).reshape(self._env.num_envs, -1)


class CubeFeature(ManagerTermBase):
    """Observation adapter that declares the cube body packed read.

    The generic :func:`feature` helper is intentionally kept as a plain
    function for the terms that only consume robot state.  Position and
    orientation, however, read ``object/cube`` through the scene read plan.
    A class based term lets UniLab collect that body during read-plan
    compilation while preserving the existing observation shapes and
    corruption parameters.
    """

    entity_name = "object"

    def __init__(self, cfg, env):
        super().__init__(env)
        self.cfg = cfg
        self.kind = str(cfg.params["kind"])

    @property
    def tensor_body_names(self):
        return (task(self._env).cube_name,)

    def __call__(self, env, **kwargs):
        # ``ObservationManager`` forwards term params to __call__ as well as
        # retaining them on the config.  They are consumed from cfg so this
        # adapter works for both the cold dimension-probe and runtime calls.
        del kwargs
        params = dict(self.cfg.params)
        params.pop("kind", None)
        return feature(env, self.kind, **params)


class PalmFeature(ManagerTermBase):
    """Declare the palm body read used by the privileged critic."""

    entity_name = "robot"

    def __init__(self, cfg, env):
        super().__init__(env)
        self.cfg = cfg

    @property
    def tensor_body_names(self):
        command = task(self._env)
        # The command's tag-frame observation is also read from the packed
        # body plane; declaring both here avoids a backend lookup in
        # ``Hand2ReorientCommand.tag_pose``.
        return (command.palm_name, command.tag_name)

    def __call__(self, env, **kwargs):
        del kwargs
        return feature(env, "palm")


__all__ = ["CubeFeature", "PalmFeature", "TipsFeature", "feature"]

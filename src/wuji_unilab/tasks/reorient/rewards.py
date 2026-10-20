# pyright: reportAttributeAccessIssue=false
# Adapted from wuji-technology/wuji-mjlab 9410a3a, mdp/rewards.py and cage.py.
# Copyright 2026 Wuji Technology Co., Ltd. Apache-2.0.
"""Reorientation signals with termination-owned cage state."""

import torch
from unilab.managers import ManagerTermBase

from .commands import task
from .math import conjugate, rotate


class CageDrop(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(env)
        self.state = task(env)

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return self.state.robot.body_names

    @property
    def entity_name(self) -> str:
        return "robot"

    @property
    def tensor_state_entity(self) -> str:
        return "robot"

    def __call__(self, env, margin: float = 0.01, max_outside_steps: int = 10):
        state = self.state
        p, q = state.palm_pose()
        robot_bodies = env.scene.read_plan.body_tensor_view("robot", self.state.robot.body_names)
        cube = env.scene.read_plan.body_tensor_view("object", (state.cube_name,))
        local = rotate(conjugate(q)[:, None, :], robot_bodies.pos_w - p[:, None, :])
        cube_local = rotate(conjugate(q), cube.pos_w[:, 0] - p)
        low = local.amin(dim=1) - torch.tensor(
            (margin, margin, margin), device=env.device, dtype=torch.float32
        )
        high = local.amax(dim=1) + torch.tensor(
            (margin, margin, 0.03), device=env.device, dtype=torch.float32
        )
        state.outside.copy_(torch.any((cube_local < low) | (cube_local > high), dim=-1))
        state.cage_counter.copy_(
            torch.clamp(state.cage_counter + torch.where(state.outside, 1.0, -0.5), 0.0, 15.0)
        )
        return state.cage_counter > max_outside_steps


def reward(env, kind: str):
    state = task(env)
    if kind in ("orientation", "orientation_alignment"):
        excess_error = torch.maximum(state.error() - 0.2, torch.zeros_like(state.error()))
        return torch.clamp(1 - excess_error / torch.pi, 0.0, 1.0)
    if kind == "hand_pose":
        joints = env.scene.read_plan.joint_tensor_view("robot")
        return torch.square(joints.joint_pos - state.robot.data.default_joint_pos(env.device)).sum(
            dim=-1
        )
    if kind == "action_rate":
        a = env.action_manager
        return torch.square(a.action - a.prev_action).sum(dim=-1) + torch.square(
            a.action - 2 * a.prev_action + a.prev_prev_action
        ).sum(dim=-1)
    if kind in ("cage", "cage_escape"):
        return state.outside.float() * state.cage_counter / 10 * 4
    if kind in ("hold", "hold_escalation"):
        return ((state.window > 0) & state.within).float() * state.reward_timer
    raise ValueError(f"Unknown Wuji reward: {kind}")


class SensorFeatureReward(ManagerTermBase):
    """Reward-function adapter that publishes its packed sensor read contract."""

    def __init__(self, cfg, env):
        super().__init__(env)
        self.cfg = cfg
        self.state = task(env)
        kind = cfg.params["kind"]
        supported = {
            "torque": ("actuator_force",),
            "tip_slide": ("tip_velocity",),
            "finger_collision": ("self_found",),
            "palm_detach": ("distal_found", "proximal_found"),
        }
        if kind not in supported:
            raise ValueError(f"Unknown Wuji sensor reward: {kind}")
        self.kind = kind
        self.sensor_keys = supported[kind]
        self.sensor_names = tuple(
            name for key in self.sensor_keys for name in self.state.sensor_groups[key]
        )
        dimensions = tuple(self._env.scene.bind_sensor_data(self.sensor_names).dimensions)
        for name, dimension in zip(self.sensor_names, dimensions, strict=True):
            self.state.sensor_dimensions[name] = int(dimension)

    @property
    def tensor_sensor_names(self) -> tuple[str, ...]:
        return self.sensor_names

    def __call__(self, env, **kwargs):
        del env, kwargs
        state = self.state
        if self.kind == "torque":
            force = state.sensor_values("actuator_force")
            return torch.square(force / state.actuator_force_limits).sum(dim=-1)
        if self.kind == "tip_slide":
            velocity = state.sensor_values("tip_velocity").reshape(self.num_envs, 5, 3)
            relative_velocity = velocity - state.cube_velocity()[:, None, :]
            contact = state.sensor_values("tips_found") > 0
            return (torch.linalg.vector_norm(relative_velocity, dim=-1) * contact).sum(dim=-1)
        if self.kind == "finger_collision":
            return (state.sensor_values("self_found") > 0).float().sum(dim=-1)
        return (
            (state.sensor_values("distal_found") > 0).any(dim=-1)
            & ~(state.sensor_values("proximal_found") > 0).any(dim=-1)
        ).float()


def _full_reset_mask(env, reset_mask) -> torch.Tensor:
    """Normalize Manager curriculum mask boundaries (``None`` or full slice)."""
    if reset_mask is None or isinstance(reset_mask, slice):
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    return torch.as_tensor(reset_mask, device=env.device).to(dtype=torch.bool)


def success_curriculum(
    env,
    reset_mask=None,
    count_threshold: int = 3,
    delta_per_loop: float = 0.08,
):
    state = task(env)
    reset_mask = _full_reset_mask(env, reset_mask)
    successes = state.goal_count >= count_threshold
    applied_delta = float(delta_per_loop * torch.where(successes, 1.0, -1.0).sum() / env.num_envs)
    state.difficulty = float(min(max(state.difficulty + applied_delta, 0.0), 1.0))
    goal_count = state.goal_count.float()
    has_reset_rows = bool(reset_mask.any())
    return {
        "value": state.difficulty,
        "mean_goal_reach_count": float(goal_count[reset_mask].mean()) if has_reset_rows else 0.0,
        "success_env_frac": float(successes[reset_mask].float().mean()) if has_reset_rows else 0.0,
        "delta": applied_delta,
    }


def adaptive_episode_curriculum(
    env,
    reset_mask=None,
    target_steps: int = 800,
    inc_rate: float = 0.1,
    dec_rate: float = 0.2,
    min_scale: float = 0.05,
    drop_gamma: float = 1.0,
):
    state = task(env)
    reset_mask = _full_reset_mask(env, reset_mask)
    progress = torch.clamp(env.episode_length_buf / target_steps, max=1.0)
    reached = progress >= 1
    dropped = env.termination_manager.terminated & ~reached
    applied_delta = float(
        (
            reached.float() * inc_rate
            - dropped.float() * torch.pow(1 - progress, drop_gamma) * dec_rate
        ).sum()
        / env.num_envs
    )
    state.adaptive = float(min(max(state.adaptive + applied_delta, min_scale), 1.0))
    selected = reset_mask
    return {
        "value": state.adaptive,
        "mean_episode_steps": float(env.episode_length_buf[selected].float().mean())
        if bool(selected.any())
        else 0.0,
        "mean_progress": float(progress[selected].mean()) if bool(selected.any()) else 0.0,
        "reached_target_frac": float(reached[selected].float().mean())
        if bool(selected.any())
        else 0.0,
        "early_drop_frac": float(dropped[selected].float().mean()) if bool(selected.any()) else 0.0,
        "target_steps": float(target_steps),
        "delta": applied_delta,
    }


def metric(env, kind: str):
    state = task(env)
    if kind in ("success", "goal_reach_count"):
        return state.goal_count.float()
    if kind == "orientation_error":
        return state.error()
    if kind == "survival":
        return env.episode_length_buf.float() * env.step_dt
    if kind == "cube_survival_steps":
        return env.episode_length_buf.float()
    if kind in ("tips_contact", "fingertip_contact_count"):
        return (state.sensor_values("tips_found") > 0).float().sum(dim=-1)
    if kind in ("cage", "cage_escape_frequency"):
        return (state.cage_counter > 0).float()
    if kind in ("joint_velocity_rms", "joint_vel_rms"):
        joints = env.scene.read_plan.joint_tensor_view("robot").joint_vel
        return torch.sqrt(torch.square(joints).mean(dim=-1))
    if kind == "torque_rms":
        force = state.sensor_values("actuator_force")
        return torch.sqrt(torch.square(force).mean(dim=-1))
    if kind == "action_delta_rms":
        action = env.action_manager
        delta = action.action - action.prev_action
        return torch.sqrt(torch.square(delta).mean(dim=-1))
    if kind in ("cube_height", "cube_height_above_palm"):
        p, q = state.palm_pose()
        cube = state.cube_tag()[0]
        return rotate(conjugate(q), cube - p)[:, 2]
    if kind == "action_jerk_rms":
        action = env.action_manager
        jerk = action.action - 2 * action.prev_action + action.prev_prev_action
        return torch.sqrt(torch.square(jerk).mean(dim=-1))
    if kind == "success_interval":
        return state.goal_timer.float()
    if kind == "torque_saturation_ratio":
        force = torch.abs(state.sensor_values("actuator_force"))
        return (force >= 0.9 * state.actuator_force_limits).float().mean(dim=-1)
    if kind == "finger_collision_frequency":
        found = state.sensor_values("self_found") > 0
        return found.any(dim=-1).float()
    if kind == "difficulty":
        return torch.full((env.num_envs,), state.difficulty, device=env.device)
    if kind == "adaptive":
        return torch.full((env.num_envs,), state.adaptive, device=env.device)
    if kind == "success_rate":
        return (state.goal_count > 0).float()
    raise ValueError(f"Unknown Wuji metric: {kind}")

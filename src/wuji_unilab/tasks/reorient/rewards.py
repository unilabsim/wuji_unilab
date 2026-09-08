# pyright: reportAttributeAccessIssue=false
# Adapted from wuji-technology/wuji-mjlab 9410a3a, mdp/rewards.py and cage.py.
# Copyright 2026 Wuji Technology Co., Ltd. Apache-2.0.
"""Reorientation signals with termination-owned cage state."""

import numpy as np
from unilab.managers import ManagerTermBase

from .commands import task
from .math import conjugate, rotate


class CageDrop(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(env)

    def __call__(self, env, margin: float = 0.01, max_outside_steps: int = 10):
        state = task(env)
        p, q = state.palm_pose()
        local = rotate(conjugate(q)[:, None, :], state.robot.data.body_link_pos_w - p[:, None, :])
        cube = rotate(conjugate(q), state.cube.data.root_link_pos_w - p)
        low = local.min(axis=1) - np.array([margin, margin, margin])
        high = local.max(axis=1) + np.array([margin, margin, 0.03])
        state.outside[:] = np.any((cube < low) | (cube > high), axis=-1)
        state.cage_counter[:] = np.clip(
            state.cage_counter + np.where(state.outside, 1, -0.5), 0, 15
        )
        return state.cage_counter >= max_outside_steps


def reward(env, kind: str):
    state = task(env)
    if kind in ("orientation", "orientation_alignment"):
        return np.clip(1 - np.maximum(state.error() - 0.2, 0) / np.pi, 0, 1)
    if kind == "hand_pose":
        return np.square(state.robot.data.joint_pos - state.robot.data.default_joint_pos).sum(
            axis=-1
        )
    if kind == "action_rate":
        a = env.action_manager
        return np.square(a.action - a.prev_action).sum(axis=-1) + np.square(
            a.action - 2 * a.prev_action + a.prev_prev_action
        ).sum(axis=-1)
    if kind == "torque":
        return np.square(state.views["actuator_force"].read()).sum(axis=-1)
    if kind == "tip_slide":
        velocity = state.views["tip_velocity"].read().reshape(env.num_envs, 5, 3)
        found = state.views["tips_found"].read() > 0
        return (np.square(velocity).sum(axis=-1) * found).sum(axis=-1)
    if kind in ("cage", "cage_escape"):
        return state.outside * state.cage_counter / 10 * 4
    if kind in ("collision", "finger_collision"):
        return (state.views["self_found"].read() > 0).sum(axis=-1).astype(np.float32)
    if kind in ("hold", "hold_escalation"):
        return np.where(
            state.window > 0, np.minimum(state.hold / state.cfg.success_hold_steps, 4), 0
        )
    if kind == "palm_detach":
        return (
            (state.views["distal_found"].read() > 0).any(axis=-1)
            & ~(state.views["proximal_found"].read() > 0).any(axis=-1)
        ).astype(np.float32)
    raise ValueError(f"Unknown Wuji reward: {kind}")


def success_curriculum(
    env,
    env_ids,
    count_threshold: int = 3,
    delta_per_loop: float = 0.08,
):
    state = task(env)
    ids = np.arange(env.num_envs) if env_ids is None else env_ids
    applied_delta = 0.0
    if len(ids):
        successes = state.goal_count[ids] >= count_threshold
        applied_delta = float(delta_per_loop * np.where(successes, 1, -1).sum() / env.num_envs)
        state.difficulty = float(np.clip(state.difficulty + applied_delta, 0, 1))
    else:
        successes = np.zeros(0, dtype=bool)
    return {
        "value": state.difficulty,
        "mean_goal_reach_count": float(np.mean(state.goal_count[ids])) if len(ids) else 0.0,
        "success_env_frac": float(np.mean(successes)) if len(ids) else 0.0,
        "delta": applied_delta,
    }


def adaptive_episode_curriculum(
    env,
    env_ids,
    target_steps: int = 800,
    inc_rate: float = 0.1,
    dec_rate: float = 0.2,
    min_scale: float = 0.05,
    drop_gamma: float = 1.0,
):
    state = task(env)
    ids = np.arange(env.num_envs) if env_ids is None else env_ids
    applied_delta = 0.0
    progress = np.zeros(len(ids), dtype=np.float32)
    reached = np.zeros(len(ids), dtype=bool)
    dropped = np.zeros(len(ids), dtype=bool)
    if len(ids):
        progress = np.minimum(env.episode_length_buf[ids] / target_steps, 1)
        reached = progress >= 1
        dropped = env.termination_manager.terminated[ids] & ~reached
        applied_delta = float(
            (reached * inc_rate - dropped * np.power(1 - progress, drop_gamma) * dec_rate).sum()
            / env.num_envs
        )
        state.adaptive = float(np.clip(state.adaptive + applied_delta, min_scale, 1))
    return {
        "value": state.adaptive,
        "mean_episode_steps": float(np.mean(env.episode_length_buf[ids])) if len(ids) else 0.0,
        "mean_progress": float(np.mean(progress)) if len(ids) else 0.0,
        "reached_target_frac": float(np.mean(reached)) if len(ids) else 0.0,
        "early_drop_frac": float(np.mean(dropped)) if len(ids) else 0.0,
        "target_steps": float(target_steps),
        "delta": applied_delta,
    }


def metric(env, kind: str):
    state = task(env)
    if kind in ("success", "goal_reach_count"):
        return state.goal_count.astype(np.float32)
    if kind == "orientation_error":
        return state.error()
    if kind == "survival":
        return env.episode_length_buf.astype(np.float32) * env.step_dt
    if kind == "cube_survival_steps":
        return env.episode_length_buf.astype(np.float32)
    if kind in ("tips_contact", "fingertip_contact_count"):
        return (state.views["tips_found"].read() > 0).sum(axis=-1).astype(np.float32)
    if kind in ("cage", "cage_escape_frequency"):
        return (state.cage_counter > 0).astype(np.float32)
    if kind in ("joint_velocity_rms", "joint_vel_rms"):
        return np.sqrt(np.square(state.robot.data.joint_vel).mean(axis=-1))
    if kind == "torque_rms":
        return np.sqrt(np.square(state.views["actuator_force"].read()).mean(axis=-1))
    if kind == "action_delta_rms":
        return np.sqrt(
            np.square(env.action_manager.action - env.action_manager.prev_action).mean(axis=-1)
        )
    if kind in ("cube_height", "cube_height_above_palm"):
        p, q = state.palm_pose()
        return rotate(conjugate(q), state.cube.data.root_link_pos_w - p)[:, 2]
    if kind == "action_jerk_rms":
        a = env.action_manager
        jerk = a.action - 2 * a.prev_action + a.prev_prev_action
        return np.sqrt(np.square(jerk).mean(axis=-1))
    if kind == "success_interval":
        return state.goal_timer.astype(np.float32)
    if kind == "torque_saturation_ratio":
        force = np.abs(state.views["actuator_force"].read())
        return (force >= 0.9 * state.actuator_force_limits).mean(axis=-1)
    if kind == "finger_collision_frequency":
        found = state.views["self_found"].read() > 0
        return found.any(axis=-1).astype(np.float32)
    raise ValueError(f"Unknown Wuji metric: {kind}")

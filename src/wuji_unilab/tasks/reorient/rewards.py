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
    if kind == "orientation":
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
    if kind == "cage":
        return state.outside * state.cage_counter / 10 * 4
    if kind == "collision":
        return (state.views["self_found"].read() > 0).sum(axis=-1).astype(np.float32)
    if kind == "hold":
        return np.where(
            state.window > 0, np.minimum(state.hold / state.cfg.success_hold_steps, 4), 0
        )
    if kind == "palm_detach":
        return (
            (state.views["distal_found"].read() > 0).any(axis=-1)
            & ~(state.views["proximal_found"].read() > 0).any(axis=-1)
        ).astype(np.float32)
    raise ValueError(f"Unknown Wuji reward: {kind}")


def curriculum(
    env,
    env_ids,
    count_threshold: int = 3,
    delta: float = 0.08,
    target_steps: int = 800,
    inc_rate: float = 0.1,
    dec_rate: float = 0.2,
    min_scale: float = 0.05,
):
    state = task(env)
    ids = np.arange(env.num_envs) if env_ids is None else env_ids
    if len(ids):
        state.difficulty = float(
            np.clip(
                state.difficulty
                + delta
                * np.where(state.goal_count[ids] >= count_threshold, 1, -1).sum()
                / env.num_envs,
                0,
                1,
            )
        )
        progress = np.minimum(env.episode_length_buf[ids] / target_steps, 1)
        reached = progress >= 1
        dropped = env.termination_manager.terminated[ids] & ~reached
        state.adaptive = float(
            np.clip(
                state.adaptive
                + (reached * inc_rate - dropped * (1 - progress) * dec_rate).sum() / env.num_envs,
                min_scale,
                1,
            )
        )
    return {"difficulty": state.difficulty, "adaptive": state.adaptive}


def metric(env, kind: str):
    state = task(env)
    if kind == "success":
        return state.goal_count.astype(np.float32)
    if kind == "orientation_error":
        return state.error()
    if kind == "survival":
        return env.episode_length_buf.astype(np.float32) * env.step_dt
    if kind == "tips_contact":
        return (state.views["tips_found"].read() > 0).sum(axis=-1).astype(np.float32)
    if kind == "cage":
        return state.outside.astype(np.float32)
    if kind == "joint_velocity_rms":
        return np.sqrt(np.square(state.robot.data.joint_vel).mean(axis=-1))
    if kind == "torque_rms":
        return np.sqrt(np.square(state.views["actuator_force"].read()).mean(axis=-1))
    if kind == "action_delta_rms":
        return np.sqrt(
            np.square(env.action_manager.action - env.action_manager.prev_action).mean(axis=-1)
        )
    if kind == "cube_height":
        p, q = state.palm_pose()
        return rotate(conjugate(q), state.cube.data.root_link_pos_w - p)[:, 2]
    raise ValueError(f"Unknown Wuji metric: {kind}")

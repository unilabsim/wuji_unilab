# pyright: reportAttributeAccessIssue=false
# Adapted from wuji-technology/wuji-mjlab 9410a3a, mdp/observations.py.
# Copyright 2026 Wuji Technology Co., Ltd. Apache-2.0.
"""Policy and privileged critic features under the target schema."""

import numpy as np

from .commands import task
from .math import conjugate, multiply, random_quaternions, rotation6d


def feature(env, kind: str, injection_prob: float = 0.0):
    state = task(env)
    robot = state.robot
    if kind in ("joint", "target_error"):
        action = env.action_manager.get_term("joint_pos")
        center = action.limits.mean(axis=-1)
        half = np.maximum((action.limits[:, 1] - action.limits[:, 0]) * 0.5, 1e-6)
        joint = np.clip((robot.data.joint_pos - center) / half, -1, 1)
        return joint if kind == "joint" else joint - np.clip((action.target - center) / half, -1, 1)
    if kind == "position":
        value = state.cube_tag()[0]
        if injection_prob:
            mask = env.rng.random(env.num_envs) < injection_prob
            value[mask] += env.rng.uniform(-0.1, 0.1, (int(mask.sum()), 3))
        return value
    if kind == "orientation":
        q = multiply(state.cube_tag()[1], conjugate(state.goal))
        if injection_prob:
            mask = env.rng.random(env.num_envs) < injection_prob
            q[mask] = random_quaternions(env.rng, int(mask.sum()))
        return rotation6d(q)
    if kind == "action":
        return env.action_manager.prev_action
    if kind == "qpos":
        return robot.data.joint_pos - robot.data.default_joint_pos
    if kind == "qvel":
        return robot.data.joint_vel
    if kind == "tips":
        return (robot.data.body_link_pos_w[:, state.tip_ids] - np.array([0, 0, 0.5])).reshape(
            env.num_envs, 15
        )
    if kind == "linvel":
        return state.cube.data.root_link_lin_vel_w
    if kind == "angvel":
        return state.cube.data.root_link_ang_vel_w
    if kind == "perturbation":
        return state.perturbation
    if kind == "force_direction":
        return state.force_direction
    if kind == "progress":
        return np.stack(
            (
                state.hold / state.cfg.success_hold_steps,
                state.window / state.cfg.goal_switch_delay,
                env.episode_length_buf / env.max_episode_length,
                state.goal_count,
            ),
            axis=-1,
        )
    if kind == "cage":
        return np.minimum(state.cage_counter / 10, 1)[:, None]
    if kind == "dr":
        return state.dr_ratios
    if kind == "palm":
        return rotation6d(state.palm_pose()[1])
    raise ValueError(f"Unknown Wuji observation feature: {kind}")

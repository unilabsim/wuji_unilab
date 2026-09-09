# pyright: reportAttributeAccessIssue=false
# Adapted from wuji-technology/wuji-mjlab 9410a3a, mdp/commands.py.
# Copyright 2026 Wuji Technology Co., Ltd. Apache-2.0.
"""SO(3) goal state and shared, explicitly owned Wuji task information."""

from dataclasses import dataclass
from typing import cast

import numpy as np
from unilab.base.entity import Entity
from unilab.managers import CommandTerm, CommandTermCfg

from .math import angle_error, conjugate, multiply, random_quaternions, rotate
from .scene import materialize_scene


@dataclass(kw_only=True)
class ReorientCommandCfg(CommandTermCfg):
    success_threshold: float = 0.2
    success_hold_steps: int = 5
    goal_switch_delay: int = 20

    def build(self, env):
        return ReorientCommand(self, env)


class ReorientCommand(CommandTerm):
    cfg: ReorientCommandCfg

    def __init__(self, cfg: ReorientCommandCfg, env):
        super().__init__(cfg, env)
        self.robot = cast(Entity, env.scene["robot"])
        self.cube = cast(Entity, env.scene["object"])
        self.palm_id = self.robot.find_bodies("robot/right_palm_link")[0][0]
        self.tip_ids = self.robot.find_bodies("robot/right_finger[1-5]_link4")[0]
        self.views = {
            k: env.scene.bind_sensor_data(v) for k, v in materialize_scene().sensors.items()
        }
        # Force ranges are immutable MJCF metadata; load them once at init.
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(materialize_scene().path))
        self.actuator_force_limits = np.maximum(model.actuator_forcerange[:, 1], 1e-8).astype(
            np.float32
        )
        self.goal = np.tile([1, 0, 0, 0], (env.num_envs, 1)).astype(np.float32)
        self.goal_count = np.zeros(env.num_envs, dtype=np.int32)
        self.hold = np.zeros(env.num_envs, dtype=np.int32)
        self.goal_timer = np.zeros(env.num_envs, dtype=np.int32)
        self.window = np.zeros(env.num_envs, dtype=np.int32)
        self.success = np.zeros(env.num_envs, dtype=bool)
        self.cage_counter = np.zeros(env.num_envs, dtype=np.float32)
        self.outside = np.zeros(env.num_envs, dtype=bool)
        self.perturbation = np.zeros((env.num_envs, 6), dtype=np.float32)
        self.force_direction = np.zeros((env.num_envs, 3), dtype=np.float32)
        self.dr_ratios = np.ones((env.num_envs, 6), dtype=np.float32)
        self.difficulty = 0.0
        self.adaptive = 0.05
        self.metrics = {
            "goal_reach_count": np.zeros(env.num_envs, dtype=np.float32),
            "ori_error": np.zeros(env.num_envs, dtype=np.float32),
            "hold_counter": np.zeros(env.num_envs, dtype=np.float32),
            "goal_timer": np.zeros(env.num_envs, dtype=np.float32),
            "in_success_window": np.zeros(env.num_envs, dtype=np.float32),
            "window_timer": np.zeros(env.num_envs, dtype=np.float32),
        }
        env.playback_overlay_provider = self.playback_overlays

    @property
    def command(self):
        return self.goal

    def palm_pose(self):
        return (
            self.robot.data.body_link_pos_w[:, self.palm_id],
            self.robot.data.body_link_quat_w[:, self.palm_id],
        )

    def tag_pose(self):
        p, q = self.palm_pose()
        tag_q = multiply(q, np.array([2**-0.5, 0, 2**-0.5, 0], dtype=np.float32))
        tag_p = p + rotate(q, np.array([0.0262, 0, -0.0563], dtype=np.float32))
        return tag_p, tag_q

    def cube_tag(self):
        p, q = self.tag_pose()
        inv = conjugate(q)
        return (
            rotate(inv, self.cube.data.root_link_pos_w - p),
            multiply(inv, self.cube.data.root_link_quat_w),
        )

    def error(self):
        return angle_error(self.cube_tag()[1], self.goal)

    def playback_overlays(self):
        """Return the task-owned target-cube overlay for playback renderers."""
        _, palm_quat = self.palm_pose()
        # The target is a visual goal above the live object, not the hand-root
        # mocap body used by physics reset.
        target_pos = self.cube.data.root_link_pos_w + np.array([0.0, 0.0, 0.10], dtype=np.float32)
        target_quat = multiply(palm_quat, self.goal)
        return {
            "overlays": [
                [
                    {
                        "type": "box",
                        "pos": target_pos[env_id],
                        "quat": target_quat[env_id],
                        "size": [0.027, 0.027, 0.027],
                        "rgba": [0.9, 0.2, 0.1, 0.45],
                    }
                ]
                for env_id in range(self.num_envs)
            ]
        }

    def _resample_command(self, env_ids):
        self.goal[env_ids] = random_quaternions(self._env.rng, len(env_ids))
        self.hold[env_ids] = 0
        self.goal_timer[env_ids] = 0
        self.window[env_ids] = 0
        self.success[env_ids] = False

    def _update_metrics(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.metrics["goal_reach_count"][ids] = self.goal_count[ids]
        self.metrics["ori_error"][ids] = self.error()[ids]
        self.metrics["hold_counter"][ids] = self.hold[ids]
        self.metrics["goal_timer"][ids] = self.goal_timer[ids]
        self.metrics["in_success_window"][ids] = self.window[ids] > 0
        self.metrics["window_timer"][ids] = self.window[ids]

    def _update_command(self, env_ids):
        if env_ids is not None:
            return  # reset backfills the new goal; no episode time has elapsed
        within = self.error() < self.cfg.success_threshold
        self.goal_timer += 1
        self.success.fill(False)
        approaching = self.window == 0
        self.hold[:] = np.where(within, self.hold + 1, 0)
        succeeded = approaching & (self.hold >= self.cfg.success_hold_steps)
        self.success[succeeded] = True
        self.goal_count[succeeded] += 1
        self.window[succeeded] = 1
        self.window[(self.window > 0) & ~succeeded] += 1
        change = np.flatnonzero(self.window >= self.cfg.goal_switch_delay)
        if len(change):
            self._resample_command(change)

    def reset(self, env_ids):
        extras = super().reset(env_ids)
        self.goal_count[env_ids] = 0
        self.goal_timer[env_ids] = 0
        self.cage_counter[env_ids] = 0
        self.outside[env_ids] = False
        self.perturbation[env_ids] = 0
        self.force_direction[env_ids] = 0
        return extras


def task(env) -> ReorientCommand:
    return env.command_manager.get_term("reorient_command")

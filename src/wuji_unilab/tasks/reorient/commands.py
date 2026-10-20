# pyright: reportAttributeAccessIssue=false
# Adapted from wuji-technology/wuji-mjlab 9410a3a, mdp/commands.py.
# Copyright 2026 Wuji Technology Co., Ltd. Apache-2.0.
"""SO(3) goal state and shared, explicitly owned Wuji task information."""

from dataclasses import dataclass
from typing import cast

import torch
from unilab.base.entity import Entity
from unilab.managers import CommandTerm, CommandTermCfg

from .math import (
    angle_error,
    conjugate,
    multiply,
    random_quaternions,
    rotate,
)
from .scene import materialize_scene


@dataclass(kw_only=True)
class ReorientCommandCfg(CommandTermCfg):
    entity_name: str = "robot"
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
        device = self._device
        self.palm_name = "robot/right_palm_link"
        self.tip_names = tuple(name for name in self.robot.body_names if name.endswith("_link4"))
        self.cube_name = "object/cube"
        self.sensor_groups = {
            key: tuple(names) for key, names in materialize_scene().sensors.items()
        }
        # Validate sensor layouts on the cold path. Manager hot reads consume
        # the compiled scene read plan, rather than scattered backend reads.
        self.sensor_dimensions = {}
        for names in self.sensor_groups.values():
            view = env.scene.bind_sensor_data(tuple(names))
            if view is None:
                continue
            self.sensor_dimensions.update(zip(names, view.dimensions, strict=True))
        # Force ranges are immutable MJCF metadata; load them once at init.
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(materialize_scene().path))
        self.actuator_force_limits = torch.tensor(
            model.actuator_forcerange[:, 1], dtype=torch.float32, device=device
        ).clamp_min(1e-8)
        self.goal = torch.tile(torch.tensor((1.0, 0.0, 0.0, 0.0), device=device), (env.num_envs, 1))
        self.goal_count = torch.zeros(env.num_envs, dtype=torch.int32, device=device)
        self.hold = torch.zeros(env.num_envs, dtype=torch.int32, device=device)
        self.goal_timer = torch.zeros(env.num_envs, dtype=torch.int32, device=device)
        self.window = torch.zeros(env.num_envs, dtype=torch.int32, device=device)
        self.success = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        self.within = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        self.reward_timer = torch.zeros(env.num_envs, dtype=torch.int32, device=device)
        self.cage_counter = torch.zeros(env.num_envs, dtype=torch.float32, device=device)
        self.outside = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        self.perturbation = torch.zeros((env.num_envs, 6), dtype=torch.float32, device=device)
        self.force_direction = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=device)
        self.dr_ratios = torch.ones((env.num_envs, 6), dtype=torch.float32, device=device)
        self.difficulty = 0.0
        self.adaptive = 0.05
        self.metrics = {
            name: torch.zeros(env.num_envs, dtype=torch.float32, device=device)
            for name in (
                "goal_reach_count",
                "ori_error",
                "hold_counter",
                "goal_timer",
                "in_success_window",
                "window_timer",
            )
        }
        self._palm_pos = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=device)
        self._palm_quat = torch.zeros((env.num_envs, 4), dtype=torch.float32, device=device)
        self._tag_pos = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=device)
        self._tag_quat = torch.zeros((env.num_envs, 4), dtype=torch.float32, device=device)
        self._tag_offset = torch.tensor((0.0262, 0.0, -0.0563), dtype=torch.float32, device=device)
        self._tag_rotation = torch.tensor(
            (2**-0.5, 0.0, 0.0, 2**-0.5), dtype=torch.float32, device=device
        )

    @property
    def command(self):
        return self.goal

    @property
    def tensor_sensor_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for key in (
                "actuator_force",
                "tip_velocity",
                "self_found",
                "tips_found",
                "distal_found",
                "proximal_found",
            )
            for name in self.sensor_groups[key]
        )

    def playback_debug_overlay_getter(self):
        """Opt in to UniLab playback overlay discovery with the goal marker."""
        from .overlay import goal_overlay_getter_for_command

        return goal_overlay_getter_for_command(self)

    def bind_read_phase(self) -> None:
        self._pose_stale = True
        self._tag_stale = True

    def body_state(self, entity_name: str, body_names: tuple[str, ...]):
        read_plan = getattr(self._env.scene, "read_plan", None)
        if read_plan is None:
            raise NotImplementedError("Wuji task requires the Manager packed tensor read plan")
        return read_plan.body_tensor_view(entity_name, body_names)

    def sensor_values(self, key: str) -> torch.Tensor:
        read_plan = getattr(self._env.scene, "read_plan", None)
        names = self.sensor_groups[key]
        if read_plan is None or not set(names).issubset(
            set(getattr(read_plan, "sensor_names", {}).get("robot", ()))
        ):
            raise NotImplementedError(
                f"Wuji sensors {names} must be compiled into the Manager packed read plan"
            )
        entity_name = "robot"
        views = read_plan.sensor_tensor_views(entity_name, names).values
        return torch.cat([views[name] for name in names], dim=-1)

    def palm_pose(self):
        if getattr(self, "_pose_stale", True):
            view = self.body_state("robot", (self.palm_name,))
            self._palm_pos.copy_(view.pos_w[:, 0])
            self._palm_quat.copy_(view.quat_w[:, 0])
            self._pose_stale = False
        return self._palm_pos, self._palm_quat

    def tag_pose(self):
        if getattr(self, "_tag_stale", True):
            p, q = self.palm_pose()
            self._tag_quat.copy_(multiply(q, self._tag_rotation))
            self._tag_pos.copy_(p + rotate(q, self._tag_offset.expand_as(p)))
            self._tag_stale = False
        return self._tag_pos, self._tag_quat

    def cube_tag(self):
        p, q = self.tag_pose()
        cube = self.body_state("object", (self.cube_name,))
        inv = conjugate(q)
        return (
            rotate(inv, cube.pos_w[:, 0] - p),
            multiply(inv, cube.quat_w[:, 0]),
        )

    def cube_position(self) -> torch.Tensor:
        return self.body_state("object", (self.cube_name,)).pos_w[:, 0]

    def cube_velocity(self) -> torch.Tensor:
        return self.body_state("object", (self.cube_name,)).lin_vel_w[:, 0]

    def error(self):
        return angle_error(self.cube_tag()[1], self.goal)

    def _update_metrics(self):
        self.metrics["goal_reach_count"].copy_(self.goal_count)
        self.metrics["ori_error"].copy_(self.error())
        self.metrics["hold_counter"].copy_(
            torch.where(self.window > 0, torch.zeros_like(self.hold), self.hold)
        )
        self.metrics["goal_timer"].copy_(self.goal_timer)
        self.metrics["in_success_window"].copy_(self.window > 0)
        self.metrics["window_timer"].copy_(self.window)

    def _resample_command(self, reset_mask):
        samples = random_quaternions(self._rng, self.num_envs, device=self._device)
        zeroed_counters = (
            self.hold,
            self.goal_timer,
            self.window,
            self.reward_timer,
        )
        for destination in zeroed_counters:
            destination.masked_fill_(reset_mask, 0)
        self.goal.copy_(torch.where(reset_mask[:, None], samples, self.goal))
        self.success.masked_fill_(reset_mask, False)

    def _update_command(self):
        self._pose_stale = True
        self._tag_stale = True
        within = self.error() < self.cfg.success_threshold
        self.within.copy_(within)
        self.goal_timer += 1
        self.success.fill_(False)
        approaching = self.window == 0
        self.hold.copy_(torch.where(within, self.hold + 1, torch.zeros_like(self.hold)))
        succeeded = approaching & (self.hold >= self.cfg.success_hold_steps)
        in_window = ~approaching | succeeded
        self.reward_timer.copy_(
            torch.where(
                in_window & within,
                self.reward_timer + 1,
                self.reward_timer,
            )
        )
        self.reward_timer.masked_fill_(succeeded, 0)
        self.success.masked_fill_(succeeded, True)
        self.goal_count.add_(succeeded.to(self.goal_count.dtype))
        self.window.masked_fill_(succeeded, 1)
        active = (self.window > 0) & ~succeeded
        self.window.add_(active.to(self.window.dtype))
        change = torch.nonzero(self.window >= self.cfg.goal_switch_delay, as_tuple=False).flatten()
        if change.numel():
            mask = torch.zeros_like(self.window, dtype=torch.bool)
            mask[change] = True
            self._resample_command(mask)

    def refresh_reset(self, reset_mask: torch.Tensor):
        del reset_mask
        self._pose_stale = True
        self._tag_stale = True

    def post_compute_reset(self):
        self._pose_stale = True
        self._tag_stale = True

    def reset(self, reset_mask):
        extras = super().reset(reset_mask)
        self.goal_count.masked_fill_(reset_mask, 0)
        self.goal_timer.masked_fill_(reset_mask, 0)
        self.within.masked_fill_(reset_mask, False)
        self.reward_timer.masked_fill_(reset_mask, 0)
        self.cage_counter.masked_fill_(reset_mask, 0)
        self.outside.masked_fill_(reset_mask, False)
        self.perturbation[reset_mask] = 0.0
        self.force_direction[reset_mask] = 0.0
        self._pose_stale = True
        self._tag_stale = True
        return extras


def task(env) -> ReorientCommand:
    return env.command_manager.get_term("reorient_command")

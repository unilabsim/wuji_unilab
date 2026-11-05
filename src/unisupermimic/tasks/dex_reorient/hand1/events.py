# pyright: reportAttributeAccessIssue=false
# Adapted from wuji-technology/wuji-mjlab 9410a3a, mdp/event_impl/.
# Copyright 2026 Wuji Technology Co., Ltd. Apache-2.0.
"""Task-owned reset and randomization through declared Entity capabilities."""

import math

import mujoco
import numpy as np
import torch
from unilab.base.reset_selectors import ResetMaskBuffer
from unilab.managers import ManagerTermBase

from .commands import task
from .math import multiply, random_quaternions, soft_limits, uniform_sample
from .reset_bridge import (
    bind_mocap_reset,
    publish_fixed_model,
    write_mocap_reset,
)
from .scene import materialize_scene


class ResetWuji(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(env)
        self.robot = env.scene["robot"]
        self.cube = env.scene["object"]
        self.pose = bind_mocap_reset(self.robot, "robot/right_palm_link", term_name="wuji_reset")
        self.default_pose = torch.as_tensor(
            np.asarray(self.pose.default_pose).copy(), device=env.device, dtype=torch.float32
        )
        self.default = self.robot.data.default_joint_pos(env.device)
        if self.default.ndim >= 2:
            self.default = self.default[0]
        self.default = self.default.to(device=env.device, dtype=torch.float32)
        limits = self.robot.data.soft_joint_pos_limits(env.device)
        if limits.ndim >= 3:
            limits = limits[0]
        self.limits = soft_limits(
            limits,
            float(cfg.params.get("soft_limit_factor", 1.0)),
        )
        self.offset_ids = self.robot.find_joints("robot/right_finger[1-5]_joint[134]")[0]
        self._reset_mask = ResetMaskBuffer(env.num_envs, env.device)

    def __call__(
        self,
        env,
        env_ids: torch.Tensor,
        soft_limit_factor: float,
        joint_noise: tuple,
        cube_noise: float,
        wrist_pitch: tuple,
    ):
        del soft_limit_factor
        # mjbatch fills derived fields only after they have been bound.
        # Register qacc before this reset's forward so the first metric is a
        # measured acceleration as in the source and Pen Spin tasks.
        from .rewards import _joint_acceleration

        _joint_acceleration(env)
        reset_mask = self._reset_mask.scatter(env_ids)
        joint = self.default.repeat(env.num_envs, 1)
        selected = torch.tensor(self.offset_ids, device=joint.device, dtype=torch.long)
        joint[:, selected] += uniform_sample(
            env.rng, joint_noise, (env.num_envs, selected.numel()), env.device
        )
        joint = torch.clamp(joint, self.limits[None, :, 0], self.limits[None, :, 1])
        self.robot.write_joint_state_to_sim_mask(joint, torch.zeros_like(joint), reset_mask)
        poses = self.default_pose.repeat(env.num_envs, 1)
        angle = uniform_sample(env.rng, wrist_pitch, (env.num_envs,), env.device)
        zeros = torch.zeros_like(angle)
        rotation = torch.stack((torch.cos(angle / 2), zeros, torch.sin(angle / 2), zeros), dim=-1)
        poses[:, 3:] = multiply(poses[:, 3:], rotation)
        write_mocap_reset(self.pose, env_ids, poses)
        state = self.cube.data.default_root_state(env.device).clone()
        if state.ndim >= 2 and state.shape[0] == 1 and env.num_envs != 1:
            state = state.repeat(env.num_envs, 1)
        state[:, :3] += uniform_sample(
            env.rng, (-cube_noise, cube_noise), (env.num_envs, 3), env.device
        )
        state[:, 3:7] = random_quaternions(env.rng, env.num_envs, device=env.device)
        state[:, 7:] = 0
        self.cube.write_root_state_tensor_to_sim(state, reset_mask)


class ResetDisturbanceCaches(ManagerTermBase):
    """Clear the source task's cached interval disturbance on reset."""

    def __init__(self, cfg, env):
        del cfg
        super().__init__(env)

    def __call__(self, env, env_ids: torch.Tensor):
        state = task(env)
        if env_ids is None or isinstance(env_ids, slice):
            rows = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        else:
            rows = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).reshape(-1)
        if rows.numel():
            state.perturbation[rows] = 0
            state.force_direction[rows] = 0


class FixedRandomization(ManagerTermBase):
    """Publish the source per-world randomization once, at startup."""

    uses_startup_rows = True

    def __init__(self, cfg, env):
        super().__init__(env)
        self.robot = env.scene["robot"]
        self.cube = env.scene["object"]
        self.device = env.device
        self.bindings = {}
        self.samples = {}
        self.ratios = {}
        term = "wuji_randomize"
        r, o = self.robot, self.cube
        # Keep the geometry groups identical to the two source mjlab startup
        # events.  Contact solver parameters are randomized only on the soft
        # palm/thumb pads and on fingers 2--5.  In particular, link1 capsules
        # and visual/mesh-disabled geoms must not receive this randomization.
        capsule_ids = r.find_geoms("robot/right_finger[1-5]_link[23]_col")[0]
        palm_contact_ids = r.find_geoms(
            (
                "robot/right_palm_collision",
                "robot/right_finger1_link[2-4]_col",
                "robot/right_finger1_link2_softbody_col",
            )
        )[0]
        finger_contact_ids = r.find_geoms(
            (
                "robot/right_finger[2-5]_link[2-4]_col",
            )
        )[0]
        friction_ids = r.find_geoms(
            ("robot/.*palm_.*", "robot/.*finger.*_col")
        )[0]
        hand_ids = r.find_bodies("robot/.*")[0]
        self.bindings["friction"] = (
            r,
            *r.bind_geom_friction_write(friction_ids, term_name=term),
            r.write_geom_friction_to_sim,
            "geom_ids",
        )
        self.bindings["hand_size"] = (
            r,
            *r.bind_geom_size_write(capsule_ids, term_name=term),
            r.write_geom_size_to_sim,
            "geom_ids",
        )
        self.bindings["solref_palm"] = (
            r,
            *r.bind_geom_solref_write(palm_contact_ids, term_name=term),
            r.write_geom_solref_to_sim,
            "geom_ids",
        )
        self.bindings["solref_fingers"] = (
            r,
            *r.bind_geom_solref_write(finger_contact_ids, term_name=term),
            r.write_geom_solref_to_sim,
            "geom_ids",
        )
        self.bindings["solimp_palm"] = (
            r,
            *r.bind_geom_solimp_write(palm_contact_ids, term_name=term),
            r.write_geom_solimp_to_sim,
            "geom_ids",
        )
        self.bindings["solimp_fingers"] = (
            r,
            *r.bind_geom_solimp_write(finger_contact_ids, term_name=term),
            r.write_geom_solimp_to_sim,
            "geom_ids",
        )
        self.bindings["cube_size"] = (
            o,
            *o.bind_geom_size_write(term_name=term),
            o.write_geom_size_to_sim,
            "geom_ids",
        )
        self.bindings["cube_mass"] = (
            o,
            *o.bind_body_mass_write(term_name=term),
            o.write_body_mass_to_sim,
            "body_ids",
        )
        self.bindings["cube_inertia"] = (
            o,
            *o.bind_body_inertia_write(term_name=term),
            o.write_body_inertia_to_sim,
            "body_ids",
        )
        self.bindings["cube_com"] = (
            o,
            *o.bind_body_ipos_write(term_name=term),
            o.write_body_ipos_to_sim,
            "body_ids",
        )
        self.bindings["hand_mass"] = (
            r,
            *r.bind_body_mass_write(hand_ids, term_name=term),
            r.write_body_mass_to_sim,
            "body_ids",
        )
        self.bindings["hand_inertia"] = (
            r,
            *r.bind_body_inertia_write(hand_ids, term_name=term),
            r.write_body_inertia_to_sim,
            "body_ids",
        )
        self.bindings["armature"] = (
            r,
            *r.bind_joint_armature_write(term_name=term),
            r.write_joint_armature_to_sim,
            "joint_ids",
        )
        self.gain_ids, self.kp, self.kd = r.bind_actuator_gain_write(term_name=term)
        # EventManager is constructed before CommandManager in UniLab. The
        # source scene is cold metadata, so it must not depend on a command
        # term that has not been constructed yet on the MJWarp path.
        model = mujoco.MjModel.from_xml_path(str(materialize_scene()))
        self._robot_geom_count = sum(
            model.body(int(body)).name.startswith("robot/") for body in model.geom_bodyid
        )
        self._object_geom_count = sum(
            model.body(int(body)).name.startswith("object/") for body in model.geom_bodyid
        )
        joints = [model.joint(name).id for name in r.joint_names]
        damping = model.dof_damping[model.jnt_dofadr[joints]]
        self._damping_ratio = float((damping / np.maximum(damping, 1e-8)).mean())

    def _sample(self, env, name: str, default: torch.Tensor, ranges: dict):
        # 1.4 model-field bindings return immutable one-row/default tables for
        # NumPy-backed fields. They are model metadata, not per-environment state.
        default = default[0] if default.ndim >= 3 else default
        shape = (env.num_envs, *default.shape)
        if name == "cube_inertia":
            factor = self.ratios["cube_mass"][..., None]
        elif name == "cube_com":
            self.samples[name] = default + uniform_sample(env.rng, ranges[name], shape, self.device)
            return
        elif name in ("hand_size", "cube_size"):
            # The source samples one scale per environment for all selected
            # geoms and all three axes, preserving object/hand proportions.
            factor = uniform_sample(env.rng, ranges[name], (env.num_envs, 1, 1), self.device)
            self.ratios[name] = factor.reshape(env.num_envs, 1)
        elif name.startswith("solref_") or name.startswith("solimp_"):
            # ``solref`` has the same ranges for both groups; ``solimp`` uses
            # the wider 2--5 mm soft zone on palm/thumb pads and 1--2 mm on
            # the remaining fingers, matching the source task's split event
            # terms.
            base_name = name.split("_", 1)[0]
            range_name = name
            if base_name == "solref":
                range_name = "solref"
            param_ranges = ranges[range_name]
            factor = torch.ones(shape, device=self.device)
            for component, bounds in param_ranges.items():
                factor[..., int(component)] = uniform_sample(
                    env.rng, bounds, (env.num_envs, 1), self.device
                )
        else:
            bounds = ranges[name]
            factor_shape = shape[:-1] if default.ndim == 3 else shape
            if name == "damping":
                factor = torch.exp(
                    uniform_sample(
                        env.rng,
                        (math.log(float(bounds[0])), math.log(float(bounds[1]))),
                        factor_shape,
                        self.device,
                    )
                )
            else:
                factor = uniform_sample(env.rng, bounds, factor_shape, self.device)
            self.ratios[name] = factor.reshape(env.num_envs, -1).float()
            if default.ndim == 3:
                factor = factor[..., None]
        self.samples[name] = default * factor

    def __call__(self, env, env_ids: torch.Tensor, ranges: dict, total_steps: int):
        del total_steps
        if not self.samples:
            for name, (_, _, default, _, _) in self.bindings.items():
                self._sample(
                    env,
                    name,
                    torch.as_tensor(np.asarray(default).copy(), device=self.device, dtype=torch.float32),
                    ranges,
                )
            for name, default in (("kp", self.kp), ("kd", self.kd)):
                default = default.to(device=self.device, dtype=torch.float32)
                lo, hi = ranges[name]
                scale = torch.exp(
                    uniform_sample(
                        env.rng,
                        (torch.log(torch.as_tensor(lo)), torch.log(torch.as_tensor(hi))),
                        (env.num_envs, default.shape[-1]),
                        self.device,
                    )
                )
                self.ratios[name] = scale.reshape(env.num_envs, -1)
                self.samples[name] = default * scale
            self.samples["encoder"] = uniform_sample(
                env.rng, ranges["encoder"], (env.num_envs, self.robot.num_joints), self.device
            )

        publish_fixed_model(env, self.bindings, self.samples, self.gain_ids)
        self.robot.data.encoder_bias_tensor[env_ids] = self.samples["encoder"][env_ids]
        friction = self.samples["friction"][..., 0]
        default_friction = torch.as_tensor(
            np.asarray(self.bindings["friction"][2]).copy(), device=self.device, dtype=torch.float32
        )
        default_friction = default_friction[0] if default_friction.ndim >= 3 else default_friction
        # Source dr_params_privileged averages all entity geoms. Visual
        # geoms keep their default friction/size but still contribute ratios.
        friction_ratio = (
            (friction / default_friction[..., 0].clamp_min(1e-8)).sum(-1)
            + self._robot_geom_count - friction.shape[1]
        ) / self._robot_geom_count
        size_ratio = (
            self.ratios["cube_size"].mean(-1) + self._object_geom_count - 1
        ) / self._object_geom_count
        ratios = torch.stack((
            friction_ratio, self.ratios["cube_mass"].mean(-1),
            self.ratios["kp"].mean(-1), self.ratios["kd"].mean(-1),
            torch.full_like(size_ratio, self._damping_ratio), size_ratio,
        ), dim=-1)
        task(env).dr_ratios[env_ids] = ratios[env_ids]


class VelocityDisturbance(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(env)
        self.cube = env.scene["object"]
        self.cube.bind_root_linear_velocity_delta(term_name="wuji_velocity_disturbance")

    def __call__(
        self,
        env,
        env_ids,
        speed_range: tuple,
        min_episode_s: float,
        total_steps: int,
        warmup_frac: float = 0.05,
        rampup_frac: float = 0.80,
        adaptive_curriculum_term: str = "adaptive_episode",
        use_time_ramp: bool = True,
    ):
        state = task(env)
        mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        if env_ids is not None:
            mask = torch.zeros_like(mask)
            mask[env_ids] = True
        state.perturbation[mask] = 0
        state.force_direction[mask] = 0
        mask &= env.episode_length_buf * env.step_dt >= min_episode_s
        if not bool(mask.any()):
            return
        direction = env.rng.standard_normal((int(mask.sum()), 3)).to(env.device)
        direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(
            1e-9
        )
        if use_time_ramp:
            # ``max_common_steps`` is the source runner's preferred total;
            # ``total_steps`` keeps standalone UniLab configs deterministic.
            total = float(getattr(env, "max_common_steps", total_steps))
            step = float(getattr(env, "common_step_counter", 0))
            warmup = float(warmup_frac) * max(total, 1.0)
            rampup = float(rampup_frac) * max(total, 1.0)
            if step <= warmup:
                progress = 0.0
            elif step >= rampup:
                progress = 1.0
            else:
                progress = (step - warmup) / max(rampup - warmup, 1.0)
        else:
            progress = 1.0

        adaptive_scale = 1.0
        if adaptive_curriculum_term:
            manager = getattr(env, "curriculum_manager", None)
            curriculum_state = getattr(manager, "_curriculum_state", {})
            value = curriculum_state.get(adaptive_curriculum_term)
            if isinstance(value, dict):
                value = value.get("value", 1.0)
            if isinstance(value, torch.Tensor):
                value = value.mean().item() if value.numel() else 1.0
            if isinstance(value, (int, float)):
                adaptive_scale = float(value)
        ramp = progress * adaptive_scale
        curr_max = max(speed_range[0], speed_range[0] + (speed_range[1] - speed_range[0]) * ramp)
        speed = uniform_sample(
            env.rng,
            (speed_range[0], curr_max),
            (int(mask.sum()), 1),
            env.device,
        )
        delta = direction * speed
        self.cube.apply_root_linear_velocity_delta_to_sim(delta, env_ids=mask.nonzero().flatten())
        rows = mask.nonzero().flatten()
        state.perturbation[rows, :3] = delta
        state.perturbation[rows, 3:] = 0
        state.force_direction[rows] = direction

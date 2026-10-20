# pyright: reportAttributeAccessIssue=false
# Adapted from wuji-technology/wuji-mjlab 9410a3a, mdp/event_impl/.
# Copyright 2026 Wuji Technology Co., Ltd. Apache-2.0.
"""Task-owned reset and randomization through declared Entity capabilities."""

import math

import numpy as np
import torch
from unilab.managers import ManagerTermBase

from .commands import task
from .math import multiply, random_quaternions, soft_limits, uniform_sample
from .reset_bridge import (
    bind_mocap_reset,
    reset_backend,
    write_geom_sizes,
    write_mocap_reset,
)


class ResetWuji(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(env)
        self.robot = env.scene["robot"]
        self.cube = env.scene["object"]
        self.pose = bind_mocap_reset(self.robot, "robot/right_palm_link", term_name="wuji_reset")
        self.default_pose = torch.as_tensor(
            np.asarray(self.pose.default_pose).copy(), device=env.device, dtype=torch.float32
        )
        self.default = self.robot.data.default_joint_pos(env.device)[0]
        self.limits = soft_limits(
            self.robot.data.soft_joint_pos_limits(env.device),
            cfg.params["soft_limit_factor"],
        )
        self.offset_ids = self.robot.find_joints("robot/right_finger[1-5]_joint[134]")[0]

    def reset(
        self,
        env,
        reset_mask: torch.Tensor,
        soft_limit_factor: float,
        joint_noise: tuple,
        cube_noise: float,
        wrist_pitch: tuple,
    ):
        del soft_limit_factor
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
        write_mocap_reset(self.pose, reset_mask, poses)
        state = self.cube.data.default_root_state(env.device).clone()
        state[:, :3] += uniform_sample(
            env.rng, (-cube_noise, cube_noise), (env.num_envs, 3), env.device
        )
        state[:, 3:7] = random_quaternions(env.rng, env.num_envs, device=env.device)
        state[:, 7:] = 0
        self.cube.write_root_state_tensor_to_sim(state, reset_mask)


class FixedRandomization(ManagerTermBase):
    """Sample per-world parameters once, then replay them on selected resets."""

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
        capsule_ids = r.find_geoms("robot/right_finger[1-5]_link2_col")[0]
        hand_ids = r.find_bodies("robot/.*")[0]
        self.bindings["friction"] = (
            r,
            *r.bind_geom_friction_write(term_name=term),
            r.write_geom_friction_to_sim,
            "geom_ids",
        )
        self.bindings["hand_size"] = (
            r,
            *r.bind_geom_size_write(capsule_ids, term_name=term),
            r.write_geom_size_to_sim,
            "geom_ids",
        )
        self.bindings["solref"] = (
            r,
            *r.bind_geom_solref_write(term_name=term),
            r.write_geom_solref_to_sim,
            "geom_ids",
        )
        self.bindings["solimp"] = (
            r,
            *r.bind_geom_solimp_write(term_name=term),
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
        self.bindings["damping"] = (
            r,
            *r.bind_joint_damping_write(term_name=term),
            r.write_joint_damping_to_sim,
            "joint_ids",
        )
        self.bindings["armature"] = (
            r,
            *r.bind_joint_armature_write(term_name=term),
            r.write_joint_armature_to_sim,
            "joint_ids",
        )
        self.bindings["frictionloss"] = (
            r,
            *r.bind_joint_frictionloss_write(term_name=term),
            r.write_joint_frictionloss_to_sim,
            "joint_ids",
        )
        self.gain_ids, self.kp, self.kd = r.bind_actuator_gain_write(term_name=term)

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
        elif name in ("solref", "solimp"):
            factor = torch.ones(shape, device=self.device)
            for component, bounds in ranges[name].items():
                factor[..., int(component)] = uniform_sample(
                    env.rng, bounds, shape[:-1], self.device
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

    def reset(self, env, reset_mask: torch.Tensor, ranges: dict, total_steps: int):
        del total_steps
        if not self.samples:
            for name, (_, _, default, _, _) in self.bindings.items():
                self._sample(
                    env,
                    name,
                    torch.as_tensor(np.asarray(default).copy(), device=self.device),
                    ranges,
                )
            for name, default in (("kp", self.kp), ("kd", self.kd)):
                default = default.to(self.device)
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

        values = {}
        for name, (_, _, _, _, _) in self.bindings.items():
            values[name] = self.samples[name]

        selected_ids = reset_mask.nonzero(as_tuple=False).flatten()
        selected_rows = {name: value[selected_ids] for name, value in values.items()}
        # Every field except geom size composes through its public Entity write.
        # mjwarp geom sizes are host-only, so their selected rows cross the one
        # documented backend boundary in reset_bridge.
        for name, (_, model_ids, _, write, selector) in self.bindings.items():
            if name not in ("hand_size", "cube_size"):
                write(
                    selected_rows[name],
                    env_ids=selected_ids,
                    **{selector: model_ids},
                    term_name="wuji_randomize",
                )
        backend = reset_backend(self.robot)
        for name in ("hand_size", "cube_size"):
            owner = self.robot if name == "hand_size" else self.cube
            local_ids, model_ids = self.bindings[name][1:3]
            backend_columns = np.asarray(owner._geom_ids, dtype=np.intp)[
                np.asarray(local_ids, dtype=np.intp)
            ]
            write_geom_sizes(
                backend,
                model_ids=model_ids,
                backend_geom_ids=backend_columns,
                reset_mask=reset_mask,
                values=self.samples[name],
            )

        # `write_actuator_gains_tensor_mask` publishes a full-width table and
        # therefore requires the exact transaction mask. In an event reset that
        # mask is subset-only, so use the selected-row facade instead.
        self.robot.write_actuator_gains_to_sim(
            self.samples["kp"][selected_ids],
            self.samples["kd"][selected_ids],
            actuator_ids=self.gain_ids,
            env_ids=selected_ids,
            term_name="wuji_randomize",
        )
        columns = torch.arange(self.robot.num_joints, device=self.device)
        self.robot.data.update_encoder_bias_tensor_mask(
            reset_mask, columns, self.samples["encoder"]
        )
        ratios = torch.stack(
            [
                self.ratios[name].mean(dim=1)
                for name in ("friction", "cube_mass", "kp", "kd", "damping", "cube_size")
            ],
            dim=-1,
        )
        task(env).dr_ratios.copy_(torch.where(reset_mask[:, None], ratios, task(env).dr_ratios))


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
        direction = env.rng.normal(size=(int(mask.sum()), 3)).to(env.device)
        direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(
            1e-9
        )
        progress = torch.clamp(
            (env.common_step_counter / max(total_steps, 1) - 0.05) / 0.8, 0.0, 1.0
        )
        curr_max = speed_range[0] + (speed_range[1] - speed_range[0]) * progress * state.adaptive
        speed = uniform_sample(
            env.rng,
            (speed_range[0], torch.maximum(torch.tensor(speed_range[0]), curr_max)),
            (int(mask.sum()), 1),
            env.device,
        )
        delta = direction * speed
        self.cube.apply_root_linear_velocity_delta_to_sim(delta, env_ids=mask.nonzero().flatten())
        rows = mask.nonzero().flatten()
        state.perturbation[rows, :3] = delta
        state.perturbation[rows, 3:] = 0
        state.force_direction[rows] = direction

# pyright: reportAttributeAccessIssue=false
# Adapted from wuji-technology/wuji-mjlab 9410a3a, mdp/event_impl/.
# Copyright 2026 Wuji Technology Co., Ltd. Apache-2.0.
"""Task-owned reset and randomization through declared Entity capabilities."""

from importlib import import_module
from typing import Any

import numpy as np
from unilab.managers import ManagerTermBase

from .commands import task
from .math import multiply, random_quaternions, soft_limits

mujoco: Any = import_module("mujoco")


class ResetWuji(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(env)
        self.robot = env.scene["robot"]
        self.cube = env.scene["object"]
        self.pose = self.robot.bind_mocap_pose_write(
            "robot/right_palm_link", term_name="wuji_reset"
        )
        self.default = self.robot.data.default_joint_pos.copy()
        self.limits = soft_limits(
            self.robot.data.soft_joint_pos_limits, cfg.params["soft_limit_factor"]
        )
        self.offset_ids = self.robot.find_joints("robot/right_finger[1-5]_joint[134]")[0]

    def __call__(
        self,
        env,
        env_ids,
        soft_limit_factor: float,
        joint_noise: tuple,
        cube_noise: float,
        wrist_pitch: tuple,
    ):
        del soft_limit_factor
        ids = np.arange(env.num_envs) if env_ids is None else env_ids
        if not len(ids):
            return
        joint = self.default[ids].copy()
        joint[:, self.offset_ids] += env.rng.uniform(
            *joint_noise, size=(len(ids), len(self.offset_ids))
        )
        joint = np.clip(joint, self.limits[:, 0], self.limits[:, 1])
        self.robot.write_joint_state_to_sim(joint, np.zeros_like(joint), env_ids=ids)
        poses = np.tile(self.pose, (len(ids), 1))
        angle = env.rng.uniform(*wrist_pitch, len(ids))
        rotation = np.stack(
            (np.cos(angle / 2), np.zeros(len(ids)), np.sin(angle / 2), np.zeros(len(ids))), axis=-1
        )
        poses[:, 3:] = multiply(poses[:, 3:], rotation)
        self.robot.write_mocap_pose_to_sim(
            poses.astype(np.float32), env_ids=ids, term_name="wuji_reset"
        )
        state = self.cube.data.default_root_state[ids].copy()
        state[:, :3] += env.rng.uniform(-cube_noise, cube_noise, (len(ids), 3))
        state[:, 3:7] = random_quaternions(env.rng, len(ids))
        state[:, 7:] = 0
        self.cube.write_root_state_to_sim(state, env_ids=ids)


class FixedRandomization(ManagerTermBase):
    """Sample per-world parameters once, then replay them on selected resets.

    Cached values are the values written to the backend, not inferred defaults.
    The cache is also the critic DR information owner.
    """

    def __init__(self, cfg, env):
        super().__init__(env)
        self.robot = env.scene["robot"]
        self.cube = env.scene["object"]
        model = mujoco.MjModel.from_xml_path(str(env.cfg.scene.model_file))
        self.bindings = {}
        self.samples = {}
        self.ratios = {}
        term = "wuji_randomize"
        # All resolution, XML inspection and default snapshots are cold-path.
        r, o = self.robot, self.cube
        capsule_ids = r.find_geoms("robot/right_finger[1-5]_link[23]_col")[0]
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
            *o.bind_body_inertia_write(
                default=model.body_inertia, default_mass=model.body_mass, term_name=term
            ),
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
            *r.bind_body_inertia_write(
                hand_ids, default=model.body_inertia, default_mass=model.body_mass, term_name=term
            ),
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

    def __call__(self, env, env_ids, ranges: dict, total_steps: int):
        ids = np.arange(env.num_envs) if env_ids is None else env_ids
        if not len(ids):
            return
        if not self.samples:
            for name, (_, _, default, _, _) in self.bindings.items():
                shape = (env.num_envs, *default.shape)
                if name == "cube_inertia":
                    factor = self.ratios["cube_mass"][..., None]
                elif name == "cube_com":
                    self.samples[name] = default[None] + env.rng.uniform(*ranges[name], size=shape)
                    continue
                elif name in ("solref", "solimp"):
                    factor = np.ones(shape, dtype=np.float32)
                    for component, bounds in ranges[name].items():
                        factor[..., int(component)] = env.rng.uniform(*bounds, size=shape[:-1])
                else:
                    bounds = ranges[name]
                    factor_shape = shape[:-1] if default.ndim == 2 else shape
                    if name == "damping":
                        factor = np.exp(
                            env.rng.uniform(np.log(bounds[0]), np.log(bounds[1]), size=factor_shape)
                        )
                    else:
                        factor = env.rng.uniform(*bounds, size=factor_shape)
                    self.ratios[name] = np.asarray(factor, dtype=np.float32)
                    if default.ndim == 2:
                        factor = factor[..., None]
                self.samples[name] = np.asarray(default[None] * factor, dtype=np.float32)
            for name, default in (("kp", self.kp), ("kd", self.kd)):
                lo, hi = ranges[name]
                scale = np.exp(
                    env.rng.uniform(np.log(lo), np.log(hi), size=(env.num_envs, default.size))
                )
                self.ratios[name] = scale.astype(np.float32)
                self.samples[name] = (default[None] * scale).astype(np.float32)
            self.samples["encoder"] = env.rng.uniform(
                *ranges["encoder"], size=self.robot.data.joint_pos.shape
            ).astype(np.float32)
        progress = np.clip((env.common_step_counter / max(total_steps, 1) - 0.05) / 0.60, 0, 1)
        friction_scale = 1 - 0.5 * progress
        size_scale = 1 - 0.05 * progress
        for name, (_, local_ids, _, write, selector) in self.bindings.items():
            values = self.samples[name][ids]
            if name == "friction":
                values = values * friction_scale
            elif name == "hand_size":
                values = values * size_scale
            write(values, env_ids=ids, term_name="wuji_randomize", **{selector: local_ids})
        self.robot.write_actuator_gains_to_sim(
            self.samples["kp"][ids],
            self.samples["kd"][ids],
            actuator_ids=self.gain_ids,
            env_ids=ids,
            term_name="wuji_randomize",
        )
        self.robot.data.encoder_bias[ids] = self.samples["encoder"][ids]
        values = [
            self.ratios[name][ids].reshape(len(ids), -1).mean(axis=1)
            for name in ("friction", "cube_mass", "kp", "kd", "damping", "cube_size")
        ]
        values[0] *= friction_scale
        task(env).dr_ratios[ids] = np.stack(values, axis=-1)


class VelocityDisturbance(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(env)
        self.cube = env.scene["object"]
        self.cube.bind_root_linear_velocity_delta(term_name="wuji_velocity_disturbance")

    def __call__(self, env, env_ids, speed_range: tuple, min_episode_s: float, total_steps: int):
        ids = np.arange(env.num_envs) if env_ids is None else env_ids
        ids = ids[env.episode_length_buf[ids] * env.step_dt >= min_episode_s]
        if not len(ids):
            return
        state = task(env)
        direction = env.rng.normal(size=(len(ids), 3))
        direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-9)
        progress = np.clip((env.common_step_counter / max(total_steps, 1) - 0.05) / 0.8, 0, 1)
        speed = (speed_range[0] + progress * (speed_range[1] - speed_range[0])) * state.adaptive
        delta = np.asarray(direction * speed, dtype=np.float32)
        self.cube.apply_root_linear_velocity_delta_to_sim(delta, env_ids=ids)
        state.perturbation[ids, :3] = delta
        state.perturbation[ids, 3:] = 0

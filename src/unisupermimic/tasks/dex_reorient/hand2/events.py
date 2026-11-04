# pyright: reportAttributeAccessIssue=false
# Adapted from wuji-technology/wuji-mjlab 9410a3a, mdp/event_impl/.
# Copyright 2026 Wuji Technology Co., Ltd. Apache-2.0.
"""Task-owned reset and randomization through declared Entity capabilities."""

import math

import numpy as np
import torch
from unilab.base.reset_selectors import ResetMaskBuffer
from unilab.managers import ManagerTermBase
from unilab.managers.reset_owner import ResetOwner

from .commands import task
from .math import multiply, random_quaternions, rotate, soft_limits, uniform_sample
from .reset_bridge import (
    bind_mocap_reset,
    reset_backend,
    write_geom_sizes,
    write_mocap_reset,
)
from .scene import _CALIBRATED_MOUNT_POS


class ResetWuji(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(env)
        self.robot = env.scene["robot"]
        self.cube = env.scene["object"]
        # mjwarp exposes the packed mocap reset transaction used by the
        # source task.  MuJoCo's host backend may not expose that private
        # transaction; its normal entity reset still restores the configured
        # root state, so keep the bridge optional there.
        try:
            self.pose = bind_mocap_reset(self.robot, "robot/right_palm_link", term_name="wuji_reset")
        except (AttributeError, RuntimeError):
            self.pose = None
        self.default_pose = (
            torch.as_tensor(np.asarray(self.pose.default_pose).copy(), device=env.device, dtype=torch.float32)
            if self.pose is not None
            else None
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
        self._pending_mocap = None

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
        if env_ids.numel() == 0:
            return
        reset_mask = self._reset_mask.scatter(env_ids)
        joint = self.default.repeat(env.num_envs, 1)
        selected = torch.tensor(self.offset_ids, device=joint.device, dtype=torch.long)
        joint[:, selected] += uniform_sample(
            env.rng, joint_noise, (env.num_envs, selected.numel()), env.device
        )
        joint = torch.clamp(joint, self.limits[None, :, 0], self.limits[None, :, 1])
        self.robot.write_joint_state_to_sim_mask(joint, torch.zeros_like(joint), reset_mask)
        if self.pose is not None and self.default_pose is not None:
            poses = self.default_pose.repeat(env.num_envs, 1)
            angle = uniform_sample(env.rng, wrist_pitch, (env.num_envs,), env.device)
            zeros = torch.zeros_like(angle)
            rotation = torch.stack((torch.cos(angle / 2), zeros, torch.sin(angle / 2), zeros), dim=-1)
            # Hand2 pitches about world Y through the calibrated mount.
            # The default palm maps local Y onto world Z, so right-multiplying
            # would spin about the palm normal instead of tilting the wrist.
            pivot = poses.new_tensor(_CALIBRATED_MOUNT_POS).expand(env.num_envs, -1)
            center = poses[:, :3] + rotate(poses[:, 3:], pivot)
            poses[:, :3] = center + rotate(rotation, poses[:, :3] - center)
            poses[:, 3:] = multiply(rotation, poses[:, 3:])
            # MJWarp's subsequent packed qpos/qvel commit calls reset_data,
            # which resets mocap too. Publish only after that commit, before
            # the Manager refreshes poses, goals and reset observations.
            self._pending_mocap = (env_ids.clone(), poses)
        state = self.cube.data.default_root_state(env.device).clone()
        if state.ndim >= 2 and state.shape[0] == 1 and env.num_envs != 1:
            state = state.repeat(env.num_envs, 1)
        state[:, :3] += uniform_sample(
            env.rng, (-cube_noise, cube_noise), (env.num_envs, 3), env.device
        )
        state[:, 3:7] = random_quaternions(env.rng, env.num_envs, device=env.device)
        state[:, 7:] = 0
        self.cube.write_root_state_tensor_to_sim(state, reset_mask)

    def commit_mocap_reset(self) -> None:
        """Flush the selected pose rows after the ordinary reset commit."""
        if self._pending_mocap is not None:
            env_ids, poses = self._pending_mocap
            write_mocap_reset(self.pose, env_ids, poses)
            self._pending_mocap = None


class CommitHand2MocapReset(ResetOwner):
    """Use UniLab's post-commit hook without replacing any Manager resets."""

    def reset_transaction(self, env_ids: torch.Tensor) -> dict[str, float]:
        return {}

    def reset_committed(self, env_ids: torch.Tensor) -> dict[str, float]:
        for name in self._env.event_manager.active_terms.get("reset", ()):
            term = self._env.event_manager.get_term_cfg(name).func
            if isinstance(term, ResetWuji):
                # Binding.write uploads mocap and runs forward at the final
                # joint/cube state. The Manager refreshes its read plan next.
                term.commit_mocap_reset()
        return {}


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
        # Keep the geometry groups identical to the two source mjlab startup
        # events.  Contact solver parameters are randomized only on the soft
        # palm/thumb pads and on fingers 2--5.  In particular, link1 capsules
        # and visual/mesh-disabled geoms must not receive this randomization.
        capsule_ids = r.find_geoms("robot/right_finger[1-5]_link[23]_col")[0]
        palm_contact_ids = r.find_geoms(
            (
                "robot/right_palm_collision",
                "robot/right_finger1_link[2-4]_col",
                "robot/right_finger1_tip_sensor_col",
            )
        )[0]
        finger_contact_ids = r.find_geoms(
            (
                "robot/right_finger[2-5]_link[2-4]_col",
                "robot/right_finger[2-5]_tip_sensor_col",
            )
        )[0]
        friction_ids = r.find_geoms(
            ("robot/.*palm_.*", "robot/.*finger.*_col")
        )[0]
        hand_ids = r.find_bodies("robot/.*")[0]
        # The visual cube stays fixed in the reference recipe. Only the
        # collision box participates in the physical object-size variation.
        cube_collision_ids = o.find_geoms("object/cube")[0]
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
            *o.bind_geom_size_write(cube_collision_ids, term_name=term),
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
        # Sampled tables are immutable after the first reset, so a geom-size row
        # only needs the host-only backend write once per env.  Track published
        # rows and skip the (host-computed, full-width-uploading) rewrite.
        self._geom_size_applied = {
            name: torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
            for name in ("hand_size", "cube_size")
        }

    def _sample(self, env, name: str, default: torch.Tensor, ranges: dict):
        # 1.4 model-field bindings return immutable one-row/default tables for
        # NumPy-backed fields. They are model metadata, not per-environment state.
        default = default[0] if default.ndim >= 3 else default
        shape = (env.num_envs, *default.shape)
        if name == "friction":
            # mjlab's geom_friction defaults to tangential friction (axis 0).
            # Torsional/rolling coefficients stay at their model defaults.
            factor = torch.ones(shape, device=self.device)
            factor[..., 0] = uniform_sample(
                env.rng, ranges[name], shape[:-1], self.device
            )
        elif name in ("hand_size", "cube_size"):
            # One isotropic factor per environment, shared by all selected
            # geoms and axes, matches randomize_geom_size_uniform upstream.
            # Independent axes would introduce hidden shape variables that
            # the single size summary in xi cannot describe.
            factor = uniform_sample(
                env.rng, ranges[name], (env.num_envs,) + (1,) * default.ndim, self.device
            )
            self.ratios[name] = factor.reshape(env.num_envs, 1)
        elif name == "cube_inertia":
            factor = self.ratios["cube_mass"][..., None]
        elif name == "cube_com":
            self.samples[name] = default + uniform_sample(env.rng, ranges[name], shape, self.device)
            return
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
                # Each parameter shares one factor across its contact group
                # in each environment; palm/thumb and other fingers are
                # still sampled independently by their separate bindings.
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
        if name == "friction":
            self.ratios[name] = (
                self.samples[name][..., 0] / default[..., 0].clamp_min(1.0e-8)
            ).reshape(env.num_envs, -1).float()
        if name == "damping":
            # The privileged source observation measures the actual ratio,
            # not the sampled multiplier. Scaling a default zero damping
            # leaves it at zero, so its context contribution must be zero.
            self.ratios[name] = (
                self.samples[name] / default.clamp_min(1.0e-8)
            ).reshape(env.num_envs, -1).float()

    def __call__(self, env, env_ids: torch.Tensor, ranges: dict, total_steps: int):
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

        selected_rows = {name: value[env_ids] for name, value in values.items()}
        # Every field except geom size composes through its public Entity write.
        # mjwarp geom sizes are host-only, so their selected rows cross the one
        # documented backend boundary in reset_bridge.
        for name, (_, model_ids, _, write, selector) in self.bindings.items():
            if name not in ("hand_size", "cube_size"):
                write(
                    selected_rows[name],
                    env_ids=env_ids,
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
            applied = self._geom_size_applied[name]
            pending_ids = env_ids[~applied[env_ids]]
            if not pending_ids.numel():
                continue
            write_geom_sizes(
                backend,
                backend_geom_ids=backend_columns,
                reset_rows=pending_ids.detach().cpu().numpy(),
                values=self.samples[name],
            )
            applied[pending_ids] = True

        # `write_actuator_gains_tensor_mask` publishes a full-width table and
        # therefore requires the exact transaction mask. In an event reset that
        # mask is subset-only, so use the selected-row facade instead.
        self.robot.write_actuator_gains_to_sim(
            self.samples["kp"][env_ids],
            self.samples["kd"][env_ids],
            actuator_ids=self.gain_ids,
            env_ids=env_ids,
            term_name="wuji_randomize",
        )
        self.robot.data.encoder_bias_tensor[env_ids] = self.samples["encoder"][env_ids]
        ratios = torch.stack(
            [
                self.ratios[name].mean(dim=1)
                for name in ("friction", "cube_mass", "kp", "kd", "damping", "cube_size")
            ],
            dim=-1,
        )
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

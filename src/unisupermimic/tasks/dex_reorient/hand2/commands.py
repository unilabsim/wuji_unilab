"""Cube-goal command state for Wuji Hand 2.

This is the UniLab counterpart of the mjlab reorientation command.  It owns
the goal quaternion and all task state used by observations, rewards, and
termination terms; backend reads are limited to the Manager read plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import mujoco
import torch
from unilab.base.entity import Entity
from unilab.managers import CommandTerm, CommandTermCfg

from unisupermimic.rl.algos.supermimic.geometry import matmul, matvec, quat_to_mat

from .analytic import pack_state
from .math import angle_error, conjugate, multiply, random_quaternions, rotate
from .scene import materialize_scene


@dataclass(kw_only=True)
class Hand2ReorientCommandCfg(CommandTermCfg):
    entity_name: str = "robot"
    object_name: str = "object"
    success_threshold: float = 0.2
    success_hold_steps: int = 5
    goal_switch_delay: int = 20

    def build(self, env):
        return Hand2ReorientCommand(self, env)


class Hand2ReorientCommand(CommandTerm):
    cfg: Hand2ReorientCommandCfg

    def __init__(self, cfg: Hand2ReorientCommandCfg, env):
        super().__init__(cfg, env)
        self.robot = cast(Entity, env.scene[cfg.entity_name])
        self.object = cast(Entity, env.scene[cfg.object_name])
        self.palm_name = "robot/right_palm_link"
        self.tag_name = "robot/right_wrist_tag_frame"
        self.cube_name = "object/cube"
        self.tip_names = tuple(f"robot/right_finger{i}_link4" for i in range(1, 6))
        # Register the MJCF sensors with UniLab's scene read compiler.  The
        # backend presents these under the robot entity even though the MJCF
        # sensor elements are attached at the composed world level.
        self._sensor_names = (
            *(f"tip_contact_{i}_{part}" for i in range(1, 6) for part in ("link4", "tip")),
            "palm_contact_palm",
            *(f"palm_contact_finger{finger}_link{link}" for finger in range(1, 6) for link in (1, 2)),
            *(f"distal_contact_{i}_{part}" for i in range(1, 6) for part in ("link3", "link4", "tip")),
            *(f"finger_collision_{i}" for i in range(25)),
            *(f"actuator_force_{i}" for i in range(20)),
            *(f"tip_velocity_{i}" for i in range(1, 6)),
        )
        try:
            env.scene.bind_sensor_data(self._sensor_names)
        except (AttributeError, KeyError, RuntimeError, ValueError):
            # A backend may not expose the optional sensor packet.  The
            # contact head remains shape-stable and the runner can train its
            # state model from the physical transition alone.
            self._sensor_names = ()
        self.goal = torch.zeros((self.num_envs, 4), device=self._device)
        self.goal[:, 0] = 1.0
        self.hold = torch.zeros(self.num_envs, dtype=torch.int32, device=self._device)
        self.window = torch.zeros_like(self.hold)
        self.goal_count = torch.zeros_like(self.hold)
        self.goal_timer = torch.zeros_like(self.hold)
        self.success = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        self.within = torch.zeros_like(self.success)
        # Keep the three reward-phase counters from the source command
        # separate.  ``window`` is the live elapsed success-window counter;
        # the replay memory exposes it as a binary flag plus ``window_timer``.
        self.reward_window_timer = torch.zeros_like(self.hold)
        self.reward_hold_counter = torch.zeros_like(self.hold)
        self.reward_hold_counter_snapshot = torch.zeros_like(self.hold)
        self.outside = torch.zeros_like(self.success)
        self.cage_counter = torch.zeros(self.num_envs, device=self._device)
        self.perturbation = torch.zeros((self.num_envs, 6), device=self._device)
        self.force_direction = torch.zeros((self.num_envs, 3), device=self._device)
        self.dr_ratios = torch.ones((self.num_envs, 6), device=self._device)
        self.difficulty = 0.0
        self.adaptive = 0.05
        self.metrics = {
            name: torch.zeros(self.num_envs, device=self._device)
            for name in (
                "goal_reach_count", "ori_error", "hold_counter", "goal_timer",
                "in_success_window", "window_timer",
            )
        }
        self._pose_stale = True
        self._palm_pos = torch.zeros((self.num_envs, 3), device=self._device)
        self._palm_quat = torch.zeros((self.num_envs, 4), device=self._device)
        self._tag_pos = torch.zeros((self.num_envs, 3), device=self._device)
        self._tag_quat = torch.zeros((self.num_envs, 4), device=self._device)
        self._cube_pos = torch.zeros((self.num_envs, 3), device=self._device)
        self._cube_quat = torch.zeros((self.num_envs, 4), device=self._device)
        # CommandManager samples before UniLab refreshes the packed reset read
        # plan.  Keep the local goal sample so ``refresh_reset`` can compose it
        # with the newly reset tag pose instead of using the previous episode's
        # wrist orientation.
        self._pending_goal_local = torch.zeros((self.num_envs, 4), device=self._device)
        self._pending_goal_local[:, 0] = 1.0
        self._pending_goal_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        self._read_plan = None
        # Construct the entity-free differentiable adapter once the command
        # has its robot defaults.  Import lazily to avoid the mdp -> rewards ->
        # commands module cycle during package registration.
        from .mdp import Hand2SuperMimicMDP
        from .scene import _FORCE_LIMITS, _GAINS

        default = self.robot.data.default_joint_pos(self._device)
        default = default[0] if default.ndim >= 2 else default
        default_velocity = self.robot.data.default_joint_vel(self._device)
        default_velocity = default_velocity[0] if default_velocity.ndim >= 2 else default_velocity
        limits = self.robot.data.soft_joint_pos_limits(self._device)
        limits = limits[0] if limits.ndim >= 3 else limits
        term_cfg = getattr(getattr(env, "_cfg", None), "terminations", {}).get("cage_drop")
        term_params = getattr(term_cfg, "params", {}) if term_cfg is not None else {}
        reward_cfgs = getattr(getattr(env, "_cfg", None), "rewards", {})
        action_cfg = getattr(getattr(env, "_cfg", None), "actions", {}).get("joint_pos")
        hold_cfg = reward_cfgs.get("hold_escalation") or reward_cfgs.get("hold")
        hold_weight = float(getattr(hold_cfg, "weight", 4.694 if float(getattr(env, "step_dt", 0.02)) <= 0.02 else 11.4))
        def reward_weight(name: str, default: float) -> float:
            term = reward_cfgs.get(name)
            return float(getattr(term, "weight", default))
        self.mdp = Hand2SuperMimicMDP(
            default,
            limits[:, 0],
            limits[:, 1],
            dt=float(getattr(env, "step_dt", 0.02)),
            default_joint_vel=default_velocity,
            hold_steps=cfg.success_hold_steps,
            switch_steps=cfg.goal_switch_delay,
            max_outside=float(term_params.get("max_outside_steps", 25)),
            max_count=float(term_params.get("max_count", 37.5)),
            hold_weight=hold_weight,
            action_rate_weight=abs(reward_weight("action_rate", -1.0)),
            action_hf_weight=abs(reward_weight("action_hf", -30.0)),
            torque_weight=abs(reward_weight("torque", -24.0)),
            cage_weight=abs(reward_weight("cage_escape", -500.0)),
            finger_collision_weight=abs(reward_weight("finger_collision", -1.0)),
            palm_detach_weight=reward_weight("palm_detach", 0.5),
            action_alpha=float(getattr(action_cfg, "ema_alpha", 0.5)),
            warmup_time_s=float(getattr(action_cfg, "warmup_time_s", 0.4)),
        )
        # Keep the differentiable torque proxy on the calibrated source
        # actuator gains.  The live MuJoCo reward still uses the measured
        # actuator-force sensors when they are available.
        self.mdp.kp_default = torch.tensor(
            [gain[0] for gain in _GAINS], device=self._device, dtype=torch.float32
        )
        self.mdp.kd_default = torch.tensor(
            [-gain[1] for gain in _GAINS], device=self._device, dtype=torch.float32
        )
        effort_limit = getattr(self.robot.data, "actuator_effort_limit", None)
        if effort_limit is not None:
            effort_limit = torch.as_tensor(effort_limit, device=self._device, dtype=torch.float32)
            if effort_limit.ndim >= 2:
                effort_limit = effort_limit[0]
            self.mdp.effort_limit = effort_limit.reshape(-1)
        self.mdp.force_limit = torch.tensor(
            _FORCE_LIMITS, device=self._device, dtype=torch.float32
        )
        # Build the same differentiable FK used by the source learner for
        # imagined cage and fingertip terms.  The runtime entity read plan is
        # intentionally not used here: imagination must remain backend-free.
        self.scene_path = materialize_scene()
        from .kinematics import Hand2Kinematics

        body_names = (
            self.palm_name,
            *(f"robot/right_finger{finger}_link{link}" for finger in range(1, 6) for link in range(1, 5)),
        )
        joint_names = tuple(
            f"robot/right_finger{finger}_joint{joint}"
            for finger in range(1, 6)
            for joint in range(1, 5)
        )
        model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.mdp.kinematics = Hand2Kinematics(
            model,
            body_names,
            joint_names,
            tuple(f"robot/right_finger{finger}_tip" for finger in range(1, 6)),
            self._device,
        )
        self.mdp.cage_up_axis = int(term_params.get("up_axis", 1))

    @property
    def command(self) -> torch.Tensor:
        return self.goal

    @property
    def tensor_sensor_names(self) -> tuple[str, ...]:
        """Sensors needed by the Hand2 world-model transition contract."""
        return self._sensor_names

    def bind_read_phase(self) -> None:
        self._read_plan = getattr(self._env.scene, "read_plan", None)
        if self._read_plan is None:
            raise RuntimeError("Hand2 reorientation requires the Manager packed read plan")
        self._read_plan.refresh()
        self._pose_stale = True

    def palm_pose(self):
        if self._pose_stale:
            view = self._read_plan.body_tensor_view("robot", (self.palm_name,))
            self._palm_pos.copy_(view.pos_w[:, 0])
            self._palm_quat.copy_(view.quat_w[:, 0])
            cube = self._read_plan.body_tensor_view("object", (self.cube_name,))
            self._cube_pos.copy_(cube.pos_w[:, 0])
            self._cube_quat.copy_(cube.quat_w[:, 0])
            self._pose_stale = False
        return self._palm_pos, self._palm_quat

    def cube_tag(self):
        tag_pos, tag_quat = self.tag_pose()
        inv = conjugate(tag_quat)
        return rotate(inv, self._cube_pos - tag_pos), multiply(inv, self._cube_quat)

    def tag_pose(self):
        self.palm_pose()
        view = self._read_plan.body_tensor_view("robot", (self.tag_name,))
        self._tag_pos.copy_(view.pos_w[:, 0])
        self._tag_quat.copy_(view.quat_w[:, 0])
        # Keep the analytic adapter's tag transform in the same palm-relative
        # frame as the source MDP.  The transform is read from the composed
        # scene because the calibration tag is not generally aligned with the
        # palm body.
        mdp = getattr(self, "mdp", None)
        if mdp is not None:
            palm_rot = quat_to_mat(self._palm_quat)
            inverse = palm_rot.transpose(-1, -2)
            tag_rot = quat_to_mat(self._tag_quat)
            mdp.tag_pos = matvec(inverse, self._tag_pos - self._palm_pos).mean(0)
            mdp.tag_rot = matmul(inverse, tag_rot).mean(0)
        return self._tag_pos, self._tag_quat

    def cube_position(self) -> torch.Tensor:
        self.palm_pose()
        return self._cube_pos

    def cube_velocity(self) -> torch.Tensor:
        view = self._read_plan.body_tensor_view("object", (self.cube_name,))
        return view.lin_vel_w[:, 0]

    def sensor_values(self, names: tuple[str, ...]) -> torch.Tensor:
        """Read named task sensors from the packed phase snapshot."""
        if self._read_plan is None or not names:
            raise RuntimeError("Hand2 sensors require a refreshed scene tensor read phase")
        views = self._read_plan.sensor_tensor_views("robot", names).values
        return torch.cat([views[name] for name in names], dim=-1)

    def contact_tensor(self) -> torch.Tensor:
        """Return the source 32-slot contact layout.

        The composed MJCF exposes five tip, one palm/proximal, and three body
        sensors per distal finger.  The source contract reduces these to five
        tip bits, one palm/proximal bit, and five distal bits; the final 25
        slots are the finger self-collision channels.
        """
        names = (
            *(f"tip_contact_{i}_{part}" for i in range(1, 6) for part in ("link4", "tip")),
            "palm_contact_palm",
            *(f"palm_contact_finger{finger}_link{link}" for finger in range(1, 6) for link in (1, 2)),
            *(f"distal_contact_{i}_{part}" for i in range(1, 6) for part in ("link3", "link4", "tip")),
        )
        try:
            if not self._sensor_names:
                return self._palm_pos.new_zeros((self.num_envs, 32))
            values = self.sensor_values(names)
        except (RuntimeError, ValueError):
            # Older UniLab read plans expose body/joint state but omit the
            # optional contact sensor plane.  Keep the transition contract
            # shape-stable; backends with the sensor capability use the real
            # values above.
            values = self._palm_pos.new_zeros((self.num_envs, len(names)))
        # Contact targets are binary found bits.  Thresholding at the task
        # boundary keeps the replay contract valid even if a backend exposes
        # counts rather than booleans.
        found = values > 0
        if found.shape[-1] != 36:
            raise RuntimeError(
                "Hand2 contact found sensors must expose 36 scalar channels, "
                f"got {found.shape[-1]}"
            )
        # Source tip-object matching covers both distal link4 and tip-sensor
        # collision geoms; reduce the two body channels per finger to five
        # fingertip bits.
        tip = found[:, :10].reshape(self.num_envs, 5, 2).any(-1).float()
        # Source ``palm_object_found`` matches palm plus all finger link1/2
        # bodies.  Collapse those eleven body sensors into one proximal bit.
        palm = found[:, 10:21].any(-1, keepdim=True).float()
        # The source contact contract has one aggregate distal bit (the
        # five link3/link4/tip-sensor groups are reduced together).
        distal = found[:, 21:].any(-1, keepdim=True).float()
        try:
            collision = self.sensor_values(tuple(f"finger_collision_{i}" for i in range(25))) > 0
        except (RuntimeError, ValueError):
            collision = torch.zeros(
                (self.num_envs, 25), dtype=torch.bool, device=self._device
            )
        compact = torch.cat((tip, palm, distal), dim=-1)
        return torch.cat((compact, collision.float()), dim=-1)

    def tip_velocity_tensor(self) -> torch.Tensor:
        """Return the five MJCF tip-site linear velocities in world frame.

        Sensor/body velocities can retain the last substep's pre-integration
        values. Use the same integrated joint state and site-offset FK as the
        analytic reward, then rotate its palm-frame result into world frame.
        """
        if self._read_plan is None:
            raise RuntimeError("Hand2 tip velocities require a refreshed scene tensor read phase")
        joints = self._read_plan.joint_tensor_view("robot")
        kinematics = self.mdp.kinematics
        _, velocities = kinematics.tips(kinematics.forward(joints.joint_pos, joints.joint_vel))
        palm = self._read_plan.body_tensor_view("robot", (self.palm_name,))
        return rotate(palm.quat_w.expand(-1, velocities.shape[-2], -1), velocities)

    def torque_tensor(self) -> torch.Tensor:
        if not self._sensor_names:
            return self._palm_pos.new_zeros((self.num_envs, 20))
        names = tuple(f"actuator_force_{i}" for i in range(20))
        try:
            return self.sensor_values(names)
        except ValueError:
            return self._palm_pos.new_zeros((self.num_envs, len(names)))

    def auxiliary_tensor(self) -> torch.Tensor:
        """Pack palm pose and exogenous perturbation metadata (21 values)."""
        self.palm_pose()
        rotation = quat_to_mat(self._palm_quat).flatten(-2)
        origins = self._env.scene.env_origins(self._device)
        return torch.cat(
            (self._palm_pos - origins, rotation, self.force_direction, self.perturbation), dim=-1
        )

    def state_tensor(self) -> torch.Tensor:
        """Return the 58D Hand 2 SuperMimic physical state.

        The reference learner works in a palm-relative frame.  Keeping this
        conversion on the command term means both MuJoCo and mjwarp consume
        the same packed read plan and the environment does not need backend
        specific ``qpos``/``qvel`` access.
        """
        self.palm_pose()
        if self._read_plan is None:
            raise RuntimeError("Hand2 state requires a refreshed scene tensor read phase")
        joints = self._read_plan.joint_tensor_view("robot")
        cube = self._read_plan.body_tensor_view("object", (self.cube_name,))
        palm_rotation = quat_to_mat(self._palm_quat)
        inverse = palm_rotation.transpose(-1, -2)
        cube_rotation = quat_to_mat(cube.quat_w[:, 0])
        return pack_state(
            joints.joint_pos,
            joints.joint_vel,
            matvec(inverse, cube.pos_w[:, 0] - self._palm_pos),
            matmul(inverse, cube_rotation),
            matvec(inverse, cube.lin_vel_w[:, 0]),
            matvec(inverse, cube.ang_vel_w[:, 0]),
        )

    def context_tensor(self) -> torch.Tensor:
        """Return the 46D simulator context used by the world model.

        The first migration exposes the six sampled DR summaries.  Exact
        per-joint gain ratios are appended when a randomization term provides
        them; otherwise unit ratios preserve the source layout and keep the
        context shape stable on MuJoCo host runs.
        """
        kp = kd = torch.ones((self.num_envs, 20), device=self._device)
        # FixedRandomization stores exact gain ratios on the event term.  The
        # event is optional in the base Hand2 config, so avoid relying on a
        # private manager accessor here.
        event_manager = getattr(self._env, "event_manager", None)
        # UniLab stores event terms by mode in ``term_cfgs``; older local
        # manager builds exposed a private ``_terms`` mapping.  Support both
        # so the context remains calibrated on mjwarp and shape-stable on
        # MuJoCo host runs.
        term_cfgs = getattr(event_manager, "term_cfgs", {}) if event_manager is not None else {}
        if isinstance(term_cfgs, dict) and term_cfgs:
            terms = tuple(getattr(cfg, "func", None) for cfg in term_cfgs.values())
        else:
            legacy = getattr(event_manager, "_terms", {}) if event_manager is not None else {}
            terms = tuple(legacy.values()) if isinstance(legacy, dict) else ()
        for term in terms:
            ratios = getattr(term, "ratios", None)
            if isinstance(ratios, dict):
                kp = ratios.get("kp", kp)
                kd = ratios.get("kd", kd)
                break
        return torch.cat((self.dr_ratios, kp, kd), dim=-1)

    def memory_tensor(self) -> dict[str, torch.Tensor]:
        """Capture controller/goal state needed by imagined Hand 2 rollouts."""
        # Refresh the calibrated palm->tag transform before converting the
        # world-frame goal into the MDP's palm-relative goal matrix.
        self.tag_pose()
        action_manager = self._env.action_manager
        action = getattr(action_manager, "action", None)
        previous = getattr(action_manager, "prev_action", None)
        if action is None:
            action = torch.zeros_like(self.goal[:, :20])
        if previous is None:
            previous = action
        action_term = action_manager.get_term("joint_pos")
        target = action_term.target
        previous_target = action_term.previous_target
        encoder_bias = getattr(
            self.robot.data,
            "encoder_bias_tensor",
            torch.zeros_like(target),
        )
        encoder_bias = encoder_bias.to(device=target.device, dtype=target.dtype)
        palm_rotation = quat_to_mat(self._palm_quat)
        goal_rotation = quat_to_mat(self.goal)
        def scalar(value: torch.Tensor) -> torch.Tensor:
            return value.to(dtype=torch.float32).reshape(-1, 1)
        return {
            "c1": target.detach().clone(),
            "c2": previous_target.detach().clone(),
            # ``c1`` is the controller target before the simulated encoder
            # offset.  The WM and analytic torque use the actually executed
            # target, matching Dex-SuperMimic's ``u = c - encoder_bias``.
            "u": (target - encoder_bias).detach().clone(),
            "bias": encoder_bias.detach().clone(),
            "a1": action.detach().clone(),
            "a2": previous.detach().clone(),
            "goal": matmul(palm_rotation.transpose(-1, -2), goal_rotation).flatten(-2),
            "hold": scalar(self.hold),
            "goal_timer": scalar(self.goal_timer),
            # ``window`` is the source command's in-success-window flag.
            # The live UniLab counter uses the same field to count elapsed
            # window steps, so keep that timer only in ``window_timer``.
            # Feeding values 1..49 into the differentiable command update
            # makes ``1 - window`` negative and can explode soft hold/window
            # counters during imagination.
            "window": scalar((self.window > 0).to(dtype=torch.float32)),
            "window_timer": scalar(self.window),
            # Source keeps a separate elapsed success-window timer and
            # within-window hold counter.  ``window`` is the live elapsed
            # counter; ``reward_timer`` is the latter.
            "reward_window": scalar(self.reward_window_timer),
            "reward_hold": scalar(self.reward_hold_counter),
            "reward_snapshot": scalar(self.reward_hold_counter_snapshot),
            "within": scalar(self.within),
            "cage": scalar(self.cage_counter),
            "age": scalar(self._env.episode_length_buf),
            "count": scalar(self.goal_count),
        }

    def error(self) -> torch.Tensor:
        # The command goal is stored in world coordinates (sampled from the
        # tag frame at reset), so compare it with the cube's world quaternion.
        # ``cube_tag()`` is only for policy observations and would mix frames
        # here when the calibrated tag is rotated relative to the palm.
        self.palm_pose()
        return angle_error(self._cube_quat, self.goal)

    def _update_metrics(self) -> None:
        self.metrics["goal_reach_count"].copy_(self.goal_count)
        self.metrics["ori_error"].copy_(self.error())
        self.metrics["hold_counter"].copy_(torch.where(self.window > 0, 0, self.hold).float())
        self.metrics["goal_timer"].copy_(self.goal_timer.float())
        self.metrics["in_success_window"].copy_((self.window > 0).float())
        self.metrics["window_timer"].copy_(self.window.float())

    def refresh_reset(self, reset_mask: torch.Tensor) -> None:
        """Invalidate cached poses and clear task counters for reset rows."""
        self._pose_stale = True
        pending = reset_mask & self._pending_goal_mask
        if bool(pending.any()):
            # The manager has refreshed the scene read plan at this point, so
            # tag_pose() observes the newly reset wrist pitch/mocap pose.
            _tag_pos, tag_quat = self.tag_pose()
            self.goal[pending] = multiply(
                tag_quat[pending], self._pending_goal_local[pending]
            )
            self._pending_goal_mask.masked_fill_(pending, False)
        self.hold.masked_fill_(reset_mask, 0)
        self.window.masked_fill_(reset_mask, 0)
        self.goal_count.masked_fill_(reset_mask, 0)
        self.goal_timer.masked_fill_(reset_mask, 0)
        self.reward_window_timer.masked_fill_(reset_mask, 0)
        self.reward_hold_counter.masked_fill_(reset_mask, 0)
        self.reward_hold_counter_snapshot.masked_fill_(reset_mask, 0)
        self.success.masked_fill_(reset_mask, False)
        self.within.masked_fill_(reset_mask, False)
        self.outside.masked_fill_(reset_mask, False)
        self.cage_counter.masked_fill_(reset_mask, 0.0)

    def post_compute_reset(self) -> None:
        self._pose_stale = True

    def post_compute(self) -> None:
        # UniLab refreshes the read plane after committed command-side writes
        # and invokes this hook before observations. Cache only within a read
        # phase; a reset/command commit may have changed the underlying pose.
        self._pose_stale = True

    def _resample_command(self, reset_mask: torch.Tensor, *, reset_snapshot: bool = True) -> None:
        # The reference samples a goal uniformly in the wrist-tag frame and
        # stores the resulting world-frame quaternion.  Sampling directly in
        # world coordinates changes the task whenever the wrist is tilted,
        # because the policy/MDP later converts this world goal back through
        # the palm/tag frame.
        samples = random_quaternions(self._rng, self.num_envs, device=self._device)
        if reset_snapshot:
            # Reset command sampling precedes UniLab's post-reset read-plan
            # refresh.  Compose with the fresh tag pose in refresh_reset().
            self._pending_goal_local.copy_(samples)
            self._pending_goal_mask.copy_(reset_mask)
        else:
            try:
                _tag_pos, tag_quat = self.tag_pose()
                samples = multiply(tag_quat, samples)
            except (RuntimeError, AttributeError):
                # Internal goal switches normally run with a live read plan;
                # keep the identity-tag fallback for cold construction only.
                pass
        self.goal.copy_(torch.where(reset_mask[:, None], samples, self.goal))
        self.hold.masked_fill_(reset_mask, 0)
        self.window.masked_fill_(reset_mask, 0)
        self.goal_timer.masked_fill_(reset_mask, 0)
        if reset_snapshot:
            # ``CommandTerm.reset`` invokes this method directly at an
            # episode boundary.  ``refresh_reset`` is intentionally not part
            # of UniLab's manager hook, so clear all per-episode counters here
            # (an internal goal switch must preserve them).
            self.goal_count.masked_fill_(reset_mask, 0)
            self.outside.masked_fill_(reset_mask, False)
            self.cage_counter.masked_fill_(reset_mask, 0.0)
            self.reward_window_timer.masked_fill_(reset_mask, 0)
        self.reward_hold_counter.masked_fill_(reset_mask, 0)
        if reset_snapshot:
            self.reward_hold_counter_snapshot.masked_fill_(reset_mask, 0)
        self.success.masked_fill_(reset_mask, False)

    def _update_command(self) -> None:
        self._pose_stale = True
        within = self.error() < self.cfg.success_threshold
        self.within.copy_(within)
        approaching = self.window == 0
        self.goal_timer += 1
        # The approach hold counter only advances before the success window.
        # Once a goal is reached it must stay at zero while the window is
        # active; the source command keeps that counter separate from the
        # reward hold counter.  Updating it during the window makes the live
        # memory disagree with ``mdp.command_step`` and slowly feeds an
        # unbounded value back into imagined rollouts.
        self.hold.copy_(torch.where(
            (self.window == 0) & within,
            self.hold + 1,
            torch.where(self.window == 0, torch.zeros_like(self.hold), self.hold),
        ))
        succeeded = approaching & (self.hold >= self.cfg.success_hold_steps)
        self.hold.masked_fill_(succeeded, 0)
        in_window = ~approaching | succeeded
        reward_window_timer = torch.where(
            in_window, self.window + 1, torch.zeros_like(self.window)
        )
        reward_hold_counter = torch.where(
            in_window & within,
            self.reward_hold_counter + 1,
            self.reward_hold_counter,
        )
        reward_hold_counter = torch.where(
            succeeded, torch.zeros_like(reward_hold_counter), reward_hold_counter
        )
        self.reward_window_timer.copy_(reward_window_timer)
        self.reward_hold_counter.copy_(reward_hold_counter)
        # The reward manager reads this snapshot before a goal switch clears
        # the live counters, exactly as the source task does.
        self.reward_hold_counter_snapshot.copy_(reward_hold_counter)
        self.success.copy_(succeeded)
        self.goal_count.add_(succeeded.to(self.goal_count.dtype))
        self.window.copy_(torch.where(succeeded, torch.ones_like(self.window), self.window))
        active = (self.window > 0) & ~succeeded
        self.window.add_(active.to(self.window.dtype))
        change = self.window >= self.cfg.goal_switch_delay
        # Preserve the pre-switch reward counters for this transition, then
        # clear live state for the newly sampled goal.
        self.reward_window_timer.copy_(torch.where(change, self.window, self.reward_window_timer))
        self.reward_hold_counter_snapshot.copy_(self.reward_hold_counter)
        self._resample_command(change, reset_snapshot=False)
        # Publish metrics after the switch/reset has been applied so the
        # manager reports the same live counters that observations and the
        # next transition consume.  The reward itself uses the snapshot
        # captured above and is therefore unaffected by this ordering.
        self._update_metrics()

def task(env) -> Hand2ReorientCommand:
    command = env.command_manager.get_term("reorient")
    if not isinstance(command, Hand2ReorientCommand):
        raise TypeError("Hand2 task expects a Hand2ReorientCommand named 'reorient'")
    return command


__all__ = ["Hand2ReorientCommand", "Hand2ReorientCommandCfg", "task"]

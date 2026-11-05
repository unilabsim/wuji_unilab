"""UniLab physics boundary for the source SuperMimic pen learner.

This adapter does not use mjlab or mutate the completed Hand2 cube task.
The native Pen task supplies physics/reset/reference playback; this training
boundary reads the source controller/camera memory and measured native rewards.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from .common import require_finite
from .kinematics import HandKinematics
from .rotations import mm, mv, quat_to_mat
from .state import pack


def __getattr__(name):
    if name == "SimReader":
        from .hand1_reader import SimReader

        return SimReader
    raise AttributeError(name)


class PenSimReader:
    """Read actual pen state/contact/force and own source controller memory."""

    world_model_object_group = "pen"

    def __init__(self, env):
        from unisupermimic.tasks.pen_spin.scene import materialize_scene

        from .pen import PEN_MEMORY_DIMS, PenMDP

        self.env = env
        self.native_env = getattr(env, "native_env", env)
        self.device = torch.device(env.device)
        self.entity = env.scene["robot"]
        self.obj = env.scene["object"]
        self.motion = env.command_manager.get_term("motion")
        self.model = mujoco.MjModel.from_xml_path(str(materialize_scene()))
        model = self.model
        self.joint_names = tuple(self.entity.joint_names)
        joint_ids = [model.joint(name).id for name in self.joint_names]
        self._joint_q = model.jnt_qposadr[joint_ids].tolist()
        self._joint_v = model.jnt_dofadr[joint_ids].tolist()
        free_joint = model.joint("object/cube_freejoint").id
        self._object_q = int(model.jnt_qposadr[free_joint])
        self._object_v = int(model.jnt_dofadr[free_joint])
        self.actuator_ids = np.array([
            model.actuator(f"{name}_actuator").id for name in self.joint_names
        ])
        body_ids = [
            i for i in range(model.nbody) if model.body(i).name.startswith("robot/")
        ]
        tip_ids = [model.site(f"robot/right_finger{i}_tip").id for i in range(1, 6)]
        self.kin = HandKinematics(model, body_ids, joint_ids, tip_ids, self.device)
        data = self.entity.data
        default = data.default_joint_pos(self.device)
        self.default_joint_pos = (default[0] if default.ndim >= 2 else default).clone()
        # PenMDP and the source joint tracking reward use joint_pos_limits
        # (hard MJCF ranges), even though the articulation also publishes a
        # 0.9 soft-limit table. Do not use the cube controller's soft table.
        limits = torch.tensor(
            np.array(model.jnt_range[joint_ids]), device=self.device, dtype=torch.float32
        )
        self.lower, self.upper = limits[:, 0].clone(), limits[:, 1].clone()

        def tensor(value):
            return torch.tensor(np.array(value), device=self.device, dtype=torch.float32)

        ids = self.actuator_ids
        self.actuator_defaults = dict(
            kp=tensor(model.actuator_gainprm[ids, 0]),
            kd=tensor(model.actuator_biasprm[ids, 2]),
            force_limit=tensor(model.actuator_forcerange[ids, 1]),
            gear=tensor(model.actuator_gear[ids, 0]),
        )
        self.action = env.action_manager.get_term("joint_pos")
        action_cfg = self.action.cfg
        self.action_cfg = SimpleNamespace(
            alpha=float(action_cfg.ema_alpha), action_scale=float(action_cfg.action_scale),
            residual_clip=tuple(action_cfg.residual_clip), mask_prob=float(action_cfg.mask_prob),
            mask_num_dofs=int(action_cfg.mask_num_dofs), mask_duration_min=int(action_cfg.mask_duration_min),
            mask_duration_max=int(action_cfg.mask_duration_max),
            mask_curriculum_term=action_cfg.mask_curriculum_term,
        )
        n = env.num_envs
        self._memory = {
            name: torch.zeros((n, dim), device=self.device)
            for name, dim in PEN_MEMORY_DIMS.items()
        }
        self._memory["camera_bias_rot"][:] = torch.eye(3, device=self.device).flatten()
        self._wrench_body_ids = body_ids + [model.body("object/cube").id]
        self.robot = SimpleNamespace(
            joint_names=self.joint_names,
            data=SimpleNamespace(encoder_bias=getattr(data, "encoder_bias_tensor", self._memory["encoder_bias"])),
        )
        camera = env.observation_manager.get_term_cfg("policy", "object_pose").func
        self.camera = camera if hasattr(camera, "memory_state") else None
        self.xi = self.context()
        self.refresh_frame()
        self.mdp = PenMDP(self)
        self.reward_snapshot = None
        env.training_reader = self
        self.reset_rows(torch.arange(n, device=self.device))

    def context(self):
        from unisupermimic.tasks.pen_spin.randomization import default_model_fields
        from unisupermimic.tasks.pen_spin.reset_bridge import get_pen_physics_model

        from .observations import DR_PARAM_NAMES

        model = getattr(self.native_env, "_pen_physics_model", None)
        fields = getattr(self.native_env, "_pen_model_fields", None)
        defaults = getattr(self.native_env, "_pen_default_model_fields", None)
        if model is None:
            model = get_pen_physics_model(self.native_env)
        if fields is None:
            # Explicit fixed-physics/play configurations omit the startup term.
            defaults = fields = default_model_fields(model, self.env.num_envs, self.device)
        contact = [g for g in range(model.ngeom) if re.fullmatch(
            r"robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)",
            model.geom(g).name,
        )]
        obj = [model.body("object/cube").id]
        geoms = [model.geom(name).id for name in ("object/cube", "object/A_bottom", "object/A_top")]
        dofs = [model.jnt_dofadr[model.joint(name).id] for name in self.joint_names]
        acts = self.actuator_ids.tolist()
        kp = fields["actuator_gainprm"][:, acts, 0] / defaults["actuator_gainprm"][:, acts, 0].clamp_min(1e-8)
        kd = fields["actuator_biasprm"][:, acts, 2] / defaults["actuator_biasprm"][:, acts, 2].clamp(max=-1e-8)
        dr = torch.cat((
            (fields["geom_friction"][:, contact, 0] / defaults["geom_friction"][:, contact, 0].clamp_min(1e-8)).mean(-1, keepdim=True),
            (fields["body_mass"][:, obj] / defaults["body_mass"][:, obj].clamp_min(1e-8)).mean(-1, keepdim=True),
            kp.mean(-1, keepdim=True), kd.mean(-1, keepdim=True),
            (fields["dof_damping"][:, dofs] / defaults["dof_damping"][:, dofs].clamp_min(1e-8)).mean(-1, keepdim=True),
            (fields["geom_size"][:, geoms] / defaults["geom_size"][:, geoms].clamp_min(1e-8)).flatten(1).mean(-1, keepdim=True),
        ), -1)
        self.context_names = [(name, 1) for name in DR_PARAM_NAMES] + [
            ("actuator_kp_ratios", 20), ("actuator_kd_ratios", 20)
        ]
        return torch.cat((dr, kp, kd), -1).clone()

    def refresh_frame(self):
        palm = self.env.scene.read_plan.body_tensor_view(
            "robot", ("robot/right_palm_link",)
        )
        self.palm_pos = palm.pos_w[:, 0].clone()
        self.palm_rot = quat_to_mat(palm.quat_w[:, 0])

    def state(self):
        # Source SuperMimic reads the final generalized qpos/qvel after the
        # last integration. Frame sensors in the Manager packed read plan can
        # retain the preceding substep's body poses on the MuJoCo backend.
        values = self.native_env.backend.get_state_views(("qpos", "qvel"), device=self.device)
        q, v = values["qpos"], values["qvel"]
        obj_q = q[:, self._object_q:self._object_q + 7]
        obj_v = v[:, self._object_v:self._object_v + 6]
        rotation = quat_to_mat(obj_q[:, 3:])
        inverse = self.palm_rot.transpose(-1, -2)
        value = pack(
            q[:, self._joint_q], v[:, self._joint_v],
            mv(inverse, obj_q[:, :3] - self.palm_pos),
            mm(inverse, rotation),
            mv(inverse, obj_v[:, :3]), mv(inverse, mv(rotation, obj_v[:, 3:])),
        ).clone()
        require_finite("pen physics state", value)
        return value

    def _sensors(self, names):
        # Contact and actuator-force labels belong to the last physics
        # evaluation, matching source mjData sensor/force timing. They are
        # intentionally not reconstructed from the final integrated qpos.
        views = self.env.scene.read_plan.sensor_tensor_views("robot", names)
        return torch.cat([views.values[name].reshape(self.env.num_envs, -1) for name in names], -1)

    def contact(self):
        model = self.model
        geom_channels = {}
        for index in range(25):
            name = f"finger_collision_{index}"
            geom = model.geom(int(model.sensor(name).objid[0])).name
            geom_channels[geom] = name
        middle = tuple(geom_channels[f"robot/right_finger{f}_link{j}_col"]
                       for f in range(1, 6) for j in (2, 3))
        distal = tuple(geom_channels[f"robot/right_finger{f}_link4_col"] for f in range(1, 6))
        tips = tuple(geom_channels[f"robot/right_finger{f}_tip_sensor_col"] for f in range(1, 6))
        return torch.cat((self._sensors(middle) > 0,
                          (self._sensors(distal) > 0) | (self._sensors(tips) > 0)), -1).float()

    def torque(self):
        return self._sensors(tuple(f"actuator_force_{i}" for i in range(20))).clone()

    def external_wrench(self):
        wrench = self.native_env._pen_external_wrench
        return wrench[:, self._wrench_body_ids].flatten(1).clone()

    def torque_limits(self):
        fields = getattr(self.native_env, "_pen_model_fields", None)
        if fields is None:
            return self.actuator_defaults["force_limit"].expand(self.env.num_envs, -1).clone()
        return fields["actuator_forcerange"][:, self.actuator_ids.tolist(), 1].clone()

    def memory(self):
        motion = self.motion
        out = {name: value.clone() for name, value in self._memory.items()}

        def scalar(value):
            return value.float().reshape(-1, 1).clone()

        out.update(
            c1=self.action.processed_action.clone(),
            c2=self.action.processed_action.clone(),
            a1=self.env.action_manager.action.clone(),
            a2=self.env.action_manager.prev_action.clone(),
            goal=torch.eye(3, device=self.device).flatten().expand(self.env.num_envs, -1),
            age=scalar(self.env.episode_length_buf), time=scalar(motion.time_steps),
            start=scalar(motion._trajectory_starts[motion.trajectory_ids]),
            end=scalar(motion._trajectory_ends[motion.trajectory_ids]),
            limits=self.torque_limits(),
            cage=self.native_env.reward_manager.get_term_cfg("overcurrent").func.charge.clone(),
            encoder_bias=self.robot.data.encoder_bias.clone(),
            frozen_target=self.action.target.clone(),
            mask=self.action._mask_dofs.float().clone(),
            mask_remaining=scalar(self.action._mask_remaining),
            mask_active=scalar(self.action._mask_active),
        )
        if self.camera is not None:
            out.update(self.camera.memory_state())
        return out

    def before_step(self):
        self.refresh_frame()
        self._pre_memory = self.memory()

    def executed_target(self):
        return self.action.target.clone()

    def policy_observation(self, ids=None):
        # The real environment owns source observation/camera scheduling.
        # Reading the published policy must not sample a second camera tick;
        # PenMDP.policy_observation remains the source imagination-only path.
        observations = self.native_env.obs_buf
        policy = observations.get("policy", observations.get("obs", observations.get("actor")))
        if policy is None:
            raise RuntimeError("Pen environment did not publish its policy observation")
        return policy.clone() if ids is None else policy[ids].clone()

    def reset_rows(self, ids):
        for value in self._memory.values():
            value[ids] = 0.0
        self._memory["camera_bias_rot"][ids] = torch.eye(3, device=self.device).flatten()
        # Camera reset/backfill is performed exactly once by ObservationManager.

    def attach_capture(self):
        manager = self.native_env.reward_manager
        self._original_reward = manager.compute

        def compute(dt):
            self.refresh_frame()
            physics, contact, torque = self.state(), self.contact(), self.torque()
            nan = self.env.termination_manager.get_term("nan_guard").clone()
            drop = self.env.termination_manager.terminated.clone()
            reward = self._original_reward(dt)
            require_finite("pen real reward", reward)
            self.reward_snapshot = dict(
                physics=physics, contact=contact, torque=torque,
                nan=nan, drop=drop, reward=reward.clone(),
                terms={name: manager._step_reward[:, j].clone() * (dt if manager._scale_by_dt else 1.)
                       for j, name in enumerate(manager._term_names)},
            )
            return reward

        manager.compute = compute

    def detach_capture(self):
        if hasattr(self, "_original_reward"):
            self.native_env.reward_manager.compute = self._original_reward

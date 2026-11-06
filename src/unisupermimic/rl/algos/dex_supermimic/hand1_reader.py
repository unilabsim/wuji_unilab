"""UniLab reader for the unchanged source Hand 1 SuperMimic equations."""

from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from .kinematics import HandKinematics
from .mdp import DexMDP
from .observations import CURRENT_TERMS, DR_PARAM_NAMES, FRAME_DIM, TaskObservations
from .rotations import mm, mv, quat_to_mat
from .state import pack


class SimReader:
    def __init__(self, env):
        self.env = env
        self.native_env = getattr(env, "native_env", env)
        self.device = torch.device(env.device)
        self.command = env.command_manager.get_term("reorient")
        self.entity = env.scene["robot"]
        self.obj = env.scene["object"]
        self.model = mujoco.MjModel.from_xml_path(str(self.command.scene_path))
        model = self.model
        joint_ids = [model.joint(name).id for name in self.entity.joint_names]
        self._joint_q = model.jnt_qposadr[joint_ids].tolist()
        self._joint_v = model.jnt_dofadr[joint_ids].tolist()
        free_joint = model.joint("object/cube_freejoint").id
        self._object_q = int(model.jnt_qposadr[free_joint])
        self._object_v = int(model.jnt_dofadr[free_joint])
        body_ids = [i for i in range(model.nbody) if model.body(i).name.startswith("robot/")]
        tip_sites = [model.site(f"robot/right_finger{i}_tip").id for i in range(1, 6)]
        self.kin = HandKinematics(model, body_ids, joint_ids, tip_sites, self.device)

        def tensor(value):
            return torch.tensor(np.array(value), device=self.device, dtype=torch.float32)

        data = self.entity.data
        default = data.default_joint_pos(self.device)
        default = default[0] if default.ndim >= 2 else default
        # UniLab's field currently exposes hard MJCF joint ranges. The source
        # robot articulation has a 0.9 soft-limit factor; use the same table
        # as the live controller for normalization and imagined controls.
        limits = env.action_manager.get_term("joint_pos").limits
        tag_body = model.body("robot/right_wrist_tag_frame")
        actuator_ids = [model.actuator(f"{name}_actuator").id for name in self.entity.joint_names]
        self.actuator_defaults = dict(
            kp=tensor(model.actuator_gainprm[actuator_ids, 0]),
            kd=tensor(model.actuator_biasprm[actuator_ids, 2]),
            force_limit=tensor(np.abs(model.actuator_forcerange[actuator_ids]).max(-1)),
            gear=tensor(model.actuator_gear[actuator_ids, 0]),
        )
        # The pinned source EntityData has no actuator_effort_limit for its
        # XML actuator. A joint force clamp is separate metadata; substituting
        # it here would change the source torque reward by 1 / limit**2.
        effort = getattr(data, "actuator_effort_limit", None)
        if effort is not None:
            effort = torch.as_tensor(effort, device=self.device, dtype=torch.float32)
            effort = effort[0] if effort.ndim >= 2 else effort
        self.mdp = DexMDP(
            self.kin, default.clone(), limits[:, 0].clone(), limits[:, 1].clone(),
            tensor(tag_body.pos), quat_to_mat(tensor(tag_body.quat)),
            round(1 / env.step_dt), effort, cage_up_axis=0,
        )
        self.mdp.kp_default = self.actuator_defaults["kp"]
        self.mdp.kd_default = self.actuator_defaults["kd"]
        self.mdp.force_limit = self.actuator_defaults["force_limit"]
        self.mdp.gear = self.actuator_defaults["gear"]
        default_velocity = data.default_joint_vel(self.device)
        if default_velocity.ndim >= 2:
            default_velocity = default_velocity[0]
        # The source learner expects Tensor-valued data attributes, whereas
        # UniLab keeps device-specific reads explicit on EntityData.
        self.robot = SimpleNamespace(
            joint_names=tuple(self.entity.joint_names),
            data=SimpleNamespace(encoder_bias=data.encoder_bias_tensor,
                                 default_joint_vel=default_velocity[None]),
        )
        self.refresh_frame()
        self.xi = self.context()
        self.reward_snapshot = None
        self._original_reward = None
        env.training_reader = self

    def refresh_frame(self):
        self.command._pose_stale = True
        position, quaternion = self.command.palm_pose()
        self.palm_pos = position.clone()
        self.palm_rot = quat_to_mat(quaternion)

    def gravity(self):
        return mv(self.palm_rot.transpose(-1, -2), self.palm_pos.new_tensor([0, 0, -9.81]))

    def state(self):
        # Source SimReader reads the generalized state, not last-substep
        # frame sensors. MuJoCo frame sensors may lag its final integration.
        values = self.native_env.backend.get_state_views(("qpos", "qvel"), device=self.device)
        q, v = values["qpos"], values["qvel"]
        obj_q = q[:, self._object_q:self._object_q + 7]
        obj_v = v[:, self._object_v:self._object_v + 6]
        rotation = quat_to_mat(obj_q[:, 3:])
        inverse = self.palm_rot.transpose(-1, -2)
        return pack(
            q[:, self._joint_q].clone(), v[:, self._joint_v].clone(),
            mv(inverse, obj_q[:, :3] - self.palm_pos), mm(inverse, rotation),
            mv(inverse, obj_v[:, :3]), mv(inverse, mv(rotation, obj_v[:, 3:])),
        )

    def memory(self):
        # Only source memory fields enter the source replay schema. UniLab's
        # command additionally exposes executed target and bias for its own
        # environment contract.
        from .state import MEMORY_DIMS

        values = self.command.memory_tensor()
        return {name: values[name].clone() for name in MEMORY_DIMS}

    def contact(self):
        return self.command.contact_tensor().clone()

    def context(self):
        self.context_names = [(name, 1) for name in DR_PARAM_NAMES] + [
            ("actuator_kp_ratios", 20), ("actuator_kd_ratios", 20)
        ]
        self.xi = self.command.context_tensor().clone()
        return self.xi

    def attach_capture(self):
        """Capture actual reward-phase labels before events/command updates."""
        if self._original_reward is not None:
            raise RuntimeError("Hand 1 reward capture is already attached")
        native = self.native_env
        self._original_reward = native._compute_rewards

        def reward():
            self.refresh_frame()
            physics = self.state()
            contact = self.contact()
            torque = self.command.torque_tensor().clone()
            drop = native.termination_manager.get_term("cage_drop").clone()
            nan = native.termination_manager.get_term("nan_guard").clone()
            result = self._original_reward()
            manager = native.reward_manager
            scale = native.step_dt if native.cfg.scale_rewards_by_dt else 1.0
            self.reward_snapshot = dict(
                physics=physics, contact=contact, torque=torque, drop=drop, nan=nan,
                reward=result.clone(),
                terms={name: manager.step_rewards[:, i].clone() * scale
                       for i, name in enumerate(manager.step_reward_names)},
            )
            return result

        native._compute_rewards = reward

    def detach_capture(self):
        if self._original_reward is not None:
            self.native_env._compute_rewards = self._original_reward
            self._original_reward = None


class Hand1Observations(TaskObservations):
    """Source clean critic/dynamics features, using UniLab-owned metadata."""

    def __init__(self, reader, legacy_context=False):
        if legacy_context:
            raise ValueError("New Hand 1 training uses the source six-ratio context")
        self.reader = reader
        self.mdp = reader.mdp
        self.legacy_context = False
        self.current_terms = CURRENT_TERMS
        self.dr_dim = len(DR_PARAM_NAMES)
        self.frame_dim = FRAME_DIM
        self.tip_ids = [reader.kin.names.index(f"robot/right_finger{i}_link4") for i in range(1, 6)]
        self.tip_scale = 5.0
        self.tip_offset = torch.tensor((0.0, 0.0, 0.5), device=reader.device)
        self.default_velocity = reader.robot.data.default_joint_vel[0].clone()
        self.episode_steps = reader.env.max_episode_length
        self.context_dim = reader.xi.shape[-1] + self.frame_dim

    def auxiliary(self):
        command = self.reader.command
        return torch.cat((command.auxiliary_tensor(), command.dr_ratios), -1).clone()


__all__ = ["Hand1Observations", "SimReader"]

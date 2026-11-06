"""Reset events for the migrated Wuji Hand 2 pen-spin task.

The source task starts every episode from the first frame of a tracking clip.
Keeping this write in an event (rather than relying on the XML defaults) is
important for both real clip playback and synthetic smoke trajectories.  The
MJWarp backend exposes a packed mocap reset transaction; MuJoCo does not, so
the mocap write is optional on the host backend where the scene's calibrated
XML pose is already the required wrist pose.
"""

from __future__ import annotations

from typing import cast

import torch
from unilab.base.entity import Entity
from unilab.managers import ManagerTermBase

from .commands import JOINT_NAMES, PenSpinCommand, task
from .curriculums import get_curriculum_value
from .reset_bridge import bind_mocap_reset, bind_physics_state, write_mocap_reset


def _quat_multiply(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = first.unbind(-1)
    w2, x2, y2, z2 = second.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _quat_rotate(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    conjugate = torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1)
    pure = torch.cat((torch.zeros_like(vector[..., :1]), vector), dim=-1)
    return _quat_multiply(_quat_multiply(quat, pure), conjugate)[..., 1:]


class ResetPenReference(ManagerTermBase):
    """Write the selected command frame into the hand and pen state."""

    def __init__(self, cfg, env):
        del cfg
        super().__init__(env)
        self.robot = cast(Entity, env.scene["robot"])
        self.object = cast(Entity, env.scene["object"])
        # Event terms are constructed before the command manager in UniLab;
        # resolve the command lazily on the first reset invocation.
        self.motion: PenSpinCommand | None = None
        joint_ids, found = self.robot.find_joints(JOINT_NAMES, preserve_order=True)
        if tuple(found) != JOINT_NAMES:
            raise ValueError(
                "pen reference joints do not match robot entity: "
                f"expected {JOINT_NAMES}, found {tuple(found)}"
            )
        # UniLab normalizes these static local joint IDs with np.asarray;
        # keep the index metadata on the host even when state tensors use CUDA.
        self.joint_ids = tuple(int(index) for index in joint_ids)
        try:
            self.pose_binding = bind_mocap_reset(
                self.robot, "robot/right_palm_link", term_name="pen_reference_reset"
            )
        except (AttributeError, RuntimeError):
            # MuJoCo host reset has no packed mocap transaction.  Its calibrated
            # scene default is the same wrist pose used by the synthetic clip.
            self.pose_binding = None

    @staticmethod
    def _env_ids(env, env_ids) -> torch.Tensor:
        if env_ids is None or isinstance(env_ids, slice):
            return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=env.device, dtype=torch.long).reshape(-1)

    def __call__(
        self, env, env_ids: torch.Tensor | None,
        object_reset_axial_slide_range=(0.0, 0.0),
        object_reset_roll_range=(0.0, 0.0),
        curriculum_term=None,
    ) -> None:
        motion = task(env)
        if not isinstance(motion, PenSpinCommand):
            raise TypeError("Pen reset requires a PenSpinCommand named 'motion'")
        self.motion = motion
        ids = self._env_ids(env, env_ids)
        if ids.numel() == 0:
            return
        bind_physics_state(env)
        mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        mask[ids] = True
        # Event execution precedes the manager's read-plan refresh.  Resetting
        # the command frame here makes the writes below deterministic even for
        # partial resets and keeps non-reset environments on their trajectory.
        motion.prepare_reset(ids)

        joint_pos = motion.joint_pos
        joint_vel = motion.joint_vel
        # The packed reset API takes a full-width value plus a row mask.  This
        # also avoids the host backend's partial-joint transaction path, which
        # cannot represent an env-id subset and would assert before commit.
        self.robot.write_joint_state_to_sim_mask(
            joint_pos,
            joint_vel,
            mask,
            joint_ids=self.joint_ids,
        )

        wrist = motion.wrist_pose_w
        if self.pose_binding is not None:
            write_mocap_reset(self.pose_binding, ids, wrist)

        local = motion.object_pose
        wrist_pos = wrist[:, :3]
        wrist_quat = wrist[:, 3:]
        world_pos = wrist_pos + _quat_rotate(wrist_quat, local[:, :3])
        world_quat = _quat_multiply(wrist_quat, local[:, 3:])
        scale = get_curriculum_value(env, curriculum_term, 0.0) if curriculum_term else 1.0
        slide = torch.empty((ids.numel(), 1), device=env.device).uniform_(
            *(float(x) * scale for x in object_reset_axial_slide_range)
        )
        roll = torch.empty(ids.numel(), device=env.device).uniform_(
            *(float(x) * scale for x in object_reset_roll_range)
        )
        axis = torch.zeros((ids.numel(), 3), device=env.device)
        axis[:, 0] = 1.0
        world_pos[ids] += _quat_rotate(world_quat[ids], axis * slide)
        roll_quat = torch.cat((torch.cos(roll[:, None] / 2), axis * torch.sin(roll[:, None] / 2)), -1)
        noisy_quat = _quat_multiply(world_quat[ids], roll_quat)
        world_quat[ids] = noisy_quat * torch.where(noisy_quat[:, :1] < 0, -1., 1.)
        root = self.object.data.default_root_state(env.device).clone()
        if root.ndim == 1:
            root = root.unsqueeze(0).expand(env.num_envs, -1).clone()
        elif root.shape[0] == 1 and env.num_envs != 1:
            root = root.expand(env.num_envs, -1).clone()
        root[ids, :3] = world_pos[ids]
        root[ids, 3:7] = world_quat[ids]
        root[ids, 7:10] = motion.object_linvel_w[ids]
        root[ids, 10:13] = motion.object_angvel_w[ids]
        self.object.write_root_state_tensor_to_sim(root, mask)


__all__ = ["ResetPenReference"]

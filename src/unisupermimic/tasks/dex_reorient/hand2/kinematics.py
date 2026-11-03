"""Task-level pose helpers and differentiable Hand 2 forward kinematics."""

from __future__ import annotations

import torch

from .math import conjugate, multiply, rotate


class Hand2Kinematics:
    """Differentiable FK for the palm-rooted Hand 2 MJCF subtree.

    This is the same one-hinge-per-body recursion used by the source
    SuperMimic reader.  Keeping it here lets imagined cage, fingertip and
    tip-slide terms use the predicted joint state instead of a fixed AABB.
    """

    def __init__(self, model, body_names, joint_names, tip_site_names, device):
        import mujoco
        import numpy as np

        self.body_ids = [int(model.body(name).id) for name in body_names]
        self.root = self.body_ids[0]
        local = {body: index for index, body in enumerate(self.body_ids)}
        joints = {int(model.joint(name).id): index for index, name in enumerate(joint_names)}
        self.entries = []
        for body in self.body_ids[1:]:
            parent = int(model.body_parentid[body])
            if parent not in local:
                raise ValueError("Hand 2 FK bodies must form a connected subtree")
            nj = int(model.body_jntnum[body])
            jid = int(model.body_jntadr[body])
            if nj > 1 or (nj and model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE):
                raise ValueError("Hand 2 FK supports one hinge per body")
            def tensor(value):
                return torch.as_tensor(
                    np.asarray(value), dtype=torch.float32, device=device
                )
            self.entries.append(
                (
                    local[parent],
                    joints[jid] if nj else -1,
                    tensor(model.body_pos[body]),
                    self._quat_to_mat(tensor(model.body_quat[body])),
                    tensor(model.jnt_axis[jid] if nj else (0.0, 0.0, 1.0)),
                    tensor(model.jnt_pos[jid] if nj else (0.0, 0.0, 0.0)),
                )
            )
        tip_bodies = []
        tip_offsets = []
        for name in tip_site_names:
            site = int(model.site(name).id)
            body = int(model.site_bodyid[site])
            if body not in local:
                raise ValueError(f"Tip site {name!r} is outside the Hand 2 subtree")
            tip_bodies.append(local[body])
            tip_offsets.append(tensor(model.site_pos[site]))
        self.tip_bodies = torch.tensor(tip_bodies, dtype=torch.long, device=device)
        self.tip_offset = torch.stack(tip_offsets)

        depths = [0]
        for parent, *_ in self.entries:
            depths.append(depths[parent] + 1)
        order = sorted(range(len(self.body_ids)), key=lambda index: depths[index])
        position = {body: index for index, body in enumerate(order)}
        self.restore = [position[index] for index in range(len(order))]
        self.levels = []
        for depth in range(1, max(depths) + 1):
            entries = [self.entries[index - 1] for index in order if depths[index] == depth]
            if not entries:
                continue
            parents, joints_at, offsets, rotations, axes, anchors = zip(*entries, strict=True)
            self.levels.append(
                {
                    "parent": torch.tensor([position[value] for value in parents], device=device),
                    "joint": torch.tensor([max(0, value) for value in joints_at], device=device),
                    "moving": torch.tensor([value >= 0 for value in joints_at], device=device),
                    "off": torch.stack(offsets),
                    "rot": torch.stack(rotations),
                    "axis": torch.stack(axes),
                    "anchor": torch.stack(anchors),
                }
            )

    @staticmethod
    def _quat_to_mat(quat: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat / torch.linalg.vector_norm(quat).clamp_min(1e-12)
        return torch.stack(
            (
                1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
            )
        ).reshape(3, 3)

    @staticmethod
    def _axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1e-12)
        x, y, z = axis.unbind(-1)
        c, s = torch.cos(angle), torch.sin(angle)
        one = 1 - c
        return torch.stack(
            (
                c + x * x * one, x * y * one - z * s, x * z * one + y * s,
                y * x * one + z * s, c + y * y * one, y * z * one - x * s,
                z * x * one - y * s, z * y * one + x * s, c + z * z * one,
            ), -1
        ).reshape(*angle.shape, 3, 3)

    def forward(self, q: torch.Tensor, qd: torch.Tensor):
        shape = q.shape[:-1]
        p = q.new_zeros(*shape, 1, 3)
        v = torch.zeros_like(p)
        w = torch.zeros_like(p)
        r = torch.eye(3, device=q.device, dtype=q.dtype).expand(*shape, 1, 3, 3)
        for level in self.levels:
            parent, joint = level["parent"], level["joint"]
            pp, rp, vp, wp = p[..., parent, :], r[..., parent, :, :], v[..., parent, :], w[..., parent, :]
            off, rot, axis, anchor = (level[key].to(device=q.device, dtype=q.dtype) for key in ("off", "rot", "axis", "anchor"))
            moving = level["moving"].to(device=q.device)
            angle = q[..., joint] * moving
            speed = qd[..., joint, None] * moving.unsqueeze(-1)
            rr = self._axis_angle(axis, angle)
            base = torch.matmul(rp, rot)
            ri = torch.matmul(base, rr)
            rotated_anchor = torch.matmul(rr, anchor.unsqueeze(-1)).squeeze(-1)
            local_p = off + torch.matmul(rot, (anchor - rotated_anchor).unsqueeze(-1)).squeeze(-1)
            pi = pp + torch.matmul(rp, local_p.unsqueeze(-1)).squeeze(-1)
            joint_local = off + torch.matmul(rot, anchor.unsqueeze(-1)).squeeze(-1)
            joint_p = pp + torch.matmul(rp, joint_local.unsqueeze(-1)).squeeze(-1)
            angular = torch.matmul(base, axis.unsqueeze(-1)).squeeze(-1) * speed
            wi = wp + angular
            vi = vp + torch.linalg.cross(wp, pi - pp, dim=-1) + torch.linalg.cross(angular, pi - joint_p, dim=-1)
            p, r, v, w = torch.cat((p, pi), -2), torch.cat((r, ri), -3), torch.cat((v, vi), -2), torch.cat((w, wi), -2)
        return p[..., self.restore, :], r[..., self.restore, :, :], v[..., self.restore, :], w[..., self.restore, :]

    def tips(self, kin):
        p, r, v, w = kin
        ids = self.tip_bodies.to(device=p.device)
        offset = torch.matmul(r[..., ids, :, :], self.tip_offset.to(device=p.device, dtype=p.dtype).unsqueeze(-1)).squeeze(-1)
        return p[..., ids, :] + offset, v[..., ids, :] + torch.linalg.cross(w[..., ids, :], offset, dim=-1)

    def tip_body_positions(self, kin):
        """Return the distal link body origins used by the privileged frame."""
        return kin[0][..., self.tip_bodies.to(device=kin[0].device), :]


def cube_pose_in_tag(
    tag_position: torch.Tensor,
    tag_quaternion: torch.Tensor,
    cube_position: torch.Tensor,
    cube_quaternion: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Express a cube pose in the wrist-tag frame."""
    inverse = conjugate(tag_quaternion)
    return rotate(inverse, cube_position - tag_position), multiply(inverse, cube_quaternion)


__all__ = ["Hand2Kinematics", "cube_pose_in_tag"]

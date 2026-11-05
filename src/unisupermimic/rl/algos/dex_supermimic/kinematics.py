"""Differentiable hand FK, including fixed links, joint anchors and tip sites."""

import numpy as np
import torch

from .rotations import axis_angle_to_mat, mm, mv, quat_to_mat


class HandKinematics:
    def __init__(self, model, body_ids, joint_ids, tip_site_ids, device):
        import mujoco

        self.body_ids = list(map(int, body_ids))
        self.root = self.body_ids[0]
        self.names = [model.body(i).name for i in self.body_ids]
        joints = {int(j): i for i, j in enumerate(joint_ids)}
        local = {b: i for i, b in enumerate(self.body_ids)}
        self.entries = []
        for b in self.body_ids[1:]:
            if int(model.body_parentid[b]) not in local:
                raise ValueError("Hand bodies must be a connected subtree")
            nj = int(model.body_jntnum[b])
            jid = int(model.body_jntadr[b])
            if nj > 1 or (nj and model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE):
                raise ValueError("Hand FK supports one hinge per body")
            def tensor(x):
                return torch.as_tensor(np.array(x), dtype=torch.float32, device=device)
            self.entries.append(
                (
                    local[int(model.body_parentid[b])],
                    joints[jid] if nj else -1,
                    tensor(model.body_pos[b]),
                    quat_to_mat(tensor(model.body_quat[b])),
                    tensor(model.jnt_axis[jid] if nj else [0, 0, 1]),
                    tensor(model.jnt_pos[jid] if nj else [0, 0, 0]),
                )
            )
        self.tip_bodies = [local[int(model.site_bodyid[i])] for i in tip_site_ids]
        self.tip_offset = torch.as_tensor(
            np.array(model.site_pos[tip_site_ids]), dtype=torch.float32, device=device
        )

        # Batch the five finger chains by tree depth. This reduces Python and GPU
        # launch overhead and keeps the compiled dynamics graph small.
        depths = [0]
        for parent, *_ in self.entries:
            depths.append(depths[parent] + 1)
        order = sorted(range(len(self.body_ids)), key=lambda i: depths[i])
        position = {body: i for i, body in enumerate(order)}
        self.restore = [position[i] for i in range(len(order))]
        self.levels = []
        for depth in range(1, max(depths) + 1):
            entries = [self.entries[i - 1] for i in order if depths[i] == depth]
            parents, joints, off, rot, axis, anchor = zip(*entries, strict=False)
            self.levels.append(
                dict(
                    parent=torch.tensor([position[v] for v in parents], device=device),
                    joint=torch.tensor([max(0, v) for v in joints], device=device),
                    moving=torch.tensor([v >= 0 for v in joints], device=device),
                    off=torch.stack(off),
                    rot=torch.stack(rot),
                    axis=torch.stack(axis),
                    anchor=torch.stack(anchor),
                )
            )

    def forward(self, q, qd):
        shape = q.shape[:-1]
        p = q.new_zeros(*shape, 1, 3)
        v = torch.zeros_like(p)
        w = torch.zeros_like(p)
        r = torch.eye(3, device=q.device, dtype=q.dtype).expand(*shape, 1, 3, 3)
        for level in self.levels:
            parent, joint = level["parent"], level["joint"]
            pp, rp, vp, wp = (
                p[..., parent, :],
                r[..., parent, :, :],
                v[..., parent, :],
                w[..., parent, :],
            )
            off, rot, axis, anchor = (
                level[k] for k in ("off", "rot", "axis", "anchor")
            )
            angle = q[..., joint] * level["moving"]
            speed = qd[..., joint, None] * level["moving"].unsqueeze(-1)
            rr = axis_angle_to_mat(axis, angle)
            base = mm(rp, rot)
            ri = mm(base, rr)
            local_p = off + mv(rot, anchor - mv(rr, anchor))
            pi = pp + mv(rp, local_p)
            joint_p = pp + mv(rp, off + mv(rot, anchor))
            angular = mv(base, axis) * speed
            wi = wp + angular
            vi = (
                vp
                + torch.cross(wp, pi - pp, dim=-1)
                + torch.cross(angular, pi - joint_p, dim=-1)
            )
            p, r, v, w = (
                torch.cat((p, pi), -2),
                torch.cat((r, ri), -3),
                torch.cat((v, vi), -2),
                torch.cat((w, wi), -2),
            )
        ids = self.restore
        return p[..., ids, :], r[..., ids, :, :], v[..., ids, :], w[..., ids, :]

    def tips(self, kin):
        p, r, v, w = kin
        ids = self.tip_bodies
        offset = mv(r[..., ids, :, :], self.tip_offset)
        return p[..., ids, :] + offset, v[..., ids, :] + torch.cross(
            w[..., ids, :], offset, dim=-1
        )

    def features(self, q, qd):
        p, r, v, w = self.forward(q, qd)
        return torch.cat((p, r[..., :2, :].flatten(-2), v, w), -1).flatten(-2)

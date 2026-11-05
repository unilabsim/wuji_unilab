"""Analytic rewards and task memory. Hard mode matches the original reward terms.

Soft mode supplies gradients for discrete cage/hold/contact decisions in imagination.
The simulator's reward is evaluated before command updates; this ordering is preserved.
"""

import math

import torch

from .rotations import mm, mv
from .state import unpack


def angle_error(r, goal):
    rel = mm(r, goal.transpose(-1, -2))
    # atan2 avoids acos' infinite derivative at aligned orientations.
    vee = (
        torch.stack(
            (
                rel[..., 2, 1] - rel[..., 1, 2],
                rel[..., 0, 2] - rel[..., 2, 0],
                rel[..., 1, 0] - rel[..., 0, 1],
            ),
            -1,
        )
        * 0.5
    )
    sine = torch.linalg.vector_norm(vee, dim=-1)
    cosine = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).clamp(-1, 1)
    return torch.atan2(sine, cosine)


class DexMDP:
    def __init__(
        self, kin, default, lower, upper, tag_pos, tag_rot, hz=50, effort_limit=None,
        cage_up_axis=1,
    ):
        self.kin = kin
        self.default = default
        self.lower = lower
        self.upper = upper
        self.tag_pos = tag_pos
        self.tag_rot = tag_rot
        self.dt = 1 / hz
        self.alpha = 0.3 if hz == 50 else 0.5
        self.hold_steps = 13 if hz == 50 else 5
        self.switch_steps = 50 if hz == 50 else 20
        self.max_outside = 25 if hz == 50 else 10
        self.max_count = 37.5 if hz == 50 else 15
        self.hold_weight = 4.694 if hz == 50 else 11.4
        self.hz = hz
        self.effort_limit = effort_limit
        self.kp_default = None
        self.kd_default = None
        self.force_limit = None
        self.gear = None
        if cage_up_axis not in (0, 1, 2):
            raise ValueError("cage_up_axis must be 0, 1 or 2")
        self.cage_margin = default.new_full((3,), 0.01)
        self.cage_margin[cage_up_axis] = 0.03
        self.cage_ids = [
            i
            for i, n in enumerate(kin.names)
            if n.rsplit("/", 1)[-1] == "right_palm_link"
            or (
                n.rsplit("/", 1)[-1].startswith("right_finger")
                and n[-5:] in ("link1", "link2", "link3", "link4")
            )
        ]

    def cage(self, x, counter, soft=False, kin=None):
        q, qd, p, *_ = unpack(x)
        kin = self.kin.forward(q, qd) if kin is None else kin
        points = kin[0][..., self.cage_ids, :]
        lo = points.amin(-2) - 0.01
        hi = points.amax(-2) + self.cage_margin
        if soft:
            signed = torch.maximum(lo - p, p - hi).amax(-1, keepdim=True)
            outside = torch.sigmoid(signed / 0.002)
        else:
            outside = ((p < lo) | (p > hi)).any(-1, keepdim=True).to(p.dtype)
        next_counter = (
            outside * (counter + 1) + (1 - outside) * (counter - 0.5).clamp_min(0)
        ).clamp_max(self.max_count)
        return outside * 4 * next_counter / self.max_outside, next_counter

    def pd_torque(self, x, u, context=None):
        """Differentiable position-actuator force at the pre-substep state.

        MuJoCo evaluates the actuator PD law before integrating a physics substep.
        The reward snapshot intentionally captures the same post-step state used
        by the world model, so the analytic force is evaluated from the previous
        rollout state; this also avoids making the torque term depend on the
        predicted post-step velocity.
        """
        q, qd, *_ = unpack(x)
        kp, kd, force_limit, gear = (
            self.kp_default,
            self.kd_default,
            self.force_limit,
            self.gear,
        )
        if context is not None:
            kp_ratio, kd_ratio = context[..., -40:-20], context[..., -20:]
            kp = kp * kp_ratio
            kd = kd * kd_ratio
        tau = gear * (kp * (u - q) + kd * qd)
        if force_limit is not None:
            tau = tau.clamp(-force_limit, force_limit)
        return tau

    def rewards(self, x1, m, a, c, contact, torque, soft=False):
        q, qd, p, r, v, w = unpack(x1)
        kin = self.kin.forward(q, qd)
        tips = self.kin.tips(kin)[1]
        theta = angle_error(r, m["goal"].reshape(-1, 3, 3))
        cage, counter = self.cage(x1, m["cage"], soft, kin)
        tau = (
            torque
            if self.effort_limit is None
            else torque / self.effort_limit.clamp_min(1e-3)
        )
        # contact layout: five tips, palm/proximal, distal, self-collision slots.
        detach = (1 - contact[..., 5]) * contact[..., 6]
        terms = {
            "orientation_alignment": 15
            * (1 - (theta - 0.2).clamp_min(0) / math.pi).clamp_min(0),
            "hand_pose": -0.2 * (q - self.default).square().sum(-1),
            "action_rate": -(1 if self.hz == 20 else 0)
            * (
                (a - m["a1"]).square().sum(-1)
                + (a - 2 * m["a1"] + m["a2"]).square().sum(-1)
            ),
            "action_hf": -(30 if self.hz == 50 else 0)
            * (c - 2 * m["c1"] + m["c2"]).square().sum(-1)
            * (m["age"].squeeze(-1) >= 2),
            "torque": -24 * tau.square().sum(-1),
            "tip_slide": -0.3
            * (
                torch.linalg.vector_norm(tips - v.unsqueeze(-2), dim=-1)
                * contact[..., :5]
            ).sum(-1),
            "cage_escape": -500 * cage.squeeze(-1),
            "finger_collision": -contact[..., 7:].sum(-1),
            "hold_escalation": self.hold_weight
            * (m["window"] * m["within"] * m["reward_snapshot"]).squeeze(-1),
            "palm_detach": 0.5 * detach,
        }
        terms = {k: v * self.dt for k, v in terms.items()}
        return sum(terms.values()), counter, terms

    def command_step(self, x, m, next_goal, soft=False):
        r = unpack(x)[3]
        err = angle_error(r, m["goal"].reshape(-1, 3, 3)).unsqueeze(-1)
        within = torch.sigmoid((0.2 - err) / 0.02) if soft else (err < 0.2).to(x.dtype)
        window = m["window"]
        approaching = 1 - window
        hold = approaching * within * (m["hold"] + 1) + window * m["hold"]
        succeeded = approaching * (
            torch.sigmoid((hold - self.hold_steps + 0.5) / 0.25)
            if soft
            else (hold >= self.hold_steps).to(x.dtype)
        )
        window = window + (1 - window) * succeeded
        timer = (1 - succeeded) * m["window_timer"] + window
        count = (1 - succeeded) * (m["reward_hold"] + window * within)
        switch = window * (
            torch.sigmoid((timer - self.switch_steps + 0.5) / 0.25)
            if soft
            else (timer >= self.switch_steps).to(x.dtype)
        )
        out = dict(m)
        # A boolean switch keeps goal rotations on SO(3). Timers retain soft gradients.
        out["goal"] = torch.where(switch > 0.5, next_goal.flatten(-2), m["goal"])
        out.update(
            hold=(1 - switch) * (1 - succeeded) * hold,
            goal_timer=(1 - switch) * (m["goal_timer"] + 1),
            window=(1 - switch) * window,
            window_timer=(1 - switch) * timer,
            reward_window=timer,
            reward_hold=(1 - switch) * count,
            reward_snapshot=count,
            within=within,
            count=m["count"] + succeeded,
        )
        return out

    def observation(self, x, m):
        q, qd, p, r, v, w = unpack(x)
        center = (self.lower + self.upper) * 0.5
        half = (self.upper - self.lower) * 0.5 + 1e-6
        qn = ((q - center) / half).clamp(-1, 1)
        cn = ((m["c1"] - center) / half).clamp(-1, 1)
        pos = mv(self.tag_rot.transpose(-1, -2), p - self.tag_pos)
        # Original uses rows 1 and 2 of R_tag^-1 R_obj R_goal^-1 R_tag.
        err = mm(
            mm(
                mm(self.tag_rot.transpose(-1, -2), r),
                m["goal"].reshape(-1, 3, 3).transpose(-1, -2),
            ),
            self.tag_rot,
        )
        return torch.cat((qn, qn - cn, pos, err[..., 1:, :].flatten(-2), m["a2"]), -1)

    @staticmethod
    def flatten_history(history):
        # ObservationManager concatenates term histories, not whole-frame histories.
        return torch.cat(
            [part.flatten(-2) for part in history.split((20, 20, 3, 6, 20), -1)], -1
        )

    @staticmethod
    def noisy_observation(obs):
        q, e, p, r, a = obs.split((20, 20, 3, 6, 20), -1)
        q = q + (torch.rand_like(q) * 2 - 1) * 0.06
        # Outliers are observations, never physical-state noise.
        p = torch.where(torch.rand_like(p[..., :1]) < 0.02, torch.rand_like(p) - 0.5, p)
        from .rotations import quat_to_mat

        rand_rot = quat_to_mat(
            torch.nn.functional.normalize(
                torch.randn(*r.shape[:-1], 4, device=r.device), dim=-1
            )
        )
        r = torch.where(
            torch.rand_like(r[..., :1]) < 0.02, rand_rot[..., 1:, :].flatten(-2), r
        )
        p = p + (torch.rand_like(p) * 2 - 1) * 0.008
        r = r + torch.randn_like(r) * 0.05
        return torch.cat((q, e, p, r, a), -1)

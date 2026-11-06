"""Source-aligned differentiable Pen Spin MDP for UniLab.

The analytic formulas and feature order follow Dex-SuperMimic's pen adapter.
Only construction and simulator metadata access are adapted to UniLab; the
reader owns all backend reads so imagined rollouts never depend on mjlab APIs.
"""

from __future__ import annotations

import hashlib
from typing import ClassVar

import torch

from unisupermimic.tasks.pen_spin.commands import LINK_NAMES

from .mdp import DexMDP
from .rotations import axis_angle_to_mat, mm, quat_to_mat
from .state import MEMORY_DIMS, unpack

PEN_MEMORY_DIMS = dict(
    MEMORY_DIMS,
    cage=20,
    time=1,
    start=1,
    end=1,
    limits=20,
    camera_raw=6,
    camera_obs=6,
    camera_holding=1,
    camera_ticks=1,
    camera_bias_pos=3,
    camera_bias_rot=9,
    encoder_bias=20,
    frozen_target=20,
    mask=20,
    mask_remaining=1,
    mask_active=1,
)


def flatten_pen_memory(memory):
    return torch.cat([memory[key] for key in PEN_MEMORY_DIMS], -1)


def unflatten_pen_memory(value):
    return dict(zip(PEN_MEMORY_DIMS, value.split(tuple(PEN_MEMORY_DIMS.values()), -1), strict=True))


def _curriculum_value(env, name, default=1.0):
    """Read the same scalar manager state used by the source action mask."""
    state = getattr(getattr(env, "curriculum_manager", None), "_curriculum_state", {})
    value = state.get(name)
    if value is None:
        return default
    if isinstance(value, dict):
        if not value:
            return default
        value = next(iter(value.values()))
    if isinstance(value, torch.Tensor):
        return value.mean().item() if value.numel() else default
    return float(value)


def _row(value):
    return value[0] if value.ndim >= 2 else value


class PenMDP(DexMDP):
    def __init__(self, reader):
        env = reader.env
        device = env.device
        default = reader.default_joint_pos.clone()
        super().__init__(
            reader.kin,
            default,
            reader.lower.clone(),
            reader.upper.clone(),
            torch.zeros(3, device=device),
            torch.eye(3, device=device),
            50,
        )
        self.motion = env.command_manager.get_term("motion")
        self.mask_cfg = reader.action_cfg
        self.alpha = float(self.mask_cfg.alpha)
        self.action_scale = float(getattr(self.mask_cfg, "action_scale", 0.5))
        self.residual_clip = tuple(getattr(self.mask_cfg, "residual_clip", (-0.6, 0.6)))
        self.link_ids = [reader.kin.names.index(name) for name in LINK_NAMES]
        defaults = reader.actuator_defaults
        self.nominal_limits = _row(defaults["force_limit"]).to(device).clone()
        for field, name in (
            ("kp", "kp_default"),
            ("kd", "kd_default"),
            ("force_limit", "force_limit"),
            ("gear", "gear"),
        ):
            setattr(self, name, _row(defaults[field]).to(device).clone())
        self.radius = float(getattr(self.motion.cfg, "axis_point_radius_m", 0.10))
        self.bound = float(self.motion.cfg.object_position_tanh_scale_m)
        self.camera_enabled = bool(env.cfg.observations["policy"].enable_corruption)
        self.env = env
        digest = hashlib.sha256()
        for field in ("joint_pos", "object_pose", "link_pos"):
            digest.update(self.motion._frames[field].detach().cpu().numpy().tobytes())
        digest.update(self.motion._trajectory_ends.detach().cpu().numpy().tobytes())
        self.reference_signature = digest.hexdigest()

    def control_target(self, action, memory, bias):
        residual = self.alpha * (self.action_scale * action).clamp(*self.residual_clip) + (1 - self.alpha) * memory["c1"]
        reference = self.reference(memory)[0]
        target = reference + residual - bias
        cfg = self.mask_cfg
        mask_prob = float(getattr(cfg, "mask_prob", 0.0))
        if mask_prob > 0:
            term = getattr(cfg, "mask_curriculum_term", None)
            scale = _curriculum_value(self.env, term, 0.0) if term else 1.0
            active_at_start = memory["mask_active"] > 0
            remaining = memory["mask_remaining"] - active_at_start.float()
            active = active_at_start & (remaining > 0)
            fire = (~active_at_start) & (torch.rand_like(remaining) < mask_prob * scale)
            chosen = torch.rand_like(action).topk(cfg.mask_num_dofs, dim=-1).indices
            selected = torch.zeros_like(action).scatter(-1, chosen, 1)
            mask = torch.where(fire, selected, memory["mask"] * active.float())
            maximum = max(
                round(cfg.mask_duration_min + (cfg.mask_duration_max - cfg.mask_duration_min) * scale),
                cfg.mask_duration_min,
            )
            duration = torch.randint(cfg.mask_duration_min, maximum + 1, remaining.shape, device=action.device).float()
            remaining = torch.where(fire, duration, remaining)
            active = active | fire
            target = torch.where((mask > 0) & (memory["age"] > 0), memory["frozen_target"], target)
            memory.update(mask=mask, mask_remaining=remaining, mask_active=active.float())
        memory["frozen_target"] = target
        return residual, target

    def reference(self, memory):
        frame = (memory["start"] + memory["time"].clamp_min(0)).minimum(memory["end"] - 1).long().squeeze(-1)
        frames = self.motion._frames
        return frames["joint_pos"][frame], frames["object_pose"][frame], frames["link_pos"][frame]

    def points(self, position, rotation, bounded=False):
        offset = rotation[..., :, 0] * self.radius
        points = torch.stack((position + offset, position - offset), -2)
        return self.bound * torch.tanh(points / self.bound) if bounded else points

    def reference_points(self, pose, bounded=False):
        return self.points(pose[..., :3], quat_to_mat(pose[..., 3:]), bounded)

    def command_step(self, x, memory, next_goal, soft=False):
        return dict(memory, time=memory["time"] + 1)

    def pd_torque(self, x, u, context=None, *, limits=None):
        q, qd, *_ = unpack(x)
        kp, kd = self.kp_default, self.kd_default
        if context is not None:
            kp = kp * context[..., -40:-20]
            kd = kd * context[..., -20:]
        if limits is None:
            limits = self.force_limit
        return (kp * (u - q) + kd * qd).clamp(-limits, limits)

    def rewards(self, x1, memory, action, residual, contact, torque, soft=False):
        q, qd, pos, rot, *_ = unpack(x1)
        ref_q, ref_pose, ref_links = self.reference(memory)
        links = self.kin.forward(q, qd)[0][..., self.link_ids, :]
        point_error = (self.points(pos, rot) - self.reference_points(ref_pose)).norm(dim=-1).mean(-1)
        load = (torque.abs() / memory["limits"].clamp_min(1e-8)).clamp(max=1)
        charge = (memory["cage"] + self.dt * (load - memory["cage"] / 2)).clamp(0, 1)
        terms = {
            "object_pos": 2 * torch.exp(-((pos - ref_pose[..., :3]).norm(dim=-1) / 0.05).square()),
            "object_rot": 8 * torch.exp(-(point_error / 0.04).square()),
            "link_pos": torch.exp(-((links - ref_links).norm(dim=-1) / 0.025).square()).mean(-1),
            "joint_pos": torch.exp(
                -((q - ref_q).abs() / (self.upper - self.lower).clamp_min(1e-6) / 0.1).square()
            ).mean(-1),
            "action_rate": -0.1
            * ((action - memory["a1"]).square() + (action - 2 * memory["a1"] + memory["a2"]).square()).sum(-1),
            "torque": -(torque / self.nominal_limits).square().sum(-1),
            "overcurrent": -4 * (charge - 0.8).clamp_min(0).sum(-1) / 0.2,
            "finger_collision": -contact.sum(-1),
        }
        terms = {name: value * self.dt for name, value in terms.items()}
        return sum(terms.values()), charge, terms

    def termination(self, x, memory):
        _, _, pos, rot, *_ = unpack(x)
        _, ref, _ = self.reference(memory)
        position = torch.sigmoid(((pos - ref[..., :3]).norm(dim=-1) - 0.05) / 0.002)
        cosine = (rot[..., :, 0] * quat_to_mat(ref[..., 3:])[..., :, 0]).sum(-1)
        orientation = torch.sigmoid((0.5 - cosine) / 0.02)
        return 1 - (1 - position) * (1 - orientation)

    def observation(self, x, memory):
        q, qd, pos, rot, *_ = unpack(x)
        ref_q, ref_pose, ref_links = self.reference(memory)
        command = torch.cat((ref_q, self.reference_points(ref_pose, True).flatten(-2), ref_links.flatten(-2)), -1)
        return torch.cat(
            (command, q - self.default, qd * 0.25, self.points(pos, rot, True).flatten(-2), memory["a1"]), -1
        )

    def policy_observation(self, x, memory, auxiliary):
        obs = self.observation(x, memory)
        if not self.camera_enabled:
            return obs
        q, qd, pos, rot, *_ = unpack(x)
        bias_rot = memory["camera_bias_rot"].reshape(-1, 3, 3)
        axis = torch.nn.functional.normalize(torch.randn_like(pos), dim=-1)
        angle = pos.new_empty(pos.shape[0]).uniform_(-0.015, 0.015)
        rotation = mm(axis_angle_to_mat(axis, angle), mm(bias_rot, rot))
        position = pos + memory["camera_bias_pos"] + torch.empty_like(pos).uniform_(-0.002, 0.002)
        raw = self.points(position, rotation, True).flatten(-2)
        roll = torch.rand_like(memory["camera_holding"])
        holding = torch.where(memory["camera_holding"] > 0, (roll < 0.78) & (memory["camera_ticks"] < 40), roll < 0.03)
        camera = torch.where(holding, memory["camera_obs"], memory["camera_raw"])
        memory.update(
            camera_raw=raw,
            camera_obs=camera,
            camera_holding=holding.float(),
            camera_ticks=torch.where(holding, memory["camera_ticks"] + 1, 0),
        )
        return torch.cat(
            (
                obs[..., :89],
                q - self.default + memory["encoder_bias"] + torch.empty_like(q).uniform_(-0.015, 0.015),
                qd * 0.25 + torch.empty_like(qd).uniform_(-0.05, 0.05),
                camera,
                memory["a1"],
            ),
            -1,
        )

    @staticmethod
    def flatten_history(history):
        return history[:, -1]


class PenObservations:
    base_terms: ClassVar[dict[str, int]] = {
        "motion": 89,
        "joint_pos": 20,
        "joint_vel": 20,
        "object_pose": 6,
        "last_action": 20,
        "link_pos": 63,
        "joint_position_error": 20,
        "link_position_error": 63,
    }

    def __init__(self, reader):
        self.reader, self.mdp = reader, reader.mdp
        self.frame_dim = sum(self.base_terms.values())
        self.context_dim = reader.xi.shape[-1] + self.frame_dim + 21

    def auxiliary(self):
        reader = self.reader
        n = reader.env.num_envs
        prefix = torch.cat((reader.palm_pos, reader.palm_rot.flatten(1), torch.zeros(n, 9, device=reader.device)), -1)
        # Reader publishes the 28 robot-body and one pen-body world wrenches in
        # model order: 29 bodies * six force/torque coordinates = 174 features.
        return torch.cat((prefix, reader.external_wrench()), -1).clone()

    @staticmethod
    def disturbance(auxiliary):
        return auxiliary[:, 21:]

    def frame(self, x, memory, auxiliary):
        q, qd, *_ = unpack(x)
        ref_q, _, ref_links = self.mdp.reference(memory)
        links = self.mdp.kin.forward(q, qd)[0][:, self.mdp.link_ids].flatten(1)
        return torch.cat((self.mdp.observation(x, memory), links, q - ref_q, links - ref_links.flatten(1)), -1)

    def context(self, x, memory, auxiliary, simulator_context):
        return torch.cat(
            (self.frame(x, memory, auxiliary), memory["cage"], memory["time"] * self.mdp.dt, simulator_context), -1
        )

    def critic(self, context):
        return context[..., : self.frame_dim]

    @staticmethod
    def dynamics(x, simulator_context, disturbance):
        return torch.cat((x, simulator_context, disturbance), -1)


def __getattr__(name):
    if name == "PenSimReader":
        from .sim import PenSimReader

        return PenSimReader
    raise AttributeError(name)


__all__ = ["PEN_MEMORY_DIMS", "flatten_pen_memory", "unflatten_pen_memory", "PenMDP", "PenObservations"]

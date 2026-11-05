# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Source pen and fingertip wrench state machines on UniLab's interval plan.

The two source producers retain their independent timers and random sampling.
Their outputs are committed together because a UniLab interval plan replaces
previously staged wrenches. No velocity perturbation is substituted for force.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import torch
from unilab.managers import ManagerTermBase
from unisim.dr.interval import INTERVAL_TERM_BODY_FORCE, INTERVAL_TERM_BODY_TORQUE
from unisim.dr.types import IntervalRandomizationPlan, IntervalTermOp

from .commands import _quat_conjugate, _quat_multiply, _quat_rotate
from .curriculums import get_curriculum_value
from .reset_bridge import get_pen_physics_model

OBJECT_BODY_NAMES = ("object/cube",)
FINGERTIP_BODY_NAMES = tuple(f"robot/right_finger{finger}_link4" for finger in range(1, 6))


def _scene_model(env):
    """Read actual compiled geometry metadata through the task's cold bridge."""
    return get_pen_physics_model(env)


def _sample_uniform(lower, upper, count, device):
    return torch.rand(count, device=device) * (upper - lower) + lower


def _body_link_quat(env, entity_name, body_names):
    return env.scene.read_plan.body_tensor_view(entity_name, body_names).quat_w


def _model_field(env, field):
    try:
        return env._pen_model_fields[field]
    except (AttributeError, KeyError) as exc:
        raise RuntimeError(
            f"Pen disturbances require startup physics metadata {field!r}"
        ) from exc


def _resolve_bodies(env, entity_name, body_names):
    asset = env.scene[entity_name]
    ids, found = asset.find_bodies(body_names, preserve_order=True)
    if tuple(found) != tuple(body_names):
        raise ValueError(f"Pen disturbance bodies do not match: {tuple(found)}")
    local_ids, backend_ids = asset.bind_body_wrench(
        ids, torque=True, term_name="pen_disturbances"
    )
    return asset, local_ids, torch.tensor(np.array(backend_ids), dtype=torch.long)


class apply_object_disturbance_wrench:
    """The original per-environment active/silent object-wrench process."""

    def __init__(self, cfg, env):
        del cfg
        self._asset, self._body_ids, backend_ids = _resolve_bodies(
            env, "object", OBJECT_BODY_NAMES
        )
        self._global_body_ids = backend_ids.to(env.device)
        self._num_bodies = len(OBJECT_BODY_NAMES)
        self._num_envs = env.num_envs
        self._device = torch.device(env.device)
        shape = (self._num_envs, self._num_bodies, 3)
        self._force = torch.zeros(shape, device=self._device)
        self._torque = torch.zeros(shape, device=self._device)
        self._phase_active = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
        self._phase_time_left = torch.zeros(self._num_envs, device=self._device)
        self._decay_rate = torch.ones(self._num_envs, device=self._device)
        self._gravity_mag = float(np.linalg.norm(_scene_model(env).opt.gravity))

    def __call__(
        self,
        env,
        env_ids,
        *,
        grav_mult_range: tuple[float, float],
        angular_accel: float,
        active_s_range: tuple[float, float],
        silence_s_range: tuple[float, float],
        decay_to_frac: float,
        warmup_time_s: float,
        curriculum_term: str,
        asset_cfg=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``grav_mult_range`` draws the linear acceleration per pulse in multiples
        of g; the angular channel follows the same draw with ``angular_accel`` as
        the peak, keeping the force/torque ratio constant.

        The pulse state machine runs continuously after the episode warmup."""
        del env_ids, asset_cfg
        device = self._device
        step_dt = float(env.step_dt)
        scale = get_curriculum_value(env, curriculum_term, 0.0)
        angular_accel = float(angular_accel) * scale

        episode_time = env.episode_length_buf.float() * step_dt
        warmup_mask = episode_time < float(warmup_time_s)
        disturb_mask = ~warmup_mask
        self._phase_time_left[disturb_mask] -= step_dt
        expired = disturb_mask & (self._phase_time_left <= 0.0)
        was_active = self._phase_active.clone()

        continuing = was_active & disturb_mask
        if continuing.any():
            rate = self._decay_rate[continuing].view(-1, 1, 1)
            self._force[continuing] *= rate
            self._torque[continuing] *= rate

        to_active = expired & (~was_active)
        if to_active.any():
            ids = to_active.nonzero(as_tuple=False).squeeze(-1)
            body_mass = self._selected_body_field(env, ids, "body_mass")
            force_dir = self._random_unit_dirs(ids.numel(), device)
            lo, hi = float(grav_mult_range[0]), float(grav_mult_range[1])
            mult = torch.rand((ids.numel(), 1, 1), device=device) * (hi - lo) + lo
            pulse_linear_accel = mult * (self._gravity_mag * scale)
            pulse_angular_accel = (mult / max(hi, 1e-6)) * angular_accel
            self._force[ids] = force_dir * (pulse_linear_accel * body_mass.unsqueeze(-1))

            body_inertia = self._selected_body_field(env, ids, "body_inertia")
            body_iquat = self._selected_body_field(env, ids, "body_iquat")
            world_quat = _body_link_quat(env, "object", OBJECT_BODY_NAMES)[ids]
            principal_to_world = _quat_multiply(world_quat, body_iquat)
            target_world = self._random_unit_dirs(ids.numel(), device) * pulse_angular_accel
            torque_principal = body_inertia * _quat_apply_inverse(
                principal_to_world, target_world
            )
            self._torque[ids] = _quat_rotate(principal_to_world, torque_principal)
            self._phase_active[ids] = True
            active_duration = _sample_uniform(*map(float, active_s_range), ids.numel(), device)
            self._phase_time_left[ids] = active_duration
            self._decay_rate[ids] = float(decay_to_frac) ** (
                step_dt / active_duration.clamp_min(step_dt)
            )
        to_silence = expired & was_active
        if to_silence.any():
            ids = to_silence.nonzero(as_tuple=False).squeeze(-1)
            self._force[ids] = 0.0
            self._torque[ids] = 0.0
            self._phase_active[ids] = False
            self._phase_time_left[ids] = _sample_uniform(
                *map(float, silence_s_range), ids.numel(), device
            )

        force_out = self._force.clone()
        torque_out = self._torque.clone()
        if warmup_mask.any():
            ids = warmup_mask.nonzero(as_tuple=False).squeeze(-1)
            force_out[ids] = 0.0
            torque_out[ids] = 0.0

        return force_out, torque_out

    def reset(self, env_ids=None) -> None:
        idx = slice(None) if env_ids is None else env_ids
        self._force[idx] = 0.0
        self._torque[idx] = 0.0
        self._phase_active[idx] = False
        self._phase_time_left[idx] = 0.0
        self._decay_rate[idx] = 1.0

    def _random_unit_dirs(self, count: int, device) -> torch.Tensor:
        direction = torch.randn((count, self._num_bodies, 3), device=device)
        return direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def _selected_body_field(self, env, ids, field):
        env_grid, body_grid = torch.meshgrid(ids, self._global_body_ids, indexing="ij")
        return _model_field(env, field)[env_grid, body_grid]


def _quat_apply_inverse(quaternion, vector):
    return _quat_rotate(_quat_conjugate(quaternion), vector)


class apply_link_random_force:
    """The original independent per-fingertip bump process."""

    def __init__(self, cfg, env):
        self._asset, self._body_ids, backend_ids = _resolve_bodies(
            env, "robot", FINGERTIP_BODY_NAMES
        )
        self._global_body_ids = backend_ids.to(env.device)
        self._num_envs = env.num_envs
        self._num_bodies = len(FINGERTIP_BODY_NAMES)
        self._device = torch.device(env.device)
        self._step_dt = float(env.step_dt)
        self._cooldown_s = tuple(cfg.params["cooldown_s"])
        self._gravity_mag = float(np.linalg.norm(_scene_model(env).opt.gravity))
        self._force_scale = self._body_scale(cfg.params.get("force_scale"), "force_scale")
        self._link_radius = self._compute_link_radius(env) * self._body_scale(
            cfg.params.get("offset_radius_scale"), "offset_radius_scale"
        )
        shape = (self._num_envs, self._num_bodies)
        self._active = torch.zeros(shape, device=self._device, dtype=torch.bool)
        self._time_remaining = torch.zeros(shape, device=self._device)
        self._interval_time_left = torch.zeros(shape, device=self._device)
        self._force = torch.zeros((*shape, 3), device=self._device)
        self._torque = torch.zeros((*shape, 3), device=self._device)

    def _body_scale(self, scales, name) -> torch.Tensor:
        """Resolve coefficients by selected body name, independent of selector order."""
        ids = (
            range(self._asset.num_bodies)[self._body_ids]
            if isinstance(self._body_ids, slice)
            else self._body_ids.tolist()
        )
        names = [self._asset.body_names[i] for i in ids]
        scales = {} if scales is None else scales
        unknown = scales.keys() - set(names)
        if unknown:
            raise ValueError(f"{name} contains unselected bodies: {sorted(unknown)}")
        values = [float(scales.get(body, 1.0)) for body in names]
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError(f"{name} must contain finite nonnegative coefficients")
        return torch.tensor(values, device=self._device, dtype=torch.float32)

    def _compute_link_radius(self, env) -> torch.Tensor:
        """Per-link offset radius from each link's geom bounding sphere (rbound)."""
        mjm = _scene_model(env)
        radii = [
            float(mjm.geom_rbound[mjm.geom_bodyid == bid].max())
            for bid in self._global_body_ids.tolist()
        ]
        return torch.tensor(radii, device=self._device, dtype=torch.float32)

    def __call__(
        self,
        env,
        env_ids,
        *,
        grav_mult_range: tuple[float, float],
        duration_s: tuple[float, float],
        cooldown_s: tuple[float, float],
        asset_cfg=None,
        force_scale: dict[str, float] | None = None,
        offset_radius_scale: dict[str, float] | None = None,
        curriculum_term: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Scale maps are bound once at construction; step mode operates on all envs.
        del env_ids, asset_cfg, force_scale, offset_radius_scale
        dt = self._step_dt
        device = self._device

        active, remaining, interval = (
            self._active,
            self._time_remaining,
            self._interval_time_left,
        )

        # Expire active bumps -> zero that cell's wrench and start its cooldown.
        remaining[active] -= dt
        expired = active & (remaining <= 0)
        if expired.any():
            self._force[expired] = 0.0
            self._torque[expired] = 0.0
            active[expired] = False
            remaining[expired] = 0.0
            lo, hi = cooldown_s
            interval[expired] = torch.rand(int(expired.sum()), device=device) * (hi - lo) + lo

        # Advance the cooldown clock on idle cells only.
        interval[~active] -= dt
        fire = (~active) & (interval <= 0)
        if fire.any():
            k = int(fire.sum())

            # Random isotropic direction, one per firing (env, body) cell.
            f_dir = torch.randn((k, 3), device=device)
            f_dir = f_dir / f_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)

            # Magnitude = mult * g * mass, scaled by the curriculum when configured.
            lo, hi = grav_mult_range
            mult = torch.rand((k, 1), device=device) * (hi - lo) + lo
            if curriculum_term is not None:
                mult = mult * get_curriculum_value(env, curriculum_term, 0.0)
            mass = _model_field(env, "body_mass")[:, self._global_body_ids][fire].unsqueeze(
                -1
            )  # (k,1)
            scale = self._force_scale.expand(self._num_envs, -1)[fire].unsqueeze(-1)
            forces = f_dir * (mult * self._gravity_mag * mass * scale)

            # Random application point inside the link bounding sphere (uniform in ball).
            off = torch.randn((k, 3), device=device)
            off = off / off.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            u = torch.rand((k, 1), device=device) ** (1.0 / 3.0)
            radius = (
                self._link_radius.view(1, -1).expand(self._num_envs, -1)[fire].unsqueeze(-1)
            )
            offset_local = off * u * radius
            link_quat = _body_link_quat(env, "robot", FINGERTIP_BODY_NAMES)
            inertial_quat = _model_field(env, "body_iquat")[:, self._global_body_ids]
            body_quat = _quat_multiply(link_quat, inertial_quat)[fire]  # (k,4)
            offset_w = _quat_rotate(body_quat, offset_local)

            self._force[fire] = forces
            self._torque[fire] = torch.cross(offset_w, forces, dim=-1)

            dl, dh = duration_s
            remaining[fire] = torch.rand(k, device=device) * (dh - dl) + dl
            active[fire] = True
            lo, hi = cooldown_s
            interval[fire] = torch.rand(k, device=device) * (hi - lo) + lo

        # Independent cells cannot be selected through env_ids/body_ids; idle cells hold zero.
        return self._force, self._torque

    def reset(self, env_ids=None) -> None:
        """Clear the per-cell wrench and seed each cell's clock with a random phase.

        Zeroing ``_interval_time_left`` would fire all five tips together on the
        step after every reset.
        """
        ids = slice(None) if env_ids is None else env_ids
        for buf in (self._active, self._time_remaining, self._force, self._torque):
            buf[ids] = 0
        hi = float(self._cooldown_s[1])
        phase = self._interval_time_left[ids]
        self._interval_time_left[ids] = torch.rand_like(phase) * hi


class PenDisturbances(ManagerTermBase):
    """Compose both source wrench producers into one backend transaction."""

    entity_name = "robot"
    tensor_body_names = FINGERTIP_BODY_NAMES

    def __init__(self, cfg, env):
        super().__init__(env)
        params = cfg.params or {}
        self._object_params = {
            "grav_mult_range": (1.0, 2.0),
            "angular_accel": 375.0,
            "active_s_range": (0.2, 0.4),
            "silence_s_range": (0.5, 1.2),
            "decay_to_frac": 0.02,
            "warmup_time_s": 1.0,
            "curriculum_term": "disturbance_success_rate",
            **dict(params.get("object_params") or {}),
        }
        self._fingertip_params = {
            "grav_mult_range": (0.0, 4.0),
            "force_scale": {name: 2.8 if index == 0 else 1.6 for index, name in enumerate(FINGERTIP_BODY_NAMES)},
            "offset_radius_scale": {name: 0.9 for name in FINGERTIP_BODY_NAMES},
            "duration_s": (0.1, 0.3),
            "cooldown_s": (0.5, 1.5),
            "curriculum_term": "disturbance_success_rate",
            **dict(params.get("fingertip_params") or {}),
        }
        self.object = apply_object_disturbance_wrench(SimpleNamespace(params=self._object_params), env)
        self.fingertips = apply_link_random_force(SimpleNamespace(params=self._fingertip_params), env)
        self._body_ids = torch.cat((
            self.object._global_body_ids.cpu(), self.fingertips._global_body_ids.cpu()
        ))
        self._body_columns = self._body_ids.to(env.device)
        if len(set(self._body_ids.tolist())) != self._body_ids.numel():
            raise ValueError("Pen disturbance producers must select disjoint bodies")
        self._wrench = torch.zeros((env.num_envs, _scene_model(env).nbody, 6), device=env.device)
        env._pen_external_wrench = self._wrench
        self._has_staged = False

    def __call__(self, env, env_ids=None, *, object_params=None, fingertip_params=None):
        del object_params, fingertip_params
        object_force, object_torque = self.object(env, env_ids, **self._object_params)
        tip_force, tip_torque = self.fingertips(env, env_ids, **self._fingertip_params)
        forces = torch.cat((object_force, tip_force), dim=1)
        torques = torch.cat((object_torque, tip_torque), dim=1)
        self._wrench[:, self._body_columns, :3] = forces
        self._wrench[:, self._body_columns, 3:] = torques
        self._commit()

    def _commit(self):
        selected = self._wrench[:, self._body_columns]
        self._env.backend.apply_interval_randomization(IntervalRandomizationPlan(ops=(
            IntervalTermOp(INTERVAL_TERM_BODY_FORCE, selected[..., :3], body_ids=self._body_ids),
            IntervalTermOp(INTERVAL_TERM_BODY_TORQUE, selected[..., 3:], body_ids=self._body_ids),
        )))
        self._has_staged = True

    def reset(self, env_ids=None):
        self.object.reset(env_ids)
        self.fingertips.reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._wrench[ids] = 0.0
        if self._has_staged:
            self._commit()


__all__ = ["PenDisturbances", "apply_object_disturbance_wrench", "apply_link_random_force"]

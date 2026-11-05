# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Reference-plus-residual EMA controller used by source pen spin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from unilab.base.entity import Entity
from unilab.base.reset_selectors import resolve_reset_env_ids
from unilab.managers import ActionTerm, ActionTermCfg

from .commands import task


@dataclass(kw_only=True)
class PenSpinActionCfg(ActionTermCfg):
    command_name: str = "motion"
    action_scale: float = 0.5
    residual_clip: tuple[float, float] = (-0.6, 0.6)
    ema_alpha: float = 0.5
    mask_prob: float = 0.0
    mask_num_dofs: int = 3
    mask_duration_min: int = 1
    mask_duration_max: int = 10
    mask_curriculum_term: str | None = None

    def build(self, env):
        return PenSpinAction(self, env)


class PenSpinAction(ActionTerm):
    cfg: PenSpinActionCfg

    def __init__(self, cfg: PenSpinActionCfg, env):
        super().__init__(cfg, env)
        self.robot = cast(Entity, self._entity)
        default = self.robot.data.default_joint_pos(env.device)
        default = default[0] if default.ndim >= 2 else default
        self.default = default.repeat(self.num_envs, 1)
        self._raw = torch.zeros_like(self.default)
        self._processed = torch.zeros_like(self.default)
        self._target = self.default.clone()
        self._previous_target = self.default.clone()
        self._targets_valid = torch.zeros(self.num_envs, device=env.device, dtype=torch.bool)
        self._mask_active = torch.zeros_like(self._targets_valid)
        self._mask_remaining = torch.zeros(self.num_envs, device=env.device, dtype=torch.long)
        self._mask_dofs = torch.zeros_like(self._raw, dtype=torch.bool)
        if not 0.0 <= cfg.ema_alpha <= 1.0 or not 0.0 <= cfg.mask_prob <= 1.0:
            raise ValueError("Pen EMA alpha and mask probability must be in [0,1]")
        if cfg.mask_num_dofs < 1 or not 1 <= cfg.mask_duration_min <= cfg.mask_duration_max:
            raise ValueError("Pen action mask requires positive DOF count and duration range")

    @property
    def action_dim(self) -> int:
        return int(self._raw.shape[-1])

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_action(self) -> torch.Tensor:
        return self._processed

    @property
    def target(self) -> torch.Tensor:
        return self._target

    def process_actions(self, actions: torch.Tensor) -> None:
        motion = task(self._env)
        self._raw.copy_(actions)
        residual = (actions * self.cfg.action_scale).clamp(*self.cfg.residual_clip)
        self._processed.mul_(1.0 - self.cfg.ema_alpha).add_(residual * self.cfg.ema_alpha)
        self._update_action_mask()
        bias = getattr(self.robot.data, "encoder_bias_tensor", None)
        target = motion.joint_pos + self._processed
        if bias is not None:
            target = target - bias
        if self.cfg.mask_prob > 0.0:
            frozen = (self._mask_active & self._targets_valid).unsqueeze(-1) & self._mask_dofs
            target = torch.where(frozen, self._target, target)
        self._previous_target.copy_(self._target)
        self._target.copy_(target)
        self._targets_valid.fill_(True)

    def _update_action_mask(self) -> None:
        if self.cfg.mask_prob == 0.0:
            return
        cfg = self.cfg
        term = cfg.mask_curriculum_term
        if term:
            from .curriculums import get_curriculum_value

            scale = get_curriculum_value(self._env, term, 0.0)
            duration_scale = get_curriculum_value(self._env, term, 1.0)
        else:
            scale = duration_scale = 1.0
        active_at_start = self._mask_active.clone()
        self._mask_remaining[active_at_start] -= 1
        expired = active_at_start & (self._mask_remaining <= 0)
        self._mask_active[expired] = False
        self._mask_dofs[expired] = False
        fire = (~active_at_start) & (
            torch.rand(self.num_envs, device=self._raw.device) < cfg.mask_prob * scale
        )
        ids = fire.nonzero(as_tuple=True)[0]
        if ids.numel():
            scores = torch.rand((ids.numel(), self.action_dim), device=self._raw.device)
            chosen = scores.topk(min(cfg.mask_num_dofs, self.action_dim), dim=1).indices
            self._mask_dofs[ids] = False
            self._mask_dofs[ids[:, None], chosen] = True
            duration_max = max(
                round(cfg.mask_duration_min + (cfg.mask_duration_max - cfg.mask_duration_min) * duration_scale),
                cfg.mask_duration_min,
            )
            self._mask_remaining[ids] = torch.randint(
                cfg.mask_duration_min, duration_max + 1, (ids.numel(),), device=self._raw.device
            )
            self._mask_active[ids] = True

    def apply_actions(self) -> None:
        self.robot.set_joint_position_target(self._target)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            rows = slice(None)
        else:
            rows = resolve_reset_env_ids(env_ids, num_envs=self.num_envs, device=self._raw.device, owner="PenSpinAction.reset")
        self._raw[rows] = 0.0
        self._processed[rows] = 0.0
        self._target[rows] = self.default[rows]
        self._previous_target[rows] = self.default[rows]
        self._targets_valid[rows] = False
        self._mask_active[rows] = False
        self._mask_remaining[rows] = 0
        self._mask_dofs[rows] = False


__all__ = ["PenSpinAction", "PenSpinActionCfg"]

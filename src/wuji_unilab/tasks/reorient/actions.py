# Adapted from wuji-technology/wuji-mjlab 9410a3a, mdp/actions.py.
# Copyright 2026 Wuji Technology Co., Ltd. Apache-2.0.
"""Offset position control with explicit EMA and warmup."""

from dataclasses import dataclass
from typing import cast

import torch
from unilab.base.entity import Entity
from unilab.managers import ActionTerm, ActionTermCfg

from .math import torch_soft_limits


@dataclass(kw_only=True)
class WujiActionCfg(ActionTermCfg):
    action_scale: float = 0.5
    ema_alpha: float = 0.5
    warmup_time_s: float = 0.4
    soft_limit_factor: float = 0.9

    def build(self, env):
        return WujiAction(self, env)


class WujiAction(ActionTerm):
    cfg: WujiActionCfg

    def __init__(self, cfg: WujiActionCfg, env):
        super().__init__(cfg, env)
        self.robot = cast(Entity, self._entity)
        device = torch.device(getattr(env, "device", "cpu"))
        self._device = device
        default = self.robot.data.default_joint_pos(device)
        default = default[0] if default.ndim >= 2 else default
        self.default = default.clone().repeat(self.num_envs, 1)
        # Keep limits as a per-joint canonical row; broadcast against full-width
        # action targets rather than repeating them per environment.
        limits = self.robot.data.soft_joint_pos_limits(device)
        limits = torch.as_tensor(limits, dtype=torch.float32, device=device)
        self.limits = torch_soft_limits(limits, cfg.soft_limit_factor)
        self.limits = self.limits[0] if self.limits.ndim == 3 else self.limits
        self.target = self.default.clone()
        self.raw = torch.zeros_like(self.target)
        if not 0 < cfg.ema_alpha <= 1 or not 0 < cfg.soft_limit_factor <= 1:
            raise ValueError("Wuji EMA and soft limit factors must lie in (0, 1]")

    @property
    def action_dim(self):
        return self.default.shape[-1]

    @property
    def raw_action(self):
        return self.raw

    def process_actions(self, actions: torch.Tensor):
        actions = torch.as_tensor(actions, dtype=self.raw.dtype, device=self.raw.device)
        self.raw.copy_(actions)
        raw_target = self.default + torch.clamp(actions, -1.0, 1.0) * self.cfg.action_scale
        raw_target = torch.clamp(raw_target, self.limits[None, :, 0], self.limits[None, :, 1])
        target = self.cfg.ema_alpha * raw_target + (1 - self.cfg.ema_alpha) * self.target
        warmup = torch.as_tensor(
            self._env.episode_length_buf * self._env.step_dt < self.cfg.warmup_time_s,
            dtype=torch.bool,
            device=self.target.device,
        )
        self.target.copy_(torch.where(warmup[:, None], self.default, target))

    def apply_actions(self):
        self.robot.set_joint_position_target(self.target)

    def reset(self, reset_mask: torch.Tensor):
        self.target[reset_mask] = self.default[reset_mask]
        self.raw[reset_mask] = 0.0

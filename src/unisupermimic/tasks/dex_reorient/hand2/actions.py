# Adapted from wuji-technology/wuji-mjlab 9410a3a, mdp/actions.py.
# Copyright 2026 Wuji Technology Co., Ltd. Apache-2.0.
"""Offset position control with explicit EMA and warmup."""

import math
from dataclasses import dataclass
from typing import cast

import torch
from unilab.base.entity import Entity
from unilab.base.reset_selectors import resolve_reset_env_ids
from unilab.managers import ActionTerm, ActionTermCfg

from .math import soft_limits


@dataclass(kw_only=True)
class WujiActionCfg(ActionTermCfg):
    action_scale: float = 0.5
    ema_alpha: float = 0.5
    warmup_time_s: float = 0.4
    # Dex-SuperMimic's JointPositionOffsetEMAAction clamps against the
    # entity's soft limits directly.  Keep this option for old UniLab callers,
    # but default to the source behavior (factor 1.0).
    soft_limit_factor: float = 1.0

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
        self.warmup_steps = max(0, math.ceil(cfg.warmup_time_s / float(self._env.step_dt) - 1.0e-9))
        # Keep limits as a per-joint canonical row; broadcast against full-width
        # action targets rather than repeating them per environment.
        limits = self.robot.data.soft_joint_pos_limits(device)
        limits = torch.as_tensor(limits, dtype=torch.float32, device=device)
        if limits.ndim >= 3:
            limits = limits[0]
        self.limits = soft_limits(limits, cfg.soft_limit_factor)
        self.limits = self.limits[0] if self.limits.ndim == 3 else self.limits
        self.target = self.default.clone()
        # c2 in the source reward is the target from the preceding control
        # step.  Keep it explicitly; falling back to c1 makes the high
        # frequency penalty a first difference and changes the training
        # dynamics at 50 Hz.
        self.previous_target = self.default.clone()
        self.previous_previous_target = self.default.clone()
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
        # The source controller receives a bounded tanh action.  A non-finite
        # value here means the learner or backend became numerically invalid;
        # fail at this boundary with a useful error instead of silently
        # replacing the action and continuing a corrupted run.
        if not bool(torch.isfinite(actions).all()):
            raise FloatingPointError("Hand2 joint_pos action contains NaN or Inf")
        actions = actions.clamp(-1.0, 1.0)
        self.raw.copy_(actions)
        raw_target = self.default + torch.clamp(actions, -1.0, 1.0) * self.cfg.action_scale
        raw_target = torch.clamp(raw_target, self.limits[None, :, 0], self.limits[None, :, 1])
        previous_target = self.target.clone()
        previous_previous_target = self.previous_target.clone()
        target = self.cfg.ema_alpha * raw_target + (1 - self.cfg.ema_alpha) * self.target
        warmup = self._env.episode_length_buf < self.warmup_steps
        if not bool(torch.isfinite(target).all()):
            raise FloatingPointError("Hand2 joint_pos target contains NaN or Inf")
        target = torch.maximum(torch.minimum(target, self.limits[None, :, 1]), self.limits[None, :, 0])
        self.previous_target.copy_(previous_target)
        self.previous_previous_target.copy_(previous_previous_target)
        self.target.copy_(torch.where(warmup[:, None], self.default, target))

    def apply_actions(self):
        # Keep EMA/history in controller coordinates c. The simulator executes
        # u = c - encoder_bias, which is also the target recorded in replay
        # and consumed by the imagined controller/world model.
        bias = getattr(self.robot.data, "encoder_bias_tensor", None)
        target = self.target
        if bias is not None:
            target = target - torch.as_tensor(bias, device=target.device, dtype=target.dtype)
        self.robot.set_joint_position_target(target)

    def reset(self, env_ids: torch.Tensor | None = None):
        ids = resolve_reset_env_ids(
            env_ids,
            num_envs=self.num_envs,
            device=self._device,
            owner="WujiAction.reset",
        )
        self.target[ids] = self.default[ids]
        self.previous_target[ids] = self.default[ids]
        self.previous_previous_target[ids] = self.default[ids]
        self.raw[ids] = 0.0

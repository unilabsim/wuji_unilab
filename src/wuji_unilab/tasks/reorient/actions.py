# Adapted from wuji-technology/wuji-mjlab 9410a3a, mdp/actions.py.
# Copyright 2026 Wuji Technology Co., Ltd. Apache-2.0.
"""Offset position control with explicit EMA and warmup."""

from dataclasses import dataclass
from typing import cast

import numpy as np
from unilab.base.entity import Entity
from unilab.managers import ActionTerm, ActionTermCfg

from .math import soft_limits


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
        self.default = self.robot.data.default_joint_pos.copy()
        self.limits = soft_limits(self.robot.data.soft_joint_pos_limits, cfg.soft_limit_factor)
        self.target = self.default.copy()
        self.raw = np.zeros_like(self.target)
        if not 0 < cfg.ema_alpha <= 1 or not 0 < cfg.soft_limit_factor <= 1:
            raise ValueError("Wuji EMA and soft limit factors must lie in (0, 1]")

    @property
    def action_dim(self):
        return self.default.shape[1]

    @property
    def raw_action(self):
        return self.raw

    def process_actions(self, actions):
        np.copyto(self.raw, actions)
        raw_target = self.default + np.clip(actions, -1, 1) * self.cfg.action_scale
        raw_target = np.clip(raw_target, self.limits[:, 0], self.limits[:, 1])
        target = self.cfg.ema_alpha * raw_target + (1 - self.cfg.ema_alpha) * self.target
        warmup = self._env.episode_length_buf * self._env.step_dt < self.cfg.warmup_time_s
        self.target[:] = np.where(warmup[:, None], self.default, target)

    def apply_actions(self):
        self.robot.set_joint_position_target(self.target)

    def reset(self, env_ids):
        self.target[env_ids] = self.default[env_ids]
        self.raw[env_ids] = 0

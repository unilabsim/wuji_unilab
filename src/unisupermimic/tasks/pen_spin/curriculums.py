# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Shared curriculum helpers used across multiple tasks."""

from __future__ import annotations

import torch


def get_curriculum_value(env, term_name: str, default: float = 1.0) -> float:
  """Read scalar curriculum state from manager, with a safe default."""
  manager = getattr(env, "curriculum_manager", None)
  if manager is None:
    return default

  state = getattr(manager, "_curriculum_state", {})
  if term_name not in state or state[term_name] is None:
    return default

  value = state[term_name]
  if isinstance(value, torch.Tensor):
    return value.mean().item() if value.numel() > 0 else default
  if isinstance(value, dict):
    if not value:
      return default
    first = next(iter(value.values()))
    if isinstance(first, torch.Tensor):
      return first.mean().item() if first.numel() > 0 else default
    return float(first)
  return float(value)





def linear_step_curriculum(
  env,
  env_ids,
  total_env_steps: int = 400_000,
  warmup_env_steps: int = 40_000,
) -> dict:
  """Disturbance progress: flat 0 during a warmup, then a linear ramp to 1.

  Replaces the success-rate rule for RSI/adaptive-sampling runs: with random
  start frames the ``time_out``-based success definition is biased by where an
  episode starts, so gating the disturbance on it couples the curriculum to the
  sampler. Here progress ignores performance entirely::

    progress = clamp((step - warmup) / (total - warmup), 0, 1)

    ``common_step_counter`` advances once per control step per rank. With the
    Wuji Hand 2 production setting (40 steps/iteration and the configured
    ``total_env_steps=240_000``), this holds the disturbance at zero for the
    first 1_000 iterations and reaches full strength at iteration 6_000. The
    dict shape (``value`` first) matches what ``get_curriculum_value`` reads for
    the object wrench, fingertip bumps, action mask, and reset-state noise.
  """
  if warmup_env_steps >= total_env_steps:
    raise ValueError(
      f"warmup_env_steps ({warmup_env_steps}) must be smaller than total_env_steps ({total_env_steps})"
    )
  step = env.common_step_counter
  span = float(total_env_steps - warmup_env_steps)
  progress = min(max((step - warmup_env_steps) / span, 0.0), 1.0)
  return {"value": progress, "progress": progress}

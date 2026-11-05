"""MLP and running normalization adapted from the existing SuperMimic implementation.
Original adaptation: Chong Zhang / ETH Zurich (MIT); see SuperMimic README.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def require_finite(name: str, value: torch.Tensor) -> None:
    """Stop at the first invalid boundary instead of altering training data."""
    if not torch.isfinite(value).all():
        invalid = (~torch.isfinite(value)).nonzero(as_tuple=False)
        first = invalid[0].detach().cpu().tolist()
        finite = value.detach()[torch.isfinite(value)]
        maximum = float(finite.abs().max()) if finite.numel() else float("nan")
        raise FloatingPointError(
            f"{name} contains NaN/Inf (shape={tuple(value.shape)}, "
            f"first_invalid_index={first}, finite_absmax={maximum:.6g})"
        )


def mlp(inp: int, hidden: list[int], out: int, act=nn.ELU, layer_norm: bool = False) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Running mean/std normalizer; updated from data without gradients, frozen after ``max_count``."""

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))
        self.max_count, self.eps = max_count, eps
        self._host_count = 0.0  # host-side mirror of ``count`` (avoids a device sync per update)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._host_count = float(self.count)

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).float()
        if x.shape[0] == 0 or self._host_count >= self.max_count:
            return
        require_finite("RunningNorm.update input", x)
        x = x[: max(0, int(self.max_count - self._host_count))]
        if not x.shape[0]:
            return
        n = x.shape[0]
        self._host_count += n
        m, v = x.mean(0), x.var(0, unbiased=False)
        tot = self.count + n
        delta = m - self.mean
        self.mean += delta * n / tot
        self.var = (self.var * self.count + v * n + delta**2 * self.count * n / tot) / tot
        self.count = tot

    def forward(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    def std(self):
        return torch.sqrt(self.var + self.eps**2)


class EmpiricalNorm(nn.Module):
  """Exact rsl-rl empirical normalization used by the upstream Wuji PPO recipe."""

  def __init__(self, dim: int, eps: float = 1e-2):
    super().__init__()
    self.eps = eps
    self.register_buffer("_mean", torch.zeros(dim).unsqueeze(0))
    self.register_buffer("_var", torch.ones(dim).unsqueeze(0))
    self.register_buffer("_std", torch.ones(dim).unsqueeze(0))
    self.register_buffer("count", torch.tensor(0, dtype=torch.long))

  @torch.jit.unused
  @torch.no_grad()
  def update(self, x):
    require_finite("EmpiricalNorm.update input", x)
    count_x = x.shape[0]
    if count_x == 0:
      return
    self.count += count_x
    rate = count_x / self.count
    var_x = x.var(0, unbiased=False, keepdim=True)
    mean_x = x.mean(0, keepdim=True)
    delta_mean = mean_x - self._mean
    self._mean += rate * delta_mean
    self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
    self._std = torch.sqrt(self._var)

  def forward(self, x):
    return (x - self._mean) / (self._std + self.eps)

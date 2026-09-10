# Derived from Wuji's rsl-rl 5.0.1+wuji1 modules/distribution.py.
# Copyright 2021-2026 ETH Zurich, NVIDIA CORPORATION; modifications Wuji Technology.
# BSD-3-Clause. Adapted to the public rsl-rl distribution API.
"""Positive state-dependent exploration scale with an explicit lower bound."""

import math

import torch
from rsl_rl.modules.distribution import HeteroscedasticGaussianDistribution


class SoftplusGaussianDistribution(HeteroscedasticGaussianDistribution):
    def __init__(self, output_dim: int, init_std: float = 0.5, min_std: float = 0.2):
        if not 0 <= min_std < init_std or not math.isfinite(init_std):
            raise ValueError("Require finite init_std > min_std >= 0")
        super().__init__(output_dim, math.log(math.expm1(init_std - min_std)), std_type="scalar")
        self.min_std = min_std

    def update(self, output: torch.Tensor):
        mean, raw = output.unbind(-2)
        self._distribution = torch.distributions.Normal(
            mean, torch.nn.functional.softplus(raw) + self.min_std
        )

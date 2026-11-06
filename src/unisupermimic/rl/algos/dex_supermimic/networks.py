"""Compact dynamics and an independent real-termination/failure head."""

from contextlib import nullcontext

import torch
from torch import nn
from torch.distributions import Normal

from .common import EmpiricalNorm, RunningNorm, mlp
from .state import DELTA_DIM, STATE_DIM, integrate


class WorldModel(nn.Module):
    def __init__(
        self, dynamics_dim, contact_dim, hidden=(2048, 2048, 2048), amp=True
    ):
        super().__init__()
        self.contact_dim = contact_dim
        self.amp = amp
        # Dynamics context contains physical state, startup simulator parameters
        # and the measured/sampled transition disturbance. Task command state is
        # deliberately absent: it is recursively derived after each WM step.
        inp = dynamics_dim + 20
        self.norm = RunningNorm(inp)
        self.net = mlp(
            inp,
            list(hidden),
            DELTA_DIM + contact_dim,
            act=nn.SiLU,
            layer_norm=True,
        )
        self.delta_norm = RunningNorm(DELTA_DIM, eps=1e-4)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[DELTA_DIM:] = -3.0
        self.register_buffer(
            "error_floor",
            torch.tensor(
                [0.003] * 20
                + [0.05] * 20
                + [0.001] * 3
                + [0.005] * 3
                + [0.02] * 3
                + [0.05] * 3
            ),
        )

    def inputs(self, dynamics, u):
        return torch.cat((dynamics, u), -1)

    def forward(self, x, u, dynamics):
        features = self.norm(self.inputs(dynamics, u))
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if self.amp and x.is_cuda else nullcontext()
        with autocast:
            y = self.net(features)
        y = y.float()
        d1, c = y.split((DELTA_DIM, self.contact_dim), -1)
        scale = self.delta_norm.std()
        return (
            integrate(x, d1 * scale + self.delta_norm.mean),
            c,
        )


class FailureHead(nn.Module):
    """Dreamer-style continue head: P(real termination entering this imagined state).

    It consumes the same privileged task/critic features built from either true
    simulator state or the world model's predicted ``x1``; no simulator state is
    queried inside imagination.
    """

    def __init__(self, context_dim):
        super().__init__()
        dim = STATE_DIM + context_dim
        self.norm = RunningNorm(dim)
        self.net = mlp(dim, [256, 128], 1)

    def features(self, x1, context):
        return torch.cat((x1, context), -1)

    def forward(self, features):
        return self.net(self.norm(features)).squeeze(-1)


class Actor(nn.Module):
    def __init__(self, obs_dim=207):
        super().__init__()
        self.norm = EmpiricalNorm(obs_dim)
        self.net = mlp(obs_dim, [512, 256, 128], 20)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs):
        return torch.tanh(self.net(self.norm(obs)))


class GaussianPPOActor(nn.Module):
    """Wuji-MJLab-style state-dependent diagonal Gaussian used by the PPO baseline.

    The final linear layer emits ``[mean, std]``. The std slice starts at
    ``init_std`` with zero weights and is clamped to the same scalar range as
    upstream's ``HeteroscedasticGaussianDistribution``. No tanh is applied: the
    task action manager itself maps/clips normalized actions to joint targets.
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden=(512, 256, 128),
        init_std=0.5,
        std_range=(0.2, 50.0),
    ):
        super().__init__()
        self.action_dim = action_dim
        self.init_std = init_std
        self.std_range = std_range
        self.norm = EmpiricalNorm(obs_dim)
        # Exactly mirror rsl-rl's default nn.Linear initialization; only the
        # std-head rows/bias are overwritten below.
        self.net = mlp(obs_dim, list(hidden), 2 * action_dim)
        self._init_std_head()

    def _init_std_head(self):
        final = self.net[-1]
        with torch.no_grad():
            final.weight[self.action_dim :].zero_()
            final.bias[self.action_dim :] = self.init_std

    def distribution(self, obs):
        mean, std = self.net(self.norm(obs)).unflatten(-1, (2, self.action_dim)).unbind(-2)
        return Normal(mean, std.clamp(*self.std_range))

    def forward(self, obs):
        return self.distribution(obs).sample()

    @torch.no_grad()
    def mean(self, obs):
        return self.distribution(obs).mean


class Critic(nn.Module):
    def __init__(self, dim, amp=True, empirical_norm=False):
        super().__init__()
        self.amp = amp
        self.norm = EmpiricalNorm(dim) if empirical_norm else RunningNorm(dim)
        self.net = mlp(dim, [512, 512, 256, 128], 1)

    def forward(self, x):
        if isinstance(self.norm, EmpiricalNorm):
            return self.net(self.norm(x)).squeeze(-1)
        features = self.norm(x)
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if self.amp and x.is_cuda else nullcontext()
        with autocast:
            return self.net(features).float().squeeze(-1)

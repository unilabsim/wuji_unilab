"""GPU time/environment replay with correct episode and numerical-validity masks."""

import torch

from .common import require_finite


class ReplayBuffer:
    def __init__(self, capacity, num_envs, device, holdout_fraction=0.1):
        self.capacity = capacity
        self.num_envs = num_envs
        self.device = device
        self.train_end = max(1, num_envs - max(1, int(num_envs * holdout_fraction))) if holdout_fraction else num_envs
        self.data = {}
        self.ptr = 0
        self.size = 0
        self._failure_pools = {}

    def add(self, transition):
        self._failure_pools.clear()
        if not self.data:
            self.data = {
                k: torch.empty((self.capacity, *v.shape), device=self.device, dtype=v.dtype)
                for k, v in transition.items()
            }
        for k, v in transition.items():
            if v.is_floating_point():
                require_finite(f"ReplayBuffer transition.{k}", v)
            self.data[k][self.ptr].copy_(v)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def sample(self, batch, window=1, holdout=False, generator=None,
               failure_frac=0.0, failure_horizon=16):
        if self.size < window:
            raise ValueError("Not enough replay steps")
        lo, hi = (self.train_end, self.num_envs) if holdout else (0, self.train_end)
        if lo == hi:
            raise ValueError("No held-out environments")
        count = self.size - window + 1
        nenv = hi - lo
        total = count * nenv
        ids = torch.randint(total, (batch,), device=self.device, generator=generator)
        importance = torch.ones(batch, device=self.device)
        if failure_frac:
            key = (window, holdout, failure_horizon)
            if key not in self._failure_pools:
                chronological = (self.ptr - self.size + torch.arange(self.size, device=self.device)) % self.capacity
                term = self.data["terminated"][chronological, lo:hi].int()
                cumulative = torch.nn.functional.pad(term.cumsum(0), (0, 0, 1, 0))
                starts = torch.arange(count, device=self.device)
                ends = (starts + failure_horizon).clamp(max=self.size)
                failure = ((cumulative[ends] - cumulative[starts]) > 0).flatten()
                self._failure_pools[key] = (failure, failure.nonzero(as_tuple=True)[0])
            failure, pool = self._failure_pools[key]
            n_failure = pool.numel()
            n_draw = int(batch * failure_frac)
            if n_draw and n_failure:
                ids[:n_draw] = pool[torch.randint(n_failure, (n_draw,), device=self.device, generator=generator)]
            ids = ids[torch.randperm(batch, device=self.device, generator=generator)]
            q = n_draw / batch if n_failure else 0.0
            selected_probability = (1 - q) / total + q * failure[ids].float() / max(n_failure, 1)
            importance = 1 / (total * selected_probability)
        t, env = ids // nenv, ids % nenv + lo
        rows = (self.ptr - self.size + t[:, None] + torch.arange(window, device=self.device)) % self.capacity
        out = {k: v[rows, env[:, None]] for k, v in self.data.items()}
        # Keep a valid terminal transition, never cross into the next episode.
        alive = torch.cat((torch.ones_like(out["done"][:, :1]), ~out["done"][:, :-1]), 1).float().cumprod(1)
        # An invalid state also poisons all later open-loop predictions.
        out["mask"] = alive * out["valid"].float().cumprod(1)
        out["importance"] = importance[:, None].expand(-1, window)
        out["env"] = env
        return out

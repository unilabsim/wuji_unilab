"""Device resident replay for the Hand 2 SuperMimic contract.

The generic SuperMimic replay buffer stores the G1 transition layout.  Hand 2
has a different state, controller-memory and contact layout, so keeping a
small schema-specific buffer avoids silently dropping fields while migrating
the source learner.  Rows are stored in chronological ``(time, environment)``
order; windows are masked at episode boundaries instead of training across a
reset.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

HAND2_REPLAY_FIELDS: dict[str, tuple[int | tuple[()], torch.dtype]] = {
    "s": (58, torch.float32),
    "s1": (58, torch.float32),
    # Runtime interval events are applied after the reward snapshot in the
    # UniLab lifecycle, just like the source ``push = next_x - physics``
    # boundary.  Keep the state jump separate from the world-model target.
    "push": (58, torch.float32),
    "a": (20, torch.float32),
    "a_prev": (20, torch.float32),
    "xi": (46, torch.float32),
    "m": (100, torch.float32),
    "m1": (100, torch.float32),
    # ``c1`` in ``m`` is the pre-action controller target.  The source
    # learner stores the post-action executed target separately; keep both it
    # and the per-environment encoder bias so imagination can reconstruct the
    # actuator input ``u = c - bias`` exactly.
    "u": (20, torch.float32),
    "bias": (20, torch.float32),
    "aux": (21, torch.float32),
    "contact": (32, torch.float32),
    "torque": (20, torch.float32),
    "obs": (207, torch.float32),
    "done": ((), torch.bool),
    "terminated": ((), torch.bool),
    "timeout": ((), torch.bool),
    "ep_step": ((), torch.long),
    "valid": ((), torch.bool),
}


def _field_shape(shape: int | tuple[()]) -> tuple[int, ...]:
    return (shape,) if isinstance(shape, int) else shape


class Hand2ReplayBuffer:
    """Fixed-size, device-local replay arranged as ``(time, environment)``.

    ``add`` accepts one batched transition per simulation step.  The buffer
    keeps the environment axis intact, which lets ``sample_windows`` select a
    true temporal window and lets ``sample_starts`` reconstruct actor history
    without crossing a reset boundary.
    """

    def __init__(
        self,
        capacity: int,
        num_envs: int,
        device: str | torch.device,
        *,
        fields: Mapping[str, tuple[int | tuple[()], torch.dtype]] | None = None,
        holdout_fraction: float = 0.0,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if num_envs < 1:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        self.capacity = int(capacity)
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.fields = dict(HAND2_REPLAY_FIELDS if fields is None else fields)
        if not 0.0 <= holdout_fraction < 1.0:
            raise ValueError("holdout_fraction must be in [0, 1)")
        self.train_end = (
            max(1, self.num_envs - max(1, int(self.num_envs * holdout_fraction)))
            if holdout_fraction
            else self.num_envs
        )
        self.data: dict[str, torch.Tensor] = {
            name: torch.zeros(
                (self.capacity, self.num_envs) + _field_shape(shape),
                dtype=dtype,
                device=self.device,
            )
            for name, (shape, dtype) in self.fields.items()
        }
        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size * self.num_envs

    @property
    def time_size(self) -> int:
        return self.size

    def _coerce(self, name: str, value: torch.Tensor) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Hand2 replay field {name!r} must be a tensor")
        expected = self.data[name].shape[2:]
        if value.shape == expected:
            value = value.unsqueeze(0)
        if value.shape != (self.num_envs, *expected):
            raise ValueError(
                f"Hand2 replay field {name!r} expected "
                f"{(self.num_envs, *expected)}, got {tuple(value.shape)}"
            )
        return value.to(device=self.device, dtype=self.data[name].dtype)

    @torch.no_grad()
    def add(self, transition: Mapping[str, torch.Tensor]) -> None:
        """Append one simulation step for all environments."""
        # ``push`` was added after the initial migration.  A zero fallback is
        # valid for lightweight backends that have no interval event support;
        # the real Hand2 environment always supplies the measured jump.
        missing = set(self.fields).difference(transition) - {"valid", "push"}
        if missing:
            raise KeyError(f"Hand2 replay transition is missing fields: {sorted(missing)}")
        finite = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for name in self.fields:
            if name == "valid" and name not in transition:
                continue
            if name == "push" and name not in transition:
                value = torch.zeros(
                    (self.num_envs, *self.data[name].shape[2:]),
                    dtype=self.data[name].dtype,
                    device=self.device,
                )
            else:
                value = self._coerce(name, transition[name])
            if value.is_floating_point():
                finite &= torch.isfinite(value).reshape(self.num_envs, -1).all(-1)
                if not bool(finite.all()):
                    raise FloatingPointError(
                        f"Hand2 replay field {name!r} contains NaN or Inf"
                    )
            self.data[name][self.ptr].copy_(value)
        if "valid" not in transition:
            self.data["valid"][self.ptr].copy_(finite)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        """Map oldest-first age indices to ring-buffer rows."""
        return (self.ptr - self.size + age) % self.capacity

    def _require_window(self, batch: int, window: int) -> None:
        if batch < 1:
            raise ValueError(f"batch must be positive, got {batch}")
        if not 1 <= window <= self.size:
            raise ValueError(f"window length must be in [1, {self.size}], got {window}")

    @torch.no_grad()
    def sample_windows(
        self,
        batch: int,
        window: int,
        *,
        holdout: bool = False,
        failure_frac: float = 0.0,
        failure_horizon: int = 16,
    ) -> dict[str, torch.Tensor]:
        """Sample windows and mask transitions after each environment reset.

        The mask at step ``j`` is one only when the start and all preceding
        transitions are alive.  A terminal transition itself remains in the
        loss; the first transition after it is masked out.
        """
        self._require_window(batch, window)
        if not 0.0 <= failure_frac < 1.0:
            raise ValueError("failure_frac must be in [0, 1)")
        if failure_horizon < 1:
            raise ValueError("failure_horizon must be positive")
        lo, hi = (self.train_end, self.num_envs) if holdout else (0, self.train_end)
        if lo >= hi:
            raise ValueError("no environments available for this replay split")
        count = self.size - window + 1
        nenv = hi - lo
        total = count * nenv
        ids = torch.randint(total, (batch,), device=self.device)
        # Failure-biased sampling changes the start distribution.  Keep the
        # inverse sampling-probability correction alongside each sampled
        # window so the failure head still fits the replay distribution used
        # by the source learner.
        importance = torch.ones(batch, device=self.device)
        if failure_frac:
            chronology = self._row(torch.arange(self.size, device=self.device))
            term = self.data["terminated"][chronology, lo:hi].int()
            cumulative = torch.nn.functional.pad(term.cumsum(0), (0, 0, 1, 0))
            starts_all = torch.arange(count, device=self.device)
            ends = (starts_all + failure_horizon).clamp(max=self.size)
            failure_by_start = (cumulative[ends] - cumulative[starts_all] > 0)
            failure = failure_by_start.flatten()
            pool = failure.nonzero(as_tuple=True)[0]
            draws = min(int(batch * failure_frac), batch)
            if draws and pool.numel():
                ids[:draws] = pool[torch.randint(pool.numel(), (draws,), device=self.device)]
                ids = ids[torch.randperm(batch, device=self.device)]
                q = draws / batch
                probability = (1.0 - q) / total + q * failure[ids].float() / max(pool.numel(), 1)
                importance = 1.0 / (total * probability)
        starts = ids // nenv
        env = ids % nenv + lo
        age = starts[:, None] + torch.arange(window, device=self.device)
        rows = self._row(age)
        envs = env[:, None].expand_as(rows)
        out = {
            name: value[rows, envs]
            for name, value in self.data.items()
        }
        done = out["done"]
        alive_before = torch.cat(
            (torch.ones((batch, 1), dtype=torch.bool, device=self.device), ~done[:, :-1]),
            dim=-1,
        )
        out["mask"] = torch.cumprod(alive_before, dim=-1).to(torch.float32)
        out["mask"] = out["mask"] * out["valid"].float().cumprod(-1)
        out["importance"] = importance[:, None].expand(-1, window)
        out["env"] = env
        out["start_age"] = starts
        return out

    @torch.no_grad()
    def sample_starts(
        self,
        batch: int,
        history: int,
        *,
        holdout: bool = False,
        all_envs: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Sample starts and history padded at episode/reset boundaries."""
        if batch < 1:
            raise ValueError(f"batch must be positive, got {batch}")
        if self.size < 1:
            raise ValueError("cannot sample from an empty Hand2 replay buffer")
        if history < 1:
            raise ValueError(f"history length must be positive, got {history}")
        if all_envs:
            lo, hi = 0, self.num_envs
        else:
            lo, hi = (self.train_end, self.num_envs) if holdout else (0, self.train_end)
        if lo >= hi:
            raise ValueError("no environments available for this replay split")
        # Imagination must start from a finite transition.  ``add`` marks
        # rows invalid when a simulator bridge returns a non-finite field;
        # sampling those rows would feed the sanitized placeholder values to
        # the actor and can still poison its running statistics.  Draw from
        # the valid pool directly, with a clear error if the replay contains
        # no usable starts yet.
        chronological = self._row(torch.arange(self.size, device=self.device))
        valid = self.data["valid"][chronological, lo:hi]
        pool = valid.nonzero(as_tuple=False)
        if pool.numel() == 0:
            raise RuntimeError("Replay contains no finite Hand2 imagination starts")
        selected = pool[torch.randint(pool.shape[0], (batch,), device=self.device)]
        age = selected[:, 0]
        env = selected[:, 1] + lo
        row = self._row(age)
        ep_step = self.data["ep_step"][row, env].long()
        offsets = torch.arange(history - 1, -1, -1, device=self.device)
        # ``ep_step`` is the pre-action episode age.  It prevents history from
        # reaching before a reset; ``age`` prevents reaching before the ring's
        # oldest retained row.
        back = torch.minimum(offsets[None, :], ep_step[:, None].clamp_min(0))
        back = torch.minimum(back, age[:, None])
        history_rows = self._row(age[:, None] - back)
        history_envs = env[:, None].expand_as(history_rows)
        start = {name: value[row, env] for name, value in self.data.items()}
        past = {
            name: value[history_rows, history_envs] for name, value in self.data.items()
        }
        start["env"] = env
        start["age"] = age
        past["env"] = env
        past["age"] = age[:, None] - back
        return start, past

    def state_dict(self) -> dict[str, object]:
        return {
            "capacity": self.capacity,
            "num_envs": self.num_envs,
            "ptr": self.ptr,
            "size": self.size,
            "data": {name: value.clone() for name, value in self.data.items()},
        }

    @torch.no_grad()
    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if int(state["capacity"]) != self.capacity or int(state["num_envs"]) != self.num_envs:
            raise ValueError("Hand2 replay capacity/num_envs mismatch")
        raw = state["data"]
        if not isinstance(raw, Mapping):
            raise ValueError("Hand2 replay field layout mismatch")
        required = set(self.data) - {"valid"}
        if set(raw) not in (set(self.data), required):
            raise ValueError("Hand2 replay field layout mismatch")
        for name, value in self.data.items():
            if name == "valid" and name not in raw:
                value.fill_(True)
                continue
            if not isinstance(raw[name], torch.Tensor) or raw[name].shape != value.shape:
                raise ValueError(f"Hand2 replay field shape mismatch for {name!r}")
            value.copy_(raw[name].to(device=self.device, dtype=value.dtype))
        self.ptr = int(state["ptr"]) % self.capacity
        self.size = min(int(state["size"]), self.capacity)


__all__ = ["HAND2_REPLAY_FIELDS", "Hand2ReplayBuffer"]

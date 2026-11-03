"""Checkpoint and lightweight scalar logging helpers for the Hand 2 runner.

The Hand 2 learner uses regular :class:`torch.nn.Module` objects, but its
environment is a host bridge and its replay payload is kept outside a PyTorch
module.  Keeping serialization here makes saving independent of the backend
and lets the runner decide whether a checkpoint should include replay data.
"""

from __future__ import annotations

import csv
import os
import random
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

CHECKPOINT_FORMAT_VERSION = 1


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any] | None) -> None:
    """Restore process RNGs saved by :func:`save_hand2_checkpoint`."""
    if not state:
        return

    def cpu_byte_tensor(value: Any) -> torch.Tensor:
        """Normalize a serialized RNG state for PyTorch's CPU API.

        Checkpoints are loaded with ``map_location=self.device``.  On CUDA
        that moves every serialized tensor, including the *CPU* generator
        state, onto ``cuda:0``; ``torch.set_rng_state`` rejects that tensor
        even though its bytes are otherwise valid.  Older checkpoints can
        also contain a NumPy array, so normalize both representation and
        dtype here without changing any random state bytes.
        """
        if isinstance(value, torch.Tensor):
            return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()
        return torch.as_tensor(value, device="cpu", dtype=torch.uint8).contiguous()

    if state.get("torch") is not None:
        torch.set_rng_state(cpu_byte_tensor(state["torch"]))
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("python") is not None:
        random.setstate(state["python"])
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        # Keep CUDA states as CPU byte tensors; the CUDA API copies them to
        # the corresponding generators and this also works when a checkpoint
        # was loaded with a CPU map location.
        torch.cuda.set_rng_state_all([cpu_byte_tensor(item) for item in cuda_state])


def _module_state(modules: Mapping[str, nn.Module]) -> dict[str, dict[str, Any]]:
    return {name: module.state_dict() for name, module in modules.items()}


def _optimizer_state(optimizers: Mapping[str, torch.optim.Optimizer]) -> dict[str, dict[str, Any]]:
    return {name: optimizer.state_dict() for name, optimizer in optimizers.items()}


def save_hand2_checkpoint(
    path: str | Path,
    *,
    cfg: Any,
    modules: Mapping[str, nn.Module],
    optimizers: Mapping[str, torch.optim.Optimizer] | None = None,
    iteration: int = 0,
    sim_steps: int = 0,
    replay: Mapping[str, list[torch.Tensor]] | Any | None = None,
    obs: torch.Tensor | None = None,
    obs_history: torch.Tensor | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save a resumable Hand 2 training checkpoint.

    ``replay`` is optional because it can be large.  When provided, tensors
    are detached and moved to CPU so a checkpoint does not retain a CUDA
    allocator reservation.  Module state includes all ``RunningNorm``
    buffers, so model statistics resume together with the weights.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(cfg, "__dataclass_fields__"):
        from dataclasses import asdict

        config = asdict(cfg)
    elif isinstance(cfg, Mapping):
        config = dict(cfg)
    else:
        config = repr(cfg)
    state: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "created_at": time.time(),
        "config": config,
        "iteration": int(iteration),
        "sim_steps": int(sim_steps),
        "modules": _module_state(modules),
        "optimizers": _optimizer_state(optimizers or {}),
        "rng": _rng_state(),
    }
    if replay is not None:
        raw_replay = replay.state_dict() if hasattr(replay, "state_dict") else replay
        if not isinstance(raw_replay, Mapping):
            raise TypeError("replay must be a mapping or expose state_dict()")

        def cpu_copy(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.detach().cpu()
            if isinstance(value, Mapping):
                return {key: cpu_copy(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(cpu_copy(item) for item in value)
            return value

        state["replay"] = cpu_copy(raw_replay)
    if obs is not None:
        state["obs"] = obs.detach().cpu()
    if obs_history is not None:
        state["obs_history"] = obs_history.detach().cpu()
    if extra:
        state["extra"] = dict(extra)

    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(state, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_hand2_checkpoint(
    path: str | Path,
    *,
    modules: Mapping[str, nn.Module],
    optimizers: Mapping[str, torch.optim.Optimizer] | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load model/optimizer state and return runner state for resumption."""
    state = torch.load(path, map_location=map_location, weights_only=False)
    version = state.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported Hand2 checkpoint format {version!r}")
    saved_modules = state.get("modules", {})
    for name, module in modules.items():
        if name not in saved_modules:
            raise KeyError(f"Checkpoint is missing module {name!r}")
        module.load_state_dict(saved_modules[name])
    saved_optimizers = state.get("optimizers", {})
    if optimizers is not None:
        for name, optimizer in optimizers.items():
            if name in saved_optimizers:
                optimizer.load_state_dict(saved_optimizers[name])
    if restore_rng:
        restore_rng_state(state.get("rng"))
    return state


def _scalar(value: Any) -> float | int | str:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return str(value.detach().cpu().tolist())
        return float(value.detach().cpu())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (float, int, str)):
        return value
    return str(value)


class Hand2CSVLogger:
    """Append scalar dictionaries to a header-stable ``train_log.csv``."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fields: list[str] = []
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open(newline="", encoding="utf-8") as handle:
                self._fields = list(csv.DictReader(handle).fieldnames or [])

    def log(self, values: Mapping[str, Any]) -> None:
        row = {str(key): _scalar(value) for key, value in values.items()}
        fields = list(self._fields)
        fields.extend(key for key in row if key not in fields)
        if fields != self._fields and self.path.exists() and self.path.stat().st_size:
            with self.path.open(newline="", encoding="utf-8") as handle:
                old_rows = list(csv.DictReader(handle))
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(old_rows)
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if not self._fields:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in fields})
        self._fields = fields

    def close(self) -> None:
        return None


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "Hand2CSVLogger",
    "load_hand2_checkpoint",
    "restore_rng_state",
    "save_hand2_checkpoint",
]

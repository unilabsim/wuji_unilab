"""Wuji VecEnv adapter and curriculum checkpoint runner on direct rsl-rl."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict
from unilab.rl import RslRlVecEnvAdapter

from wuji_unilab.tasks.reorient.commands import task

# Checkpoints written by the uni_rl-based runtime carry the curriculum envelope
# under this key; keep it so those checkpoints still resume.
_STATE_KEY = "uni_rl_training_state"
_STATE_VERSION = 1


def _to_torch(x: Any, device: str | torch.device) -> torch.Tensor:
    """Convert numpy-like input to torch on the target device."""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, np.ndarray):
        tensor = torch.from_numpy(x).to(device)
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            tensor = tensor.float()
        return tensor
    arr = np.asarray(x, dtype=np.float32)
    return torch.from_numpy(arr).to(device)


def _copy_training_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Validate plain data and detach the snapshot from mutable provider state."""
    ancestors: set[int] = set()

    def copy_value(value: Any) -> Any:
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError("Training state numbers must be finite")
            return value
        if type(value) not in (dict, list):
            raise TypeError(
                "Training state must contain only plain JSON dictionaries/lists/scalars"
            )
        identity = id(value)
        if identity in ancestors:
            raise ValueError("Training state must not contain reference cycles")
        ancestors.add(identity)
        try:
            if isinstance(value, dict):
                if any(type(key) is not str for key in value):
                    raise TypeError("Training state dictionary keys must be strings")
                return {key: copy_value(item) for key, item in value.items()}
            return [copy_value(item) for item in value]
        finally:
            ancestors.remove(identity)

    if not isinstance(state, Mapping):
        raise TypeError("Training state export must return a mapping")
    return dict(copy_value(dict(state)))


class WujiWrapper(RslRlVecEnvAdapter):
    def _obs_to_tensordict(self, obs: dict[str, Any]) -> TensorDict:
        """Adapt UniLab's flat groups without the legacy policy alias.

        UniLab exposes a flat policy group as ``obs["obs"]``.  The upstream
        adapter also aliases it as the RSL-RL legacy ``policy`` key.  The task's
        RSL-RL actor group is ``actor``; keeping the alias would allocate and
        mini-batch a second 207-wide buffer on every rollout/update without a
        reader.
        """
        td_dict: dict[str, Any] = {"actor": _to_torch(obs["obs"], self.device)}
        if "critic" in obs:
            td_dict["critic"] = _to_torch(obs["critic"], self.device)
        return TensorDict(td_dict, batch_size=self.num_envs, device=self.device)

    def export_training_state(self) -> Mapping[str, Any]:
        state = task(self.env)
        return {
            "schema": "wuji_reorient",
            "version": 1,
            "progress": self.env.export_training_state(),
            "difficulty": state.difficulty,
            "adaptive": state.adaptive,
            "obs_groups": dict(self.env.obs_groups_spec),
        }

    def import_training_state(self, value: Mapping[str, Any]) -> None:
        if value.get("schema") != "wuji_reorient" or value.get("version") != 1:
            raise ValueError("Unsupported Wuji training state schema")
        if value.get("obs_groups") != dict(self.env.obs_groups_spec):
            raise ValueError("Wuji checkpoint observation groups do not match the environment")
        difficulty, adaptive = value.get("difficulty"), value.get("adaptive")
        if (
            not isinstance(difficulty, (int, float))
            or not isinstance(adaptive, (int, float))
            or isinstance(difficulty, bool)
            or isinstance(adaptive, bool)
            or not np.isfinite([difficulty, adaptive]).all()
            or not 0 <= difficulty <= 1
            or not 0.05 <= adaptive <= 1
        ):
            raise ValueError("Invalid Wuji curriculum state")
        self.env.import_training_state(value["progress"])
        state = task(self.env)
        state.difficulty, state.adaptive = float(difficulty), float(adaptive)


class WujiOnPolicyRunner(OnPolicyRunner):
    """Round-trip Wuji curriculum progress alongside the rsl-rl checkpoint.

    The wrapped env must implement ``export_training_state`` /
    ``import_training_state`` (see :class:`WujiWrapper`).  Load errors abort
    resume: algorithm and owner state loading is not a transaction that rolls
    back a successfully loaded algorithm on owner error.
    """

    def __init__(
        self,
        env: Any,
        train_cfg: dict[str, Any],
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        if not (hasattr(env, "export_training_state") and hasattr(env, "import_training_state")):
            raise TypeError(
                "WujiOnPolicyRunner requires a wrapper implementing "
                "export_training_state/import_training_state"
            )
        self.training_state_provider = env
        super().__init__(env, train_cfg, log_dir, device)

    def save(self, path: str, infos: dict | None = None) -> None:
        state = _copy_training_state(self.training_state_provider.export_training_state())
        checkpoint_infos = dict(infos or {})
        checkpoint_infos[_STATE_KEY] = {"version": _STATE_VERSION, "state": state}
        super().save(path, checkpoint_infos)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
        *,
        restore_training_state: bool = True,
    ) -> dict:
        state: dict[str, Any] | None = None
        if not restore_training_state and (
            load_cfg is None
            or not load_cfg.get("actor")
            or any(enabled for key, enabled in load_cfg.items() if key != "actor")
        ):
            raise ValueError(
                "restore_training_state=False requires an explicit actor-only load_cfg; "
                "optimizer/iteration resume cannot omit curriculum state"
            )
        if restore_training_state:
            # Preflight the envelope before the parent mutates algorithm state.
            checkpoint = torch.load(path, weights_only=False, map_location=map_location)
            infos = checkpoint.get("infos") if isinstance(checkpoint, Mapping) else None
            envelope = infos.get(_STATE_KEY) if isinstance(infos, Mapping) else None
            if not isinstance(envelope, Mapping):
                raise ValueError(
                    "Checkpoint is missing required training state; curriculum resume "
                    "cannot proceed. Use restore_training_state=False only for an "
                    "explicit policy-only load."
                )
            version = envelope.get("version")
            if type(version) is not int or version != _STATE_VERSION:
                raise ValueError(f"Unsupported training state envelope version: {version!r}")
            if not isinstance(envelope.get("state"), Mapping):
                raise ValueError("Invalid training state envelope: 'state' must be a mapping")
            state = _copy_training_state(envelope["state"])

        result: dict = super().load(path, load_cfg, strict, map_location)
        if state is not None:
            self.training_state_provider.import_training_state(state)
        return result

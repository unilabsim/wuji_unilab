"""Hand 1 environment factory and the SuperMimic transition boundary."""

from __future__ import annotations

from typing import Any

import torch
from unilab.base.backend_factory import create_backend, env_backend_kwargs
from unilab.base.cpu_runtime import apply_cpu_confinement
from unilab.base.entity import EntityCfg
from unilab.base.reset_selectors import resolve_reset_env_ids
from unilab.base.torch_env import TorchEnvState
from unilab.envs.manager_based_rl_env import (
    ManagerBasedRlEnv,
    ManagerBasedRlEnvCfg,
    _resolve_backend_entity_contract,
)
from unisim.backend.base import SimBackend

from unisupermimic.rl.contracts.supermimic import (
    SUPERMIMIC_STATE_KEY,
    SUPERMIMIC_TRANSITION_KEY,
    SuperMimicMdpProtocol,
)

from .commands import Hand1ReorientCommand
from .memory import flatten_memory


def _clone(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone()


def _clone_memory(memory: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: _clone(value) for name, value in memory.items()}


def _merge_info(
    previous: dict[str, Any], current: dict[str, Any], reset_mask: torch.Tensor
) -> dict[str, Any]:
    """Merge selected reset rows while retaining live rows from ``previous``."""
    if set(previous) != set(current):
        raise ValueError("Hand1 SuperMimic current-info keys changed across reset")
    merged: dict[str, Any] = {}
    for name, value in current.items():
        old = previous[name]
        if isinstance(value, dict):
            if not isinstance(old, dict) or set(value) != set(old):
                raise ValueError(f"Hand1 memory keys changed across reset for {name!r}")
            merged[name] = _merge_info(old, value, reset_mask)
            continue
        if not isinstance(value, torch.Tensor) or not isinstance(old, torch.Tensor):
            merged[name] = value
            continue
        mask = reset_mask.reshape((reset_mask.shape[0],) + (1,) * (value.ndim - 1))
        merged[name] = torch.where(mask, value, old).detach().clone()
    return merged


class Hand1SuperMimicManagerBasedRlEnv(ManagerBasedRlEnv):
    """Manager environment exposing Hand 1 SuperMimic snapshots."""

    def __init__(self, cfg: ManagerBasedRlEnvCfg, backend: SimBackend, num_envs: int):
        super().__init__(cfg, backend, num_envs)
        commands = [
            self.command_manager.get_term(name)
            for name in self.command_manager.active_terms
            if isinstance(self.command_manager.get_term(name), Hand1ReorientCommand)
        ]
        if len(commands) != 1:
            raise ValueError("Hand1 SuperMimic requires exactly one reorientation command")
        self._hand1_command = commands[0]
        self._pre_action: dict[str, Any] | None = None

    @property
    def supermimic_mdp(self) -> SuperMimicMdpProtocol:
        mdp = getattr(self._hand1_command, "mdp", None)
        if mdp is None:
            raise RuntimeError("Hand1 command has no SuperMimic MDP adapter")
        return mdp

    def _current_info(self) -> dict[str, Any]:
        command = self._hand1_command
        memory = _clone_memory(command.memory_tensor())
        return {
            "s": _clone(command.state_tensor()),
            "xi": _clone(command.context_tensor()),
            "a_prev": _clone(self.action_manager.prev_action),
            "ep_step": _clone(self.episode_length_buf),
            "f": _clone(self.episode_length_buf),
            "memory": memory,
            "memory_flat": _clone(flatten_memory(memory)),
            "auxiliary": _clone(self._hand1_command.auxiliary_tensor()),
        }

    def apply_action(self, actions: torch.Tensor, state: TorchEnvState) -> torch.Tensor:
        memory = _clone_memory(self._hand1_command.memory_tensor())
        self._pre_action = {
            "state": _clone(self._hand1_command.state_tensor()),
            "context": _clone(self._hand1_command.context_tensor()),
            "previous_action": _clone(self.action_manager.prev_action),
            "ep_step": _clone(self.episode_length_buf),
            "memory": memory,
            "memory_flat": _clone(flatten_memory(memory)),
            "auxiliary": _clone(self._hand1_command.auxiliary_tensor()),
            "orientation_error": _clone(self._hand1_command.error()),
            "action": _clone(actions),
        }
        return super().apply_action(actions, state)

    def _update_state_in_read_phase(self, state: TorchEnvState) -> TorchEnvState:
        # UniLab applies interval/step events after the reward snapshot and
        # before command updates.  Capture the physical post-step state at
        # that exact boundary so the world-model target excludes an external
        # velocity jump, matching the source runner's ``cap["physics"]``.
        if self._pre_action is None:
            raise RuntimeError("Hand1 SuperMimic update requires a preceding action snapshot")
        # The parent update_state has refreshed the post-physics read plan.
        # Drop command-owned pose caches from the preceding control frame
        # before termination/reward consume them. Playback has no training
        # reader to invalidate these caches at the reward-capture boundary.
        self._hand1_command._pose_stale = True
        physics_state = _clone(self._hand1_command.state_tensor())
        updated = super()._update_state_in_read_phase(state)
        command = self._hand1_command
        post_event_state = _clone(command.state_tensor())
        next_memory = _clone_memory(command.memory_tensor())
        transition = {
            "s": self._pre_action["state"],
            "s1": physics_state,
            # This is the state jump produced by runtime interval events. It
            # is applied after the predicted physics state during supervised
            # multi-step WM fitting, exactly as in Dex-SuperMimic.
            "push": post_event_state - physics_state,
            "a": self._pre_action["action"],
            "a_prev": self._pre_action["previous_action"],
            "xi": self._pre_action["context"],
            "r": _clone(updated.reward),
            "f": self._pre_action["ep_step"].to(dtype=torch.long),
            "ep_step": self._pre_action["ep_step"].to(dtype=torch.int32),
            "term": _clone(updated.terminated),
            "done": _clone(updated.terminated | updated.truncated),
            "timeout": _clone(updated.truncated),
            "m": self._pre_action["memory"],
            "m1": next_memory,
            "m_flat": self._pre_action["memory_flat"],
            "m1_flat": _clone(flatten_memory(next_memory)),
            "auxiliary": self._pre_action["auxiliary"],
            "contact": _clone(command.contact_tensor()),
            "torque": _clone(command.torque_tensor()),
            # Evaluation-owned task events.  These are captured before an
            # automatic reset can clear command counters, while remaining
            # optional for learners that only consume the core transition
            # contract above.
            "goal_count": _clone(command.goal_count),
            "orientation_error": _clone(command.error()),
            "orientation_error_before": self._pre_action["orientation_error"],
            "cage_counter": _clone(command.cage_counter),
            "outside": _clone(command.outside),
            "success": _clone(command.success),
        }
        updated.info[SUPERMIMIC_TRANSITION_KEY] = transition
        updated.info[SUPERMIMIC_STATE_KEY] = self._current_info()
        self._pre_action = None
        return updated

    def reset(
        self,
        env_ids: torch.Tensor | None = None,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        if env_ids is None:
            reset_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            ids = resolve_reset_env_ids(
                env_ids,
                num_envs=self.num_envs,
                device=self.device,
                owner="Hand1SuperMimicManagerBasedRlEnv.reset",
            )
            reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            reset_mask[ids] = True
        obs, info = super().reset(env_ids, seed=seed, options=options)
        assert self._state is not None
        current = self._current_info()
        previous = self._state.info.get(SUPERMIMIC_STATE_KEY)
        if previous is not None:
            current = _merge_info(previous, current, reset_mask)
        self._state.info[SUPERMIMIC_STATE_KEY] = current
        info[SUPERMIMIC_STATE_KEY] = current
        return obs, info


SUPPORTED_BACKENDS = ("mjwarp", "mujoco")


def make_env(
    cfg: ManagerBasedRlEnvCfg,
    num_envs: int = 1,
    backend_type: str = "mjwarp",
) -> Hand1SuperMimicManagerBasedRlEnv:
    """Construct Hand 1 through UniLab's public backend factory."""
    if backend_type not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"DexHand1Reorient supports backends {', '.join(SUPPORTED_BACKENDS)}; "
            f"got {backend_type!r}"
        )
    if not isinstance(cfg, ManagerBasedRlEnvCfg):
        raise TypeError("make_env expected ManagerBasedRlEnvCfg")
    assert cfg.scene is not None
    robot_cfg = cfg.scene.entities.get("robot")
    if not isinstance(robot_cfg, EntityCfg):
        raise TypeError("Hand1 robot scene entity must be an EntityCfg")
    cfg.validate()
    apply_cpu_confinement(cfg.cpu_ids)
    base_name, body_state_required, tracked_body_names = _resolve_backend_entity_contract(cfg)
    kwargs = env_backend_kwargs(cfg)
    kwargs["base_name"] = base_name
    if backend_type in ("mjwarp", "mujoco"):
        # UniSim's mjwarp interval velocity capability is bound to its
        # configured free root.  Hand1's controlled robot root is a mocap body,
        # while the cube owns the actual free joint receiving disturbances.
        # Keep the manager/entity contract unchanged and select the cube only
        # for the backend's interval-push target.
        kwargs["base_name"] = "object/cube"
        kwargs["push_body_name"] = "object/cube"
    if backend_type == "mujoco" and tracked_body_names is not None:
        kwargs["tracked_body_names"] = tracked_body_names
    backend = create_backend(
        backend_type,
        cfg.scene,
        num_envs,
        cfg.sim_dt,
        body_state_required=body_state_required,
        **kwargs,
    )
    try:
        return Hand1SuperMimicManagerBasedRlEnv(cfg, backend, num_envs)
    except Exception:
        backend.cleanup_scene_assets()
        raise


__all__ = ["Hand1SuperMimicManagerBasedRlEnv", "SUPPORTED_BACKENDS", "make_env"]

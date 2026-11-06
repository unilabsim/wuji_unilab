"""Environment configuration shared by the Hand 1 SuperMimic entry point."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf
from unilab.base import registry
from unilab.base.config_adapter import EnvConfigAdapter

from .registry import TASK_NAME, register_resolvers


def make_env_cfg(*, backend_type="mujoco", control_hz=50, seed=42, play=False):
    if backend_type not in ("mujoco", "mjwarp"):
        raise ValueError("Hand 1 supports mujoco and mjwarp")
    if control_hz not in (20, 50):
        raise ValueError("Hand 1 source supports 20 Hz and 50 Hz")
    register_resolvers()
    root = Path(__file__).resolve().parents[3] / "conf/supermimic"
    cfg = OmegaConf.load(root / "task/dex_hand1_reorient/base.yaml")
    overlay = OmegaConf.load(root / f"task/dex_hand1_reorient/{backend_type}.yaml")
    cfg = OmegaConf.merge(cfg, {"env": overlay.get("env", {})})
    if control_hz == 50:
        recipe = OmegaConf.load(root / "config_hand1.yaml")
        cfg = OmegaConf.merge(cfg, {"env": recipe.env, "reward": recipe.reward})
    cfg.env.seed = seed
    cfg.env.auto_reset = False
    if play:
        cfg.env.max_episode_seconds = 60.0
        cfg.env.observations.policy.enable_corruption = False
        for name in ("position", "orientation"):
            cfg.env.observations.policy.terms[name].params.injection_prob = 0.0
        cfg.env.events.randomize = None
        cfg.env.events.disturbance = None
        cfg.env.curriculum = {}
    return EnvConfigAdapter(cfg, root_dir=Path.cwd(), algo_name="supermimic_hand1").build_task_env_cfg_override()


def make_hand1_env(*, num_envs=1024, backend_type="mjwarp", control_hz=50, seed=42, play=False):
    cfg = make_env_cfg(backend_type=backend_type, control_hz=control_hz, seed=seed, play=play)
    return registry.make(TASK_NAME, sim_backend=backend_type, env_cfg_override=cfg, num_envs=num_envs)


__all__ = ["make_env_cfg", "make_hand1_env"]

"""Train the Hand 2 SuperMimic world model on UniLab.

This entry point intentionally has its own Hydra config.  The existing
``unisupermimic-train`` command is the G1 motion-library runner and performs
G1-only configuration checks; using it for the dexterous hand silently selects
the wrong state/goal contract.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from unilab.base.config_adapter import EnvConfigAdapter, create_env
from unilab.training import ensure_registries, resolve_run_log_dir
from unilab.training.seed import apply_configured_training_seed
from unilab.utils.device import get_default_device

from unisupermimic.rl.algos.supermimic.hand2_runner import (
    Hand2SuperMimicConfig,
    Hand2SuperMimicRunner,
)


def build_runner_config(cfg: DictConfig) -> Hand2SuperMimicConfig:
    raw = OmegaConf.to_container(cfg.algo, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("cfg.algo must resolve to a mapping")
    values = {
        field.name: raw[field.name]
        for field in fields(Hand2SuperMimicConfig)
        if field.name in raw
    }
    for name in ("wm_hidden", "policy_hidden", "critic_hidden"):
        if name in values:
            values[name] = tuple(values[name])
    return Hand2SuperMimicConfig(**values)


def _close_env(env) -> None:
    env.close()
    backend = getattr(env, "backend", None)
    if backend is not None:
        backend.close()


def _select_backend_task(cfg: DictConfig) -> DictConfig:
    """Apply the backend capability overlay when only sim_backend is overridden."""
    root = Path(__file__).parents[1] / "conf/supermimic"
    backend = str(cfg.training.sim_backend)
    if backend == "mjwarp":
        OmegaConf.set_struct(cfg, False)
        base = OmegaConf.load(root / "task/dex_hand2_reorient/base.yaml")
        warp = OmegaConf.load(root / "task/dex_hand2_reorient/mjwarp.yaml")
        cfg = OmegaConf.merge(
            cfg,
            {
                "env": {
                    "events": {
                        "randomize": base.env.events.randomize,
                        "disturbance": base.env.events.disturbance,
                    },
                    "scene": {"entities": warp.env.scene.entities},
                }
            },
        )
    elif backend == "mujoco":
        mujoco = OmegaConf.load(root / "task/dex_hand2_reorient/mujoco.yaml")
        cfg = OmegaConf.merge(cfg, {"env": {"events": mujoco.env.events}})
    return cfg


def main(cfg: DictConfig) -> None:
    ensure_registries()
    # Hydra delays interpolation resolution until the environment override is
    # built.  Register the local scene resolver explicitly because this entry
    # point is also usable from a source checkout without plugin discovery.
    from unisupermimic.tasks.dex_reorient.hand2.registry import register_resolvers

    register_resolvers()
    if str(cfg.training.task_name) != "DexHand2Reorient":
        raise ValueError(
            "train_hand2_supermimic requires training.task_name=DexHand2Reorient; "
            f"got {cfg.training.task_name!r}"
        )
    cfg = _select_backend_task(cfg)
    seed = apply_configured_training_seed(cfg, torch_runtime=True, cuda=True)
    device = get_default_device()
    override = EnvConfigAdapter(
        cfg, root_dir=Path.cwd(), algo_name="supermimic_hand2"
    ).build_task_env_cfg_override()
    env = None
    try:
        env = create_env(
            cfg,
            num_envs=int(cfg.algo.num_envs),
            env_cfg_override=override,
        )
        log_dir = Path(cfg.training.log_dir) if cfg.training.log_dir else Path(resolve_run_log_dir(cfg))
        learner = Hand2SuperMimicRunner(
            env,
            build_runner_config(cfg),
            log_dir,
            device=str(device),
        )
        summary = learner.train()
        summary["seed"] = seed.to_dict()
        print(json.dumps(summary, indent=2))
    finally:
        if env is not None:
            _close_env(env)


@hydra.main(
    version_base="1.3", config_path="../conf/supermimic", config_name="config_hand2"
)
def hydra_main(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_main()

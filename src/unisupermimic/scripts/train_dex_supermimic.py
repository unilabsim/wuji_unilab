"""Source SuperMimic learner for the independently migrated Hand1/Pen tasks."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from unilab.base.config_adapter import EnvConfigAdapter, create_env
from unilab.training import ensure_registries
from unilab.training.seed import apply_configured_training_seed

from unisupermimic.rl.algos.dex_supermimic.config import DexSuperMimicCfg
from unisupermimic.rl.algos.dex_supermimic.env_bridge import SourceEnvBridge


def build_runner_config(cfg: DictConfig) -> DexSuperMimicCfg:
    raw = OmegaConf.to_container(cfg.algo, resolve=True)
    assert isinstance(raw, dict)
    raw["max_iterations"] = raw.pop("iterations")
    accepted = {field.name for field in fields(DexSuperMimicCfg)}
    unknown = set(raw) - accepted
    if unknown:
        raise ValueError(f"Unsupported SuperMimic options: {sorted(unknown)}")
    raw["log_dir"] = str(Path(cfg.training.log_dir).resolve())
    raw["wm_hidden"] = tuple(raw["wm_hidden"])
    return DexSuperMimicCfg(**raw)


def main(cfg: DictConfig) -> None:
    from unisupermimic.rl.algos.dex_supermimic.runner import DexSuperMimicRunner

    ensure_registries()
    task = str(cfg.algo.task)
    if task == "hand2-pen-spin":
        from unisupermimic.tasks.pen_spin.registry import register_resolvers

        task_folder = "dex_hand2_pen_spin"
        expected = "DexHand2PenSpin"
    elif task == "hand1-reorient":
        from unisupermimic.tasks.dex_reorient.hand1.registry import register_resolvers

        task_folder = "dex_hand1_reorient"
        expected = "DexHand1Reorient"
    else:
        raise ValueError("This entry supports hand1-reorient and hand2-pen-spin only")
    if str(cfg.training.task_name) != expected:
        raise ValueError(f"{task} requires training.task_name={expected}")
    register_resolvers()
    backend = str(cfg.training.sim_backend)
    if backend not in ("mjwarp", "mujoco"):
        raise ValueError("Choose sim_backend=mjwarp or mujoco")
    if backend == "mjwarp" and not torch.cuda.is_available():
        raise RuntimeError("MJWarp training requires CUDA; use training.sim_backend=mujoco for CPU smoke")
    if backend == "mujoco":
        # Physics and learner stay on one device in the source runner.
        cfg.training.collector_tensor_device = "cpu"
        torch.set_num_threads(int(cfg.training.torch_num_threads))
    overlay = Path(__file__).parents[1] / "conf/supermimic/task" / task_folder / f"{backend}.yaml"
    overlay_cfg = OmegaConf.load(overlay)
    if "env" in overlay_cfg:
        cfg = OmegaConf.merge(cfg, {"env": overlay_cfg.env})
    seed = apply_configured_training_seed(cfg, torch_runtime=True, cuda=True)
    # UniLab's training helper seeds Python/Torch, while entity RNGs consume
    # env.seed separately. Keep all source runtime streams on algo.seed.
    cfg.env.seed = int(cfg.algo.seed)
    np.random.seed(int(cfg.algo.seed))
    runner_cfg = build_runner_config(cfg)
    log_dir = Path(runner_cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    if (log_dir / "config.json").exists() and not runner_cfg.checkpoint:
        raise FileExistsError(
            f"{log_dir} already contains a training run. Set algo.checkpoint to resume "
            "or training.log_dir to a new directory."
        )
    override = EnvConfigAdapter(cfg, root_dir=Path.cwd(), algo_name="supermimic").build_task_env_cfg_override()
    env = runner = None
    try:
        env = create_env(cfg, num_envs=runner_cfg.num_envs, env_cfg_override=override)
        bridge = SourceEnvBridge(env)
        runner = DexSuperMimicRunner(bridge, runner_cfg, log_dir)
        layout = json.loads((log_dir / "input_layout.json").read_text())
        print(
            f"algorithm=SuperMimic task={task} backend={backend} "
            f"actor_input={runner.obs.shape[-1]} wm_input={layout['wm_input_dim']} "
            f"critic_input={layout['critic_input_dim']} log_dir={log_dir}",
            flush=True,
        )
        summary = runner.learn()
        summary["seed"] = seed.to_dict()
        (log_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    finally:
        if runner is not None:
            runner.close()
        if env is not None:
            env.close()


@hydra.main(version_base="1.3", config_path="../conf/supermimic", config_name="config_pen_spin_supermimic")
def pen_main(cfg: DictConfig) -> None:
    main(cfg)


@hydra.main(version_base="1.3", config_path="../conf/supermimic", config_name="config_hand1_supermimic")
def hand1_main(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    pen_main()

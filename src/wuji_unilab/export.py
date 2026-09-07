"""Export a schema-described Wuji policy and require ONNX numerical agreement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
import torch
import warp as wp
from uni_rl.algos.rsl_rl import normalize_ppo_train_cfg
from uni_rl.algos.rsl_rl_training_state import TrainingStateOnPolicyRunner
from unilab.base import registry
from unilab.envs import ManagerBasedRlEnv
from unilab.training import algo_config_dict
from unilab.training.onnx_export import export_policy_onnx, verify_policy_onnx

from wuji_unilab.config import compose_task, env_overrides
from wuji_unilab.rl.runtime import WujiWrapper

EXPORT_BATCH_SIZE = 1


def _parser():
    parser = argparse.ArgumentParser(prog="wuji-export")
    parser.add_argument("--checkpoint-file", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--task",
        choices=("WujiHand_Reorient", "WujiHand_Reorient_Light"),
        default="WujiHand_Reorient",
    )
    parser.add_argument("--max-diff", type=float, default=1e-4)
    return parser


def build_policy_schema(
    *,
    task: str,
    control_dt_s: float,
    observation_dim: int,
    action_dim: int,
    joint_names: list[str],
    checkpoint_file: Path,
    max_abs_diff: float,
    mean_abs_diff: float,
) -> dict[str, object]:
    """Describe the fixed policy I/O contract beside the exported artifact."""
    if observation_dim <= 0 or action_dim <= 0:
        raise ValueError("Policy observation and action dimensions must be positive")
    if not joint_names:
        raise ValueError("Wuji policy schema requires ordered joint names")
    return {
        "schema": "wuji_unilab_policy",
        "version": 1,
        "task": task,
        "physics_backend": "mjwarp",
        "control_dt_s": control_dt_s,
        "observation": {"name": "obs", "shape": [EXPORT_BATCH_SIZE, observation_dim]},
        "action": {"name": "actions", "shape": [EXPORT_BATCH_SIZE, action_dim]},
        "joint_names": joint_names,
        "checkpoint": str(checkpoint_file.resolve()),
        "onnx_max_abs_diff": max_abs_diff,
        "onnx_mean_abs_diff": mean_abs_diff,
    }


def main() -> None:
    args = _parser().parse_args()
    if not args.checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_file}")
    if args.max_diff <= 0 or not np.isfinite(args.max_diff):
        raise ValueError("--max-diff must be finite and positive")
    wp.init()
    wp.set_device("cuda:0")
    cfg = compose_task(args.task, overrides=["algo.num_envs=1"])
    env = cast(
        ManagerBasedRlEnv,
        registry.make(
            cfg.training.task_name,
            sim_backend="mjwarp",
            env_cfg_override=env_overrides(cfg),
            num_envs=1,
        ),
    )
    wrapper = WujiWrapper(env, device="cuda:0")
    try:
        runner = TrainingStateOnPolicyRunner(
            wrapper, normalize_ppo_train_cfg(algo_config_dict(cfg)), device="cuda:0"
        )
        runner.load(str(args.checkpoint_file), map_location="cuda:0")
        actor = runner.alg.get_policy()
        module = actor.as_onnx(verbose=False).eval()
        output = args.output or args.checkpoint_file.with_suffix(".onnx")
        output.parent.mkdir(parents=True, exist_ok=True)
        obs = torch.randn(EXPORT_BATCH_SIZE, int(env.obs_groups_spec["obs"]), device="cuda:0")
        export_policy_onnx(
            module, str(output), (obs,), input_names=["obs"], output_names=["actions"]
        )
        max_diff, mean_diff = verify_policy_onnx(
            module, str(output), (obs,), input_names=["obs"], max_diff_tol=args.max_diff
        )
        if max_diff > args.max_diff:
            raise RuntimeError(
                f"Refusing ONNX artifact: max difference {max_diff:.3e} exceeds {args.max_diff:.3e}"
            )
        action_shape = env.action_space.shape
        if action_shape is None:
            raise ValueError("Wuji action space must declare a concrete shape")
        schema = build_policy_schema(
            task=args.task,
            control_dt_s=float(env.step_dt),
            observation_dim=int(env.obs_groups_spec["obs"]),
            action_dim=int(action_shape[0]),
            joint_names=list(env.scene["robot"].joint_names),
            checkpoint_file=args.checkpoint_file,
            max_abs_diff=max_diff,
            mean_abs_diff=mean_diff,
        )
        output.with_suffix(".json").write_text(json.dumps(schema, indent=2) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()

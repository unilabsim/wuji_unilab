"""Compile the Hand 2 scene and optionally construct a UniLab environment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--construct", action="store_true", help="also construct one CPU MuJoCo env")
    parser.add_argument(
        "--step",
        action="store_true",
        help="after construction, reset and execute one zero-action transition",
    )
    args = parser.parse_args()

    from unisupermimic.tasks.dex_reorient.hand2.scene import materialize_scene

    path = materialize_scene()
    print(f"scene={path}")
    if not args.construct and not args.step:
        return
    from hydra import compose, initialize_config_dir
    from unilab.base import registry
    from unilab.base.config_adapter import EnvConfigAdapter
    from unilab.training import ensure_registries

    # Keep this utility usable as a standalone script.  Importing the scene
    # module above does not guarantee that UniLab has loaded package entry
    # points, while ``registry.make`` requires the task registration.
    ensure_registries()

    conf = Path(__file__).parents[1] / "src/unisupermimic/conf/supermimic"
    with initialize_config_dir(config_dir=str(conf), version_base="1.3"):
        cfg = compose("config_hand2", overrides=["task=dex_hand2_reorient/mujoco", "algo.num_envs=1"])
    override = EnvConfigAdapter(cfg, root_dir=Path.cwd(), algo_name="supermimic").build_task_env_cfg_override()
    env = registry.make("DexHand2Reorient", sim_backend="mujoco", env_cfg_override=override, num_envs=1)
    try:
        if args.step:
            import torch

            env.reset()
            state = env.step(torch.zeros((1, 20), device=env.device))
            transition = state.info["supermimic"]
            required = {"s", "s1", "push", "xi", "m", "m1", "contact", "torque"}
            missing = required.difference(transition)
            if missing:
                raise RuntimeError(f"Hand2 transition is missing fields: {sorted(missing)}")
            print(
                "transition=ok "
                f"state={tuple(transition['s'].shape)} "
                f"context={tuple(transition['xi'].shape)} "
                f"contact={tuple(transition['contact'].shape)}"
            )
        else:
            env.reset()
        print(f"env=ok obs_groups={tuple(env.state.obs)}")
    finally:
        env.close()


if __name__ == "__main__":
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/unisupermimic-cache")
    main()

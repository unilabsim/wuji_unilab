"""Train the migrated Hand 2 task with UniLab's native rsl_rl PPO runner."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


def main(cfg: DictConfig) -> None:
    # Keep UniLab's runner as the single owner of PPO lifecycle, checkpointing,
    # and playback.  Calling its undecorated function avoids a second Hydra
    # composition and lets this package keep its own task configuration.
    from unilab.scripts.train_rsl_rl import main as train_rsl_rl_main

    train_rsl_rl_main.__wrapped__(cfg)


@hydra.main(version_base="1.3", config_path="../conf/supermimic", config_name="config_ppo")
def hydra_main(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_main()


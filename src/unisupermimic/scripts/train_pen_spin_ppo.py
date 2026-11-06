"""Train Wuji Hand 2 Pen Spin with UniLab's native PPO runner."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


def main(cfg: DictConfig) -> None:
    from unilab.scripts.train_rsl_rl import main as train_rsl_rl_main

    train_rsl_rl_main.__wrapped__(cfg)


@hydra.main(version_base="1.3", config_path="../conf/supermimic", config_name="config_pen_spin")
def hydra_main(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_main()

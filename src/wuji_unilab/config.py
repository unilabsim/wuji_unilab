"""Programmatic composition of the same owner configuration as the CLI."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from unilab.base.config_adapter import BackendAdapter
from unilab.cli import package_root

from wuji_unilab.tasks.reorient import register_resolvers

CONF_ROOT = Path(__file__).parent / "conf/ppo"
TASK_OWNERS = {
    "WujiHand_Reorient": "wuji_reorient",
    "WujiHand_Reorient_Light": "wuji_reorient_light",
}


def compose_task(task="WujiHand_Reorient", overrides=()):
    register_resolvers()
    if task not in TASK_OWNERS:
        raise ValueError(f"Unknown Wuji task {task!r}; choices: {list(TASK_OWNERS)}")
    with initialize_config_dir(config_dir=str(package_root() / "conf/ppo"), version_base="1.3"):
        return compose(
            "config",
            overrides=[
                f"hydra.searchpath=[file://{CONF_ROOT}]",
                f"task={TASK_OWNERS[task]}/mjwarp",
                *overrides,
            ],
        )


def env_overrides(cfg):
    return BackendAdapter(cfg, root_dir=Path.cwd(), algo_name="ppo").build_task_env_cfg_override()

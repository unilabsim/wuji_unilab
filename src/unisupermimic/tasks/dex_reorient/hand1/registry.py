"""UniLab registration for Wuji Hand 1 cube reorientation."""

from omegaconf import OmegaConf
from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg

from .env import make_env
from .scene import materialize_scene

TASK_NAME = "DexHand1Reorient"


def register_resolvers() -> None:
    if not OmegaConf.has_resolver("hand1_scene"):
        OmegaConf.register_new_resolver("hand1_scene", lambda: str(materialize_scene()))


register_resolvers()
registry.register_env_config(TASK_NAME, ManagerBasedRlEnvCfg)
for _backend in ("mjwarp", "mujoco"):
    registry.register_env(TASK_NAME, make_env, sim_backend=_backend)


__all__ = ["TASK_NAME", "make_env", "register_resolvers"]

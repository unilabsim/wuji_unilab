"""Wuji task registry; all configurations are owned by packaged Hydra YAML."""

from omegaconf import OmegaConf
from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env

from .scene import materialize_scene


def register_resolvers():
    if not OmegaConf.has_resolver("wuji_scene"):
        OmegaConf.register_new_resolver("wuji_scene", lambda: str(materialize_scene().path))


def make_env(cfg, num_envs=1, backend_type="mjwarp"):
    if backend_type != "mjwarp":
        raise ValueError("Wuji supports only mjwarp physics")
    register_resolvers()
    return make_manager_based_rl_env(cfg, num_envs=num_envs, backend_type=backend_type)


register_resolvers()
for task_name in ("WujiHand_Reorient", "WujiHand_Reorient_Light"):
    registry.register_env_config(task_name, ManagerBasedRlEnvCfg)
    registry.register_env(task_name, make_env, sim_backend="mjwarp")

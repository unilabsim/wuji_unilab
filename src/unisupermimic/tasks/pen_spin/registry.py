"""UniLab registration for the Wuji Hand 2 pen-spin task."""

from omegaconf import OmegaConf
from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env

from .reset_bridge import bind_cpu_observation_phase
from .scene import materialize_scene

TASK_NAME = "DexHand2PenSpin"


def register_resolvers() -> None:
    if not OmegaConf.has_resolver("pen_spin_scene"):
        OmegaConf.register_new_resolver("pen_spin_scene", lambda: str(materialize_scene()))


def make_env(cfg, num_envs=1, backend_type="mjwarp"):
    if backend_type not in ("mjwarp", "mujoco"):
        raise ValueError(f"{TASK_NAME} supports mjwarp and mujoco; got {backend_type!r}")
    register_resolvers()
    env = make_manager_based_rl_env(cfg, num_envs=num_envs, backend_type=backend_type)
    bind_cpu_observation_phase(env)
    return env


register_resolvers()
registry.register_env_config(TASK_NAME, ManagerBasedRlEnvCfg)
for _backend in ("mjwarp", "mujoco"):
    registry.register_env(TASK_NAME, make_env, sim_backend=_backend)

__all__ = ["TASK_NAME", "make_env", "register_resolvers"]

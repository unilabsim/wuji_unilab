"""UniLab translation of Dex-Supermimic's Pen Spin ``play=True`` profile.

The source profile lives in ``tasks/pen_spin/config/wuji_hand2/env_cfgs.py``.
It uses clean current pen observations and a first-frame hold, rather than
replaying training's camera state or adaptive random start frames.
"""

from __future__ import annotations

import copy
from pathlib import Path

import mujoco
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict
from unilab.managers import ManagerTermBase

from .randomization import default_model_fields, randomize_pen_dimensions
from .registry import register_resolvers
from .reset_bridge import get_pen_physics_model, publish_startup_model

HARDWARE_PEN_DIAMETER_M = 0.0149
HARDWARE_PEN_LENGTH_M = 0.250
PEN_MASS_KG = 0.030


class FixedHardwarePenDimensions(ManagerTermBase):
    """Pin the source hardware pen geometry and its consistent inertia once."""

    def __init__(self, cfg, env):
        del cfg
        super().__init__(env)
        self._applied = False

    def __call__(self, env, env_ids=None):
        del env_ids
        if self._applied:
            raise RuntimeError("Hardware pen dimensions are a startup-only event")
        model = get_pen_physics_model(env)
        defaults = default_model_fields(model, env.num_envs, env.device)
        fields = {name: value.clone() for name, value in defaults.items()}
        fields["body_mass"][:, model.body("object/cube").id] = PEN_MASS_KG
        randomize_pen_dimensions(
            fields, model,
            diameter_range=(HARDWARE_PEN_DIAMETER_M, HARDWARE_PEN_DIAMETER_M),
            length_range=(HARDWARE_PEN_LENGTH_M, HARDWARE_PEN_LENGTH_M),
        )
        publish_startup_model(env, fields)
        env.scene["robot"].data.encoder_bias_tensor.zero_()
        env._pen_default_model_fields = defaults
        env._pen_model_fields = fields
        env._pen_physics_model = model
        # The play profile has no force-producing event. Keep the source
        # reader's optional exogenous-input channel explicitly zero as well.
        env._pen_external_wrench = torch.zeros(
            (env.num_envs, model.nbody, 6), device=env.device
        )
        self._applied = True


def prepare_playback_model(env) -> mujoco.MjModel:
    """Copy fixed startup parameters into a detached visual model by name.

    UniSim's visual oracle intentionally omits runtime model-table writes.
    The executor also strips visual-only geometries, so its integer geometry
    IDs cannot index that oracle. Only this viewer copy receives the actual
    world's pen size/marker pose, collision bounds and inertia.
    """
    oracle = env.get_playback_model(0)
    visual = (
        mujoco.MjModel.from_xml_path(str(oracle))
        if isinstance(oracle, (str, Path))
        else copy.deepcopy(oracle)
    )
    physics = env._pen_physics_model
    fields = env._pen_model_fields
    for kind, count, names in (
        (mujoco.mjtObj.mjOBJ_GEOM, physics.ngeom,
         ("geom_size", "geom_pos", "geom_rbound", "geom_aabb")),
        (mujoco.mjtObj.mjOBJ_BODY, physics.nbody,
         ("body_mass", "body_inertia", "body_iquat", "body_ipos")),
    ):
        for source_id in range(count):
            name = mujoco.mj_id2name(physics, kind, source_id)
            if name is None:
                continue
            target_id = mujoco.mj_name2id(visual, kind, name)
            if target_id < 0:
                raise ValueError(f"Pen playback visual model has no {name!r}")
            for field in names:
                target = getattr(visual, field)
                value = fields[field][0, source_id].detach().cpu().numpy()
                target[target_id] = np.asarray(value).reshape(np.shape(target[target_id]))
    mujoco.mj_setConst(visual, mujoco.MjData(visual))
    return visual


def build_play_config(
    *,
    backend: str = "mujoco",
    seed: int = 42,
    recipe_path: str | Path | None = None,
    clips_root: str | Path | None = None,
) -> DictConfig:
    """Compose the source Pen play contract without mutating training config.

    A single environment follows a uniformly selected whole clip. Reference
    residual/EMA control, observation ordering and saved actor normalization
    remain the same as training; only the source's explicit play overrides
    below differ. Returned config feeds ``EnvConfigAdapter``/``create_env``.
    """
    if backend not in ("mujoco", "mjwarp"):
        raise ValueError("Pen playback supports mujoco or mjwarp")
    register_resolvers()
    package = Path(__file__).parents[2]
    with initialize_config_dir(
        config_dir=str(package / "conf/supermimic"), version_base="1.3"
    ):
        cfg = compose(
            "config_pen_spin_supermimic",
            overrides=[
                f"task=dex_hand2_pen_spin/{backend}",
                f"training.sim_backend={backend}",
                "algo.num_envs=1",
                f"algo.seed={int(seed)}",
            ],
        )
    cfg.training.collector_tensor_device = "cpu" if backend == "mujoco" else "cuda:0"
    with open_dict(cfg.env):
        cfg.env.collector_tensor_device = cfg.training.collector_tensor_device
    cfg.env.seed = int(seed)
    cfg.env.auto_reset = False
    motion = cfg.env.commands.motion
    data = Path(__file__).parent / "data"
    motion.recipe_path = str(Path(recipe_path).expanduser().resolve()) if recipe_path else str(
        data / "recipe/03_continuous_multi_axis.json"
    )
    motion.clips_root = str(Path(clips_root).expanduser().resolve()) if clips_root else str(
        data / "rawdata/clips"
    )
    motion.synthetic_reference = False
    motion.trajectory_sampling_mode = "uniform"
    motion.first_frame_hold_prob = 1.0
    motion.first_frame_hold_s = [1.0, 1.0]
    # UniLab's CommandManager rejects task-owned viewer callbacks. The play
    # entrypoint owns reference overlays instead of source debug_vis hooks.
    with open_dict(motion):
        motion.debug_vis = False
    for group in cfg.env.observations.values():
        group.enable_corruption = False
    policy = cfg.env.observations.policy
    policy.terms.joint_pos.params.noise_bound = 0.0
    policy.terms.joint_vel.params.noise_bound = 0.0
    policy.terms.object_pose.func = (
        "unisupermimic.tasks.pen_spin.observations.PenObjectPoseFeature"
    )
    policy.terms.object_pose.params = {}
    reset = OmegaConf.to_container(cfg.env.events.reset_to_reference_state, resolve=False)
    reset["params"] = {}
    cfg.env.events = {
        "reset_to_reference_state": reset,
        "hardware_pen_dimensions": {
            "func": "unisupermimic.tasks.pen_spin.play_config.FixedHardwarePenDimensions",
            "mode": "startup",
        },
    }
    # The source reward helper returns early for play, before attaching the
    # training curriculum, reset noise, action freezing and metrics. Clear
    # them explicitly because this profile starts from the training recipe.
    cfg.env.curriculum = {}
    cfg.env.metrics = {}
    cfg.env.actions.joint_pos.mask_prob = 0.0
    cfg.env.actions.joint_pos.mask_duration_max = 10
    cfg.env.actions.joint_pos.mask_curriculum_term = None
    cfg.env.terminations.object_position_error.params.threshold = 0.20
    with open_dict(cfg.env.terminations):
        del cfg.env.terminations["object_orientation_error"]
    return cfg


__all__ = ["build_play_config", "FixedHardwarePenDimensions", "prepare_playback_model"]

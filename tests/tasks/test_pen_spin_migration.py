"""Contract checks for the migrated Wuji Hand 2 pen-spin task."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from unisupermimic.tasks.pen_spin.scene import materialize_scene

ROOT = Path(__file__).parents[2]
CFG = ROOT / "src/unisupermimic/conf/supermimic/task/dex_hand2_pen_spin/base.yaml"
PEN_XML = ROOT / "src/unisupermimic/assets/objects/pen250/xmls/pen_tracking.xml"


def test_pen_scene_is_repository_local_and_preserves_hand_contract():
    assert PEN_XML.is_file()
    scene = materialize_scene()
    model = mujoco.MjModel.from_xml_path(str(scene))
    assert (model.nq, model.nv, model.nu, model.nmocap) == (27, 26, 20, 1)
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)}
    assert {"object/cube", "object/A_bottom", "object/A_top"} <= names
    assert model.opt.timestep == 0.005
    object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object/cube")
    assert model.body_mass[object_body] == pytest.approx(0.03, rel=1e-6)
    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "robot/right_palm_collision")
    assert model.geom_condim[palm] == 4
    # Pen's source actor subtracts zero defaults; cube grasp defaults must
    # not leak into the separate pen scene or its mocap reset keyframe.
    from unisupermimic.tasks.pen_spin.commands import FIXED_WRIST_POSE_W

    np.testing.assert_array_equal(model.key("home").qpos[7:], np.zeros(20))
    np.testing.assert_array_equal(model.key("home").ctrl, np.zeros(20))
    np.testing.assert_allclose(model.key("home").mpos.reshape(-1, 3)[0], FIXED_WRIST_POSE_W[:3])
    np.testing.assert_allclose(model.key("home").mquat.reshape(-1, 4)[0], FIXED_WRIST_POSE_W[3:])


def test_pen_config_exposes_reference_tracking_contract():
    text = CFG.read_text(encoding="utf-8")
    for required in (
        "PenSpinCommandCfg",
        "PenSpinActionCfg",
        "PenObjectPoseFeature",
        "sim_dt: 0.005",
        "ctrl_dt: 0.02",
        "task_name: DexHand2PenSpin",
    ):
        assert required in text


def test_pen_source_asset_has_tracking_geoms():
    root = ET.parse(PEN_XML).getroot()
    assert root.find(".//body[@name='cube']") is not None


def test_pen_mujoco_env_reset_and_step_contract():
    from hydra import compose, initialize_config_dir
    from unilab.base import registry
    from unilab.base.config_adapter import EnvConfigAdapter
    from unilab.training import ensure_registries

    ensure_registries()
    with initialize_config_dir(config_dir=str(CFG.parents[2].resolve()), version_base="1.3"):
        cfg = compose(
            "config_ppo",
            overrides=["task=dex_hand2_pen_spin/mujoco", "algo.num_envs=1"],
        )
    override = EnvConfigAdapter(
        cfg, root_dir=ROOT, algo_name="ppo"
    ).build_task_env_cfg_override()
    env = registry.make(
        "DexHand2PenSpin",
        sim_backend="mujoco",
        env_cfg_override=override,
        num_envs=1,
    )
    try:
        observations, _ = env.reset()
        assert observations["obs"].shape == (1, 155)
        assert observations["critic"].shape == (1, 301)
        result = env.step(torch.zeros((1, 20)))
        assert torch.isfinite(result.reward).all()
        assert result.terminated.dtype == torch.bool
        assert result.truncated.dtype == torch.bool
    finally:
        env.close()

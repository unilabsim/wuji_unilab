"""Compare effective Pen executor settings with the final source task."""

from __future__ import annotations

import ast
from pathlib import Path

import mujoco
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from unilab.base import registry
from unilab.base.config_adapter import EnvConfigAdapter
from unilab.training import ensure_registries

from unisupermimic.tasks.pen_spin.play_config import build_play_config

ROOT = Path(__file__).parents[2]
SOURCE = ROOT / "references/Dex-Supermimic/src/dex_supermimic"


@pytest.mark.parametrize("play", [False, True])
def test_executor_pen_contacts_match_source_asset_after_final_task_overrides(monkeypatch, play):
    """Check the actual model pool, including startup publication in both modes."""
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/unisupermimic-test-cache")
    source_asset = SOURCE / "assets/resources/aruco_pen250/xmls/pen_tracking.xml"
    if not source_asset.is_file():
        pytest.skip("source reference is not installed")
    expected = mujoco.MjModel.from_xml_path(str(source_asset))
    source_task = SOURCE / "tasks/pen_spin/pen_spin_env_cfg.py"
    tree = ast.parse(source_task.read_text())
    sim = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Name) and node.func.id == "MujocoCfg")
    source_options = {item.arg: ast.literal_eval(item.value) for item in sim.keywords}
    env_cfg = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Name) and node.func.id == "ManagerBasedRlEnvCfg")
    source_substeps = ast.literal_eval(next(item.value for item in env_cfg.keywords
                                           if item.arg == "decimation"))
    constants = SOURCE / "tasks/pen_spin/config/wuji_hand2/hand2_constants.py"
    tree = ast.parse(constants.read_text())
    geom_cfg = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id == "GeomCfg")
    source_hand_priority = ast.literal_eval(next(item.value for item in geom_cfg.keywords
                                                if item.arg == "priority"))
    source_impratio = ast.literal_eval(next(node.value for node in tree.body
                                            if isinstance(node, ast.Assign)
                                            and any(isinstance(target, ast.Name)
                                                    and target.id == "HAND2_MUJOCO_IMPRATIO"
                                                    for target in node.targets)))
    ensure_registries()
    if play:
        cfg = build_play_config(backend="mujoco")
    else:
        with initialize_config_dir(
            config_dir=str(ROOT / "src/unisupermimic/conf/supermimic"), version_base="1.3"
        ):
            cfg = compose("config_pen_spin_supermimic", overrides=[
                "task=dex_hand2_pen_spin/mujoco", "algo.num_envs=2",
                "training.sim_backend=mujoco", "training.collector_tensor_device=cpu",
            ])
    override = EnvConfigAdapter(cfg, root_dir=ROOT, algo_name="supermimic").build_task_env_cfg_override()
    env = registry.make("DexHand2PenSpin", sim_backend="mujoco", env_cfg_override=override, num_envs=2)
    try:
        env.reset()
        actual, pool = env.backend.model, env.backend._pool
        for field in ("timestep", "iterations", "ls_iterations"):
            assert getattr(actual.opt, field) == source_options[field]
        assert actual.opt.integrator == getattr(mujoco.mjtIntegrator, "mjINT_" + source_options["integrator"].upper())
        assert actual.opt.impratio == source_impratio
        assert env.step_dt == pytest.approx(source_options["timestep"] * source_substeps)
        assert env.cfg.sim_substeps == source_substeps
        for name in ("cube", "A_bottom", "A_top"):
            source_id, actual_id = expected.geom(name).id, actual.geom(f"object/{name}").id
            for field in ("geom_friction", "geom_solref", "geom_solimp"):
                value = getattr(expected, field)[source_id]
                np.testing.assert_allclose(getattr(actual, field)[actual_id], value, rtol=1e-6)
                published = pool.expand(field)[:, actual_id]
                np.testing.assert_allclose(published, np.tile(value, (env.num_envs, 1)), rtol=1e-6)
            for field in ("geom_priority", "geom_condim"):
                assert getattr(actual, field)[actual_id] == getattr(expected, field)[source_id]
            palm_id = actual.geom("robot/right_palm_collision").id
            assert actual.geom_priority[palm_id] == source_hand_priority
            # The source gives hand contacts higher priority than the pen.
            # Their parameters govern hand-pen contact in both profiles, so
            # correcting pen defaults primarily changes other contact pairs.
            assert actual.geom_priority[palm_id] > actual.geom_priority[actual_id]
    finally:
        env.close()

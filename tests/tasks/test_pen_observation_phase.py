"""Actor observations use final poses even across backend read phases."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
from unilab.base.config_adapter import EnvConfigAdapter, create_env
from unilab.training import ensure_registries

from unisupermimic.tasks.pen_spin.commands import LINK_NAMES
from unisupermimic.tasks.pen_spin.play_config import build_play_config

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("backend", [
    "mujoco", pytest.param("mjwarp", marks=[pytest.mark.gpu, pytest.mark.slow]),
])
def test_clean_actor_pose_and_critic_links_match_final_qpos(backend, monkeypatch):
    """Compare returned observations, before other reads, with independent FK.

    Source mjlab performs forward before policy observations. UniSim instead
    refreshes tracked body kinematics during its read phase; no additional
    forward should be inserted solely because the call sites differ.
    """
    if backend == "mjwarp" and not torch.cuda.is_available():
        pytest.skip("CUDA required for the actual MJWarp observation phase")
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/unisupermimic-test-cache")
    ensure_registries()
    cfg = build_play_config(backend=backend, seed=42)
    if not Path(cfg.env.commands.motion.clips_root, "03_continuous_multi_axis/index.json").is_file():
        pytest.skip("upstream pen clip catalog is not installed")
    override = EnvConfigAdapter(cfg, root_dir=ROOT, algo_name="supermimic").build_task_env_cfg_override()
    env = create_env(cfg, num_envs=1, env_cfg_override=override)
    try:
        env.reset()
        model = env._pen_physics_model
        oracle = mujoco.MjData(model)
        palm = model.body("robot/right_palm_link").id
        pen = model.body("object/cube").id
        link_ids = [model.body(name).id for name in LINK_NAMES]
        motion = env.command_manager.get_term("motion")
        # The axis point order is part of the source Actor contract.
        axis_points = motion.axis_points.detach().cpu().numpy().copy()
        start = None
        max_motion = 0.0
        for step in range(8):
            action = torch.linspace(-0.7, 0.7, 20, device=env.device)[None]
            result = env.step(action if step % 2 == 0 else -action)
            # Preserve the actual returned inputs before requesting any new
            # backend views that might refresh derived state themselves.
            actor_points = result.obs["obs"][:, 129:135].detach().cpu().numpy().copy()
            critic_links = result.obs["critic"][:, 155:218].detach().cpu().numpy().copy()
            physical = env.backend.get_state_views(("qpos", "qvel"), device=env.device)
            oracle.qpos[:] = physical["qpos"][0].detach().cpu().numpy()
            oracle.qvel[:] = physical["qvel"][0].detach().cpu().numpy()
            oracle.mocap_pos[:] = model.body_pos[palm]
            oracle.mocap_quat[:] = model.body_quat[palm]
            mujoco.mj_forward(model, oracle)
            palm_rotation = oracle.xmat[palm].reshape(3, 3)
            pen_rotation = oracle.xmat[pen].reshape(3, 3)
            points_world = oracle.xpos[pen] + axis_points @ pen_rotation.T
            points_local = (points_world - oracle.xpos[palm]) @ palm_rotation
            expected_points = (0.15 * np.tanh(points_local / 0.15)).reshape(1, 6)
            expected_links = ((oracle.xpos[link_ids] - oracle.xpos[palm]) @ palm_rotation).reshape(1, 63)
            np.testing.assert_allclose(actor_points, expected_points, rtol=1e-5, atol=2e-6)
            np.testing.assert_allclose(critic_links, expected_links, rtol=1e-5, atol=2e-6)
            if start is None:
                start = oracle.xpos[pen].copy()
            max_motion = max(max_motion, float(np.linalg.norm(oracle.xpos[pen] - start)))
        assert max_motion > 1e-4
    finally:
        env.close()

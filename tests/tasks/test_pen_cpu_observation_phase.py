"""Check final CPU poses reach the camera without extra camera ticks."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict
from unilab.base.config_adapter import EnvConfigAdapter, create_env
from unilab.training import ensure_registries

from unisupermimic.tasks.pen_spin.reset_bridge import bind_cpu_observation_phase

ROOT = Path(__file__).parents[2]


def test_cpu_step_camera_raw_is_final_pose_and_partial_reset_keeps_survivor(monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/unisupermimic-test-cache")
    ensure_registries()
    with initialize_config_dir(
        config_dir=str(ROOT / "src/unisupermimic/conf/supermimic"), version_base="1.3"
    ):
        cfg = compose("config_pen_spin_supermimic", overrides=[
            "task=dex_hand2_pen_spin/mujoco", "algo.num_envs=2",
            "training.sim_backend=mujoco", "training.collector_tensor_device=cpu",
            "env.auto_reset=false",
        ])
    if not Path(cfg.env.commands.motion.clips_root, "03_continuous_multi_axis/index.json").is_file():
        pytest.skip("upstream pen clip catalog is not installed")
    params = cfg.env.observations.policy.terms.object_pose.params
    with open_dict(params):
        for name in ("bias_position_m", "bias_orientation_rad", "noise_position_m", "noise_orientation_rad"):
            params[name] = [0.0, 0.0]
        params.dropout_enter_prob = 0.0
    override = EnvConfigAdapter(cfg, root_dir=ROOT, algo_name="supermimic").build_task_env_cfg_override()
    env = create_env(cfg, num_envs=2, env_cfg_override=override)
    try:
        env.reset()
        dispatch = env._compute_observations
        bind_cpu_observation_phase(env)
        assert env._compute_observations is dispatch
        camera = env.observation_manager.get_term_cfg("policy", "object_pose").func
        motion = env.command_manager.get_term("motion")
        model = env._pen_physics_model
        oracle = mujoco.MjData(model)
        palm, pen = model.body("robot/right_palm_link").id, model.body("object/cube").id
        axis_points = motion.axis_points.cpu().numpy()
        for step in range(4):
            previous_raw = camera.memory_state()["camera_raw"].clone()
            previous_length = camera._length.clone()
            action = torch.linspace(-.7, .7, 20).expand(2, -1).contiguous()
            result = env.step(action if step % 2 == 0 else -action)
            raw = camera.memory_state()["camera_raw"].clone()
            torch.testing.assert_close(result.obs["obs"][:, 129:135], previous_raw)
            torch.testing.assert_close(camera._length, previous_length + 1)
            physical = env.backend.get_state_views(("qpos", "qvel"), device="cpu")
            for row in range(2):
                oracle.qpos[:] = physical["qpos"][row].numpy()
                oracle.qvel[:] = physical["qvel"][row].numpy()
                oracle.mocap_pos[:] = model.body_pos[palm]
                oracle.mocap_quat[:] = model.body_quat[palm]
                mujoco.mj_forward(model, oracle)
                points = oracle.xpos[pen] + axis_points @ oracle.xmat[pen].reshape(3, 3).T
                local = (points - oracle.xpos[palm]) @ oracle.xmat[palm].reshape(3, 3)
                expected = (.15 * np.tanh(local / .15)).reshape(6)
                np.testing.assert_allclose(raw[row].numpy(), expected, rtol=1e-5, atol=2e-6)

        survivor_physics = {name: value[1].clone() for name, value in physical.items()}
        survivor_history = camera._history[:, 1].clone()
        survivor_memory = {name: value[1].clone() for name, value in camera.memory_state().items()}
        survivor_length = camera._length[1].clone()
        env.reset(env_ids=torch.tensor([0]))
        after_physics = env.backend.get_state_views(("qpos", "qvel"), device="cpu")
        for name, before in survivor_physics.items():
            torch.testing.assert_close(after_physics[name][1], before, rtol=0, atol=0)
        torch.testing.assert_close(camera._history[:, 1], survivor_history, rtol=0, atol=0)
        torch.testing.assert_close(camera._length[1], survivor_length, rtol=0, atol=0)
        for name, before in survivor_memory.items():
            torch.testing.assert_close(camera.memory_state()[name][1], before, rtol=0, atol=0)
    finally:
        env.close()

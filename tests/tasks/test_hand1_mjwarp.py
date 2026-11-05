"""Hand1 reset and solver contracts on the real CUDA MJWarp executor."""

import mujoco
import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.slow, pytest.mark.gpu]
PALM_NAME = "robot/right_palm_link"


@pytest.fixture(scope="module")
def native_env():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real MJWarp Hand1 contracts")
    wp = pytest.importorskip("warp")
    from unilab.base import registry
    from unilab.training import ensure_registries

    from unisupermimic.tasks.dex_reorient.hand1.factory import make_env_cfg
    from unisupermimic.tasks.dex_reorient.hand1.registry import TASK_NAME

    wp.init()
    wp.set_device("cuda:0")
    ensure_registries()
    cfg = make_env_cfg(backend_type="mjwarp", control_hz=50, seed=42, play=False)
    cfg["events"]["reset"]["params"]["wrist_pitch"] = (-0.2, -0.2)
    env = registry.make(TASK_NAME, sim_backend="mjwarp", env_cfg_override=cfg, num_envs=4)
    try:
        assert torch.device(env.device).type == "cuda"
        yield env
    finally:
        env.close()


def _mocap(env):
    backend = env.backend
    return {
        name: backend._warp.to_torch(getattr(backend._device_data, name)).clone()
        for name in ("mocap_pos", "mocap_quat")
    }


def _expected_wrist(env, pitch):
    """Compose the source default wrist and Y pitch with MuJoCo's oracle."""
    model = env.backend._cpu_model
    body_id = model.body(PALM_NAME).id
    rotation = np.array([np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0])
    quaternion = np.empty(4)
    mujoco.mju_mulQuat(quaternion, model.body_quat[body_id], rotation)
    return (
        torch.tensor(model.body_pos[body_id].copy(), device=env.device, dtype=torch.float32),
        torch.tensor(quaternion, device=env.device, dtype=torch.float32),
    )


def _assert_wrist_read_plane(env, mocap):
    palm = env.scene.read_plan.body_tensor_view("robot", (PALM_NAME,))
    assert palm.pos_w.is_cuda and palm.quat_w.is_cuda
    torch.testing.assert_close(palm.pos_w, mocap["mocap_pos"], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(palm.quat_w, mocap["mocap_quat"], rtol=1e-6, atol=1e-6)


def _assert_policy(observations):
    policy = observations["obs"]
    assert policy.shape == (4, 207) and policy.is_cuda
    assert torch.isfinite(policy).all()


def test_source_solver_startup_parameters_and_fixed_pitch_reach_cuda(native_env):
    env = native_env
    env.event_manager.get_term_cfg("reset").params["wrist_pitch"] = (-0.2, -0.2)
    observations, _ = env.reset()
    _assert_policy(observations)
    backend = env.backend
    assert backend._cpu_model.opt.iterations == 10
    assert backend._cpu_model.opt.ls_iterations == 20
    assert backend._device_model.opt.iterations == 10
    assert backend._device_model.opt.ls_iterations == 20
    assert backend._cuda_graph_enabled

    # Exercise the full training profile: startup DR must reach device model
    # tables, rather than being a host-only reset or a play-mode shortcut.
    randomization = env.event_manager.get_term_cfg("randomize").func
    actuator_ids = [backend._cpu_model.actuator(f"{name}_actuator").id
                    for name in env.scene["robot"].joint_names]
    gains = backend._warp.to_torch(backend._device_model.actuator_gainprm)
    assert gains.is_cuda and gains.shape[0] == env.num_envs
    torch.testing.assert_close(gains[:, actuator_ids, 0], randomization.samples["kp"])
    torch.testing.assert_close(env.scene["robot"].data.encoder_bias_tensor,
                               randomization.samples["encoder"])

    position, quaternion = _expected_wrist(env, -0.2)
    mocap = _mocap(env)
    assert mocap["mocap_quat"].is_cuda and mocap["mocap_quat"].shape == (4, 1, 4)
    torch.testing.assert_close(mocap["mocap_pos"][:, 0], position.expand(4, -1))
    torch.testing.assert_close(mocap["mocap_quat"][:, 0], quaternion.expand(4, -1))
    _assert_wrist_read_plane(env, mocap)


def test_selected_wrist_reset_preserves_survivors_and_persists_next_step(native_env):
    env = native_env
    reset_cfg = env.event_manager.get_term_cfg("reset")
    reset_cfg.params["wrist_pitch"] = (-0.2, -0.2)
    env.reset()
    result = env.step(torch.zeros(4, 20, device=env.device))
    _assert_policy(result.obs)
    assert torch.isfinite(result.reward).all()
    physical = {name: value.clone() for name, value in env.backend.get_state_views(
        ("qpos", "qvel"), device=env.device
    ).items()}
    mocap_before = _mocap(env)
    gains_before = env.backend._warp.to_torch(env.backend._device_model.actuator_gainprm).clone()

    reset_cfg.params["wrist_pitch"] = (0.1, 0.1)
    observations, _ = env.reset(env_ids=torch.tensor([0], device=env.device))
    _assert_policy(observations)
    after = env.backend.get_state_views(("qpos", "qvel"), device=env.device)
    for name, value in physical.items():
        torch.testing.assert_close(after[name][1:], value[1:], rtol=0, atol=0)
    mocap_after = _mocap(env)
    for name, value in mocap_before.items():
        torch.testing.assert_close(mocap_after[name][1:], value[1:], rtol=0, atol=0)
    position, quaternion = _expected_wrist(env, 0.1)
    torch.testing.assert_close(mocap_after["mocap_pos"][0, 0], position)
    torch.testing.assert_close(mocap_after["mocap_quat"][0, 0], quaternion)
    assert not torch.allclose(mocap_before["mocap_quat"][0], mocap_after["mocap_quat"][0])
    _assert_wrist_read_plane(env, mocap_after)
    torch.testing.assert_close(
        env.backend._warp.to_torch(env.backend._device_model.actuator_gainprm),
        gains_before, rtol=0, atol=0,
    )

    # Normal control stepping may advance qpos/qvel, but must not overwrite
    # the selected wrist or upload default wrist poses over surviving rows.
    result = env.step(torch.zeros(4, 20, device=env.device))
    _assert_policy(result.obs)
    assert torch.isfinite(result.reward).all()
    for name, value in _mocap(env).items():
        torch.testing.assert_close(value, mocap_after[name], rtol=0, atol=0)
    _assert_wrist_read_plane(env, mocap_after)

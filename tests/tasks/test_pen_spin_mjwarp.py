"""Production Pen Spin contracts on an actual CUDA MJWarp executor."""

from pathlib import Path

import numpy as np
import pytest
import torch

from unisupermimic.rl.algos.dex_supermimic.env_bridge import SourceEnvBridge
from unisupermimic.rl.algos.dex_supermimic.pen import PenObservations, flatten_pen_memory
from unisupermimic.rl.algos.dex_supermimic.sim import PenSimReader
from unisupermimic.tasks.pen_spin.randomization import MODEL_FIELDS

pytestmark = [pytest.mark.slow, pytest.mark.gpu]
ROOT = Path(__file__).parents[2]


@pytest.fixture(scope="module")
def native_env():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real MJWarp Pen Spin contracts")
    wp = pytest.importorskip("warp")
    from hydra import compose, initialize_config_dir
    from unilab.base import registry
    from unilab.base.config_adapter import EnvConfigAdapter
    from unilab.training import ensure_registries

    clips = ROOT / "src/unisupermimic/tasks/pen_spin/data/rawdata/clips"
    recipe = ROOT / "src/unisupermimic/tasks/pen_spin/data/recipe/01_single_axis.json"
    if not (clips / "01_single_axis/index.json").is_file():
        pytest.skip("upstream pen clip catalog is not installed")
    wp.init()
    wp.set_device("cuda:0")
    ensure_registries()
    with initialize_config_dir(
        config_dir=str(ROOT / "src/unisupermimic/conf/supermimic"), version_base="1.3"
    ):
        cfg = compose("config_pen_spin_supermimic", overrides=[
            "algo.num_envs=4",
            f"env.commands.motion.recipe_path={recipe}",
            f"env.commands.motion.clips_root={clips}",
            "env.commands.motion.first_frame_hold_prob=0.0",
        ])
    override = EnvConfigAdapter(cfg, root_dir=ROOT, algo_name="ppo").build_task_env_cfg_override()
    env = registry.make("DexHand2PenSpin", sim_backend="mjwarp", env_cfg_override=override, num_envs=4)
    assert torch.device(env.device).type == "cuda"
    try:
        yield env
    finally:
        env.close()


def _device_field(env, name):
    return env.backend._warp.to_torch(getattr(env.backend._device_model, name)).clone()


def test_startup_model_tables_reach_cuda_and_survive_step_and_partial_reset(native_env):
    env = native_env
    before = {}
    for name in MODEL_FIELDS:
        actual = _device_field(env, name)
        assert actual.is_cuda and actual.shape[0] == env.num_envs
        torch.testing.assert_close(actual, env._pen_model_fields[name], rtol=1e-6, atol=1e-7)
        before[name] = actual
    model = env._pen_physics_model
    marker = model.geom("object/A_top").id
    actuator = model.actuator("robot/right_finger2_joint2_actuator").id
    assert before["geom_pos"][:, marker, 0].unique().numel() == env.num_envs
    assert before["actuator_forcerange"][:, actuator, 1].unique().numel() == env.num_envs
    assert env.backend._cuda_graph_enabled

    env.reset()
    result = env.step(torch.zeros(env.num_envs, 20, device=env.device))
    assert torch.isfinite(result.reward).all()
    assert all(value.is_cuda and torch.isfinite(value).all() for value in result.obs.values())
    physical = env.backend.get_state_views(("qpos", "qvel"), device=env.device)
    survivors = {name: value[1:].clone() for name, value in physical.items()}
    motion = env.command_manager.get_term("motion")
    frames = motion.frame_ids.clone()
    bias = env.scene["robot"].data.encoder_bias_tensor.clone()
    env.reset(env_ids=torch.tensor([0], device=env.device))
    after = env.backend.get_state_views(("qpos", "qvel"), device=env.device)
    for name, value in survivors.items():
        torch.testing.assert_close(after[name][1:], value, rtol=0, atol=0)
    torch.testing.assert_close(motion.frame_ids[1:], frames[1:], rtol=0, atol=0)
    torch.testing.assert_close(env.scene["robot"].data.encoder_bias_tensor, bias, rtol=0, atol=0)
    for name, value in before.items():
        torch.testing.assert_close(_device_field(env, name), value, rtol=0, atol=0)


def _raw_sensors(env, names):
    model = env._pen_physics_model
    sensor_ids = [model.sensor(name).id for name in names]
    addresses = torch.tensor(model.sensor_adr[sensor_ids], device=env.device, dtype=torch.long)
    values = env.backend._warp.to_torch(env.backend._device_data.sensordata)
    return values[:, addresses]


def test_nonzero_cuda_wrenches_reader_labels_and_partial_reset(native_env):
    native = native_env
    env = SourceEnvBridge(native)
    env.common_step_counter = 240_000
    env.reset()
    assert native.curriculum_manager._curriculum_state["disturbance_success_rate"]["value"] == 1.0
    native.set_episode_length_buf(torch.full((env.num_envs,), 60, device=env.device, dtype=torch.long))
    disturbance = native.event_manager.get_term_cfg("source_disturbances").func
    disturbance.fingertips._interval_time_left.zero_()
    reader = PenSimReader(env)
    reader.attach_capture()
    try:
        actions = torch.linspace(-0.2, 0.2, 20, device=env.device).expand(env.num_envs, -1).contiguous()
        for _ in range(2):
            observations, reward, terminated, truncated, _ = env.step(actions)
            assert observations["policy"].shape == (env.num_envs, 155)
            assert observations["policy"].is_cuda
            assert torch.isfinite(reward).all()
            assert terminated.dtype == truncated.dtype == torch.bool
            assert reader.state().shape == (env.num_envs, 58)
            captured = reader.reward_snapshot
            torch.testing.assert_close(reward, sum(captured["terms"].values()))

        selected = native._pen_external_wrench[:, disturbance._body_columns]
        assert selected[:, 0, :3].abs().sum() > 0
        assert selected[:, 1:, :3].abs().sum() > 0
        assert selected[..., 3:].abs().sum() > 0
        # The single composed interval plan must reach both backend staging
        # channels. The second step above consumes the preceding physical plan.
        assert native.backend._xfrc_pending
        np.testing.assert_allclose(
            native.backend._xfrc_staging,
            native._pen_external_wrench.detach().cpu().numpy(), rtol=0, atol=0,
        )
        torch.testing.assert_close(
            reader.external_wrench(),
            native._pen_external_wrench[:, reader._wrench_body_ids].flatten(1),
        )
        torch.testing.assert_close(
            reader.torque(), _raw_sensors(native, [f"actuator_force_{i}" for i in range(20)])
        )
        assert reader.torque().abs().sum() > 0
        model = native._pen_physics_model
        contact_names = {
            model.geom(int(model.sensor_objid[model.sensor(f"finger_collision_{i}").id])).name:
                f"finger_collision_{i}"
            for i in range(25)
        }
        middle = [contact_names[f"robot/right_finger{f}_link{j}_col"] for f in range(1, 6) for j in (2, 3)]
        distal = [contact_names[f"robot/right_finger{f}_link4_col"] for f in range(1, 6)]
        tips = [contact_names[f"robot/right_finger{f}_tip_sensor_col"] for f in range(1, 6)]
        expected_contact = torch.cat((
            _raw_sensors(native, middle) > 0,
            (_raw_sensors(native, distal) > 0) | (_raw_sensors(native, tips) > 0),
        ), -1).float()
        torch.testing.assert_close(reader.contact(), expected_contact)

        memory = reader.memory()
        assert flatten_pen_memory(memory).shape == (env.num_envs, 230)
        observation = PenObservations(reader)
        auxiliary = observation.auxiliary()
        context = observation.context(reader.state(), memory, auxiliary, reader.xi)
        assert auxiliary.shape == (env.num_envs, 195)
        assert context.shape == (env.num_envs, 368)
        assert observation.critic(context).shape == (env.num_envs, 301)
        assert reader.context().shape == (env.num_envs, 46)
        assert (reader.context()[:, :4].std(0) > 0).all()
        acts = reader.actuator_ids.tolist()
        torch.testing.assert_close(memory["limits"], _device_field(native, "actuator_forcerange")[:, acts, 1])
        torch.testing.assert_close(memory["encoder_bias"], native.scene["robot"].data.encoder_bias_tensor)

        physical = native.backend.get_state_views(("qpos", "qvel"), device=env.device)
        physical = {name: value.clone() for name, value in physical.items()}
        previous_wrench = native._pen_external_wrench.clone()
        frames = reader.motion.frame_ids.clone()
        env.reset(env_ids=torch.tensor([0], device=env.device))
        after = native.backend.get_state_views(("qpos", "qvel"), device=env.device)
        for name, value in physical.items():
            torch.testing.assert_close(after[name][1:], value[1:], rtol=0, atol=0)
        torch.testing.assert_close(reader.motion.frame_ids[1:], frames[1:], rtol=0, atol=0)
        assert not native._pen_external_wrench[0].any()
        torch.testing.assert_close(native._pen_external_wrench[1:], previous_wrench[1:], rtol=0, atol=0)
        after_memory = reader.memory()
        for name, value in memory.items():
            torch.testing.assert_close(after_memory[name][1:], value[1:], rtol=0, atol=0)
    finally:
        reader.detach_capture()

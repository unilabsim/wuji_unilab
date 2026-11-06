"""Source-play settings and their actual UniLab physics/observation effects."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from unilab.base.config_adapter import EnvConfigAdapter, create_env

from unisupermimic.rl.algos.dex_supermimic.env_bridge import SourceEnvBridge
from unisupermimic.tasks.pen_spin.observations import PenObjectPoseFeature, object_axis_points
from unisupermimic.tasks.pen_spin.play_config import build_play_config, prepare_playback_model

ROOT = Path(__file__).parents[2]
SOURCE = ROOT / "references/Dex-Supermimic/src/dex_supermimic/tasks/pen_spin"


def _source_literal_assignments(function_name):
    path = SOURCE / "config/wuji_hand2/env_cfgs.py"
    if not path.is_file():
        pytest.skip("source reference is not installed")
    module = ast.parse(path.read_text())
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                    and node.name == function_name)
    result = {}
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            continue
        for target in node.targets:
            result[ast.unparse(target)] = value
    return result


def test_play_profile_preserves_reference_play_overrides_without_training_surface():
    cfg = build_play_config()
    source = _source_literal_assignments("wuji_hand2_pen_spin_env_cfg")
    motion = cfg.env.commands.motion
    for field in ("trajectory_sampling_mode", "first_frame_hold_prob", "first_frame_hold_s"):
        actual = motion[field]
        expected = source[f"motion.{field}"]
        assert tuple(actual) == tuple(expected) if isinstance(expected, tuple) else actual == expected
    assert cfg.env.terminations.object_position_error.params.threshold == source[
        "cfg.terminations['object_position_error'].params['threshold']"
    ]
    assert "object_orientation_error" not in cfg.env.terminations
    assert set(cfg.env.events) == {"reset_to_reference_state", "hardware_pen_dimensions"}
    assert all(not group.enable_corruption for group in cfg.env.observations.values())
    assert cfg.env.observations.policy.terms.object_pose.func.endswith(".PenObjectPoseFeature")
    assert cfg.env.observations.policy.terms.joint_pos.params.noise_bound == 0
    assert cfg.env.observations.policy.terms.joint_vel.params.noise_bound == 0
    # Literal assignments alone hide the helper's early return. Verify that
    # source play exits BEFORE the training curriculum and action overrides.
    module = ast.parse((SOURCE / "config/wuji_hand2/env_cfgs.py").read_text())
    helper = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_configure_hand2_reward_and_curriculum_surface")
    guard = next(index for index, node in enumerate(helper.body)
                 if isinstance(node, ast.If) and ast.unparse(node.test) == "play"
                 and any(isinstance(item, ast.Return) for item in node.body))
    training_assignments = {
        ast.unparse(target): index
        for index, node in enumerate(helper.body) if isinstance(node, ast.Assign)
        for target in node.targets
    }
    assert guard < training_assignments["cfg.curriculum"]
    assert guard < training_assignments["act.mask_prob"]
    assert cfg.env.actions.joint_pos.mask_prob == 0
    assert cfg.env.actions.joint_pos.mask_curriculum_term is None
    assert not cfg.env.curriculum and not cfg.env.metrics
    assert not cfg.env.events.reset_to_reference_state.params
    assert cfg.env.collector_tensor_device == "cpu"


@pytest.fixture
def play_env():
    cfg = build_play_config(seed=11)
    if not Path(cfg.env.commands.motion.clips_root, "03_continuous_multi_axis/index.json").is_file():
        pytest.skip("upstream pen clip catalog is not installed")
    override = EnvConfigAdapter(cfg, root_dir=ROOT, algo_name="supermimic").build_task_env_cfg_override()
    # Two rows exercise selected reset isolation while the public viewer uses1.
    env = SourceEnvBridge(create_env(cfg, num_envs=2, env_cfg_override=override))
    try:
        env.reset()
        yield env
    finally:
        env.close()


def test_clean_axis_observation_and_one_second_first_frame_hold(play_env):
    env = play_env
    native = env.native_env
    motion = env.command_manager.get_term("motion")
    assert motion.time_steps.tolist() == [-49, -49]
    camera = env.observation_manager.get_term_cfg("policy", "object_pose").func
    assert type(camera) is PenObjectPoseFeature
    obs, _ = env.reset()
    assert obs["policy"].shape == (2, 155)
    torch.testing.assert_close(obs["policy"][:, 129:135], object_axis_points(native, motion))
    # Re-reading clean observations does not sample camera latency/bias/noise.
    first = env.observation_manager.compute(update_history=False)["policy"]
    second = env.observation_manager.compute(update_history=False)["policy"]
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    _, reward, *_ = env.step(torch.zeros(2, 20))
    assert torch.isfinite(reward).all()
    assert motion.time_steps.tolist() == [-48, -48]
    assert not env.action_manager.get_term("joint_pos")._mask_active.any()


def test_play_stays_clean_above_training_curriculum_warmup(play_env):
    env = play_env
    env.common_step_counter = 240_001
    observations, _ = env.reset()
    motion = env.command_manager.get_term("motion")
    assert not env.curriculum_manager._curriculum_state
    action = env.action_manager.get_term("joint_pos")
    assert action.cfg.mask_prob == 0 and action.cfg.mask_curriculum_term is None
    # At full training strength the old profile applied an axial slide and
    # roll on reset. Source play starts at the exact commanded object pose.
    from unisupermimic.tasks.pen_spin.observations import _runtime_pose

    position, quaternion = _runtime_pose(env.native_env, motion)
    torch.testing.assert_close(position, motion.object_pose[:, :3], rtol=1e-5, atol=1e-6)
    cosine = (quaternion * motion.object_pose[:, 3:]).sum(-1).abs()
    torch.testing.assert_close(cosine, torch.ones_like(cosine), rtol=1e-5, atol=1e-6)
    for _ in range(4):
        observations, *_ = env.step(torch.zeros(2, 20))
        assert not action._mask_active.any()
        torch.testing.assert_close(
            observations["policy"][:, 129:135], object_axis_points(env.native_env, motion)
        )
    assert not env.curriculum_manager._curriculum_state


def test_fixed_hardware_geometry_inertia_and_partial_reset_are_real(play_env):
    env = play_env
    native = env.native_env
    model = native._pen_physics_model
    fields = native._pen_model_fields
    body, shaft, bottom, top = (
        model.body("object/cube").id,
        model.geom("object/cube").id,
        model.geom("object/A_bottom").id,
        model.geom("object/A_top").id,
    )
    pool = native.backend._pool
    values = {
        name: np.array(pool.expand(name), copy=True).reshape(tuple(value.shape))
        for name, value in fields.items()
    }
    np.testing.assert_allclose(values["body_mass"][:, body], .030, rtol=1e-6)
    np.testing.assert_allclose(values["geom_size"][:, shaft, 0], .0149 / 2, rtol=1e-6)
    np.testing.assert_allclose(values["geom_size"][:, shaft, 1], (.250 - .072) / 2, rtol=1e-6)
    np.testing.assert_allclose(values["geom_pos"][:, bottom, 0], -(.250 / 2 - .018), rtol=1e-6)
    np.testing.assert_allclose(values["geom_pos"][:, top, 0], .250 / 2 - .018, rtol=1e-6)
    # Hardware dimensions rebuild uniform-volume inertia; the transverse
    # moments agree and exceed the axial moment for this long thin pen.
    inertia = values["body_inertia"][:, body]
    np.testing.assert_allclose(inertia[:, 1], inertia[:, 2])
    assert (inertia[:, 1] > inertia[:, 0]).all()
    assert not native._pen_external_wrench.any()
    assert not env.scene["robot"].data.encoder_bias_tensor.any()
    env.step(torch.zeros(2, 20))
    survivor = native.backend.get_state_views("qpos", device="cpu")["qpos"][1].clone()
    env.reset(env_ids=torch.tensor([0]))
    torch.testing.assert_close(native.backend.get_state_views("qpos", device="cpu")["qpos"][1], survivor)
    assert env.command_manager.get_term("motion").time_steps.tolist() == [-49, -48]
    for name, before in values.items():
        np.testing.assert_array_equal(pool.expand(name).reshape(before.shape), before)


def test_play_backend_validation_and_device_selection():
    assert build_play_config(backend="mjwarp").env.collector_tensor_device == "cuda:0"
    with pytest.raises(ValueError, match="mujoco or mjwarp"):
        build_play_config(backend="motrix")


def test_visual_copy_uses_actual_startup_geometry_without_mutating_oracle(play_env):
    env = play_env
    oracle = env.get_playback_model(0)
    before_size, before_pos = oracle.geom_size.copy(), oracle.geom_pos.copy()
    before_mass, before_inertia = oracle.body_mass.copy(), oracle.body_inertia.copy()
    fields = {name: value.clone() for name, value in env._pen_model_fields.items()}
    visual = prepare_playback_model(env)
    assert visual is not oracle
    assert visual.ngeom > env._pen_physics_model.ngeom
    shaft, body = visual.geom("object/cube").id, visual.body("object/cube").id
    np.testing.assert_allclose(visual.geom_size[shaft, 0], .0149 / 2, rtol=1e-6)
    np.testing.assert_allclose(visual.geom_pos[visual.geom("object/A_top").id, 0], .107, rtol=1e-6)
    np.testing.assert_allclose(visual.geom_rbound[shaft], np.linalg.norm(visual.geom_size[shaft, :2]))
    physics_body = env._pen_physics_model.body("object/cube").id
    np.testing.assert_allclose(visual.body_inertia[body], fields["body_inertia"][0, physics_body])
    np.testing.assert_array_equal(oracle.geom_size, before_size)
    np.testing.assert_array_equal(oracle.geom_pos, before_pos)
    np.testing.assert_array_equal(oracle.body_mass, before_mass)
    np.testing.assert_array_equal(oracle.body_inertia, before_inertia)
    for name, before in fields.items():
        torch.testing.assert_close(env._pen_model_fields[name], before, rtol=0, atol=0)


def test_play_profile_does_not_leak_into_training_defaults():
    build_play_config()
    with initialize_config_dir(
        config_dir=str(ROOT / "src/unisupermimic/conf/supermimic"), version_base="1.3"
    ):
        training = compose("config_pen_spin_supermimic")
    motion = training.env.commands.motion
    assert motion.trajectory_sampling_mode == "adaptive"
    assert motion.first_frame_hold_prob == .15
    assert training.env.observations.policy.enable_corruption
    assert training.env.observations.policy.terms.object_pose.func.endswith(".PenObjectPoseCameraModel")
    assert set(training.env.events) == {
        "source_physics", "reset_to_reference_state", "source_disturbances"
    }
    # Mutating a returned play profile is confined to that config instance.
    snapshot = OmegaConf.to_container(training, resolve=False)
    play = build_play_config()
    play.env.commands.motion.first_frame_hold_s = [2., 2.]
    assert OmegaConf.to_container(training, resolve=False) == snapshot

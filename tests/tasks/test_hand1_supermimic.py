"""Hand 1 source contracts across the UniLab simulation/training boundary."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from unisupermimic.rl.algos.dex_supermimic.env_bridge import SourceEnvBridge
from unisupermimic.rl.algos.dex_supermimic.hand1_reader import Hand1Observations, SimReader
from unisupermimic.rl.algos.dex_supermimic.rotations import quat_to_mat
from unisupermimic.rl.algos.dex_supermimic.state import flatten_memory, unpack
from unisupermimic.tasks.dex_reorient.hand1.events import FixedRandomization
from unisupermimic.tasks.dex_reorient.hand1.factory import make_env_cfg, make_hand1_env
from unisupermimic.tasks.dex_reorient.hand1.rewards import _joint_acceleration, metric
from unisupermimic.tasks.dex_reorient.hand1.scene import (
    _HOME_JOINT_POS,
    _TAG_POS,
    _TAG_QUAT,
    materialize_scene,
)


@pytest.fixture
def hand1(monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/unisupermimic-test-cache")
    env = SourceEnvBridge(make_hand1_env(num_envs=2, backend_type="mujoco"))
    env.reset()
    reader = SimReader(env)
    try:
        yield env, reader
    finally:
        reader.detach_capture()
        env.close()


def test_scene_preserves_hand1_source_physics(monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/unisupermimic-test-cache")
    model = mujoco.MjModel.from_xml_path(str(materialize_scene()))
    assert (model.nq, model.nv, model.nu, model.nmocap) == (27, 26, 20, 1)
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
    assert model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_EULERDAMP
    assert model.opt.iterations == 10
    assert model.opt.ls_iterations == 20
    tag = model.body("robot/right_wrist_tag_frame")
    np.testing.assert_allclose(tag.pos, _TAG_POS)
    np.testing.assert_allclose(tag.quat, _TAG_QUAT)
    key = model.key("home")
    np.testing.assert_allclose(key.qpos[7:], _HOME_JOINT_POS)
    collisions = [model.sensor(f"finger_collision_{i}").objid[0] for i in range(16)]
    assert len(set(collisions)) == 16
    assert "robot/right_finger1_link2_softbody_col" in [model.geom(int(i)).name for i in collisions]
    # Hand 1 uses its 0.4 Nm-scale actuator, not Hand 2's measured 1.35 gain.
    assert model.actuator_gainprm[0, 0] == pytest.approx(0.40844710645048043)


@pytest.mark.parametrize("play", (False, True))
def test_executor_uses_source_task_solver_overrides(monkeypatch, play):
    """Check effective task options, not the hand asset's earlier 5/8 values."""
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/unisupermimic-test-cache")
    source = (Path(__file__).parents[2] / "references/Dex-Supermimic/src/dex_supermimic"
              / "tasks/dex_reorient/reorient_env_cfg.py")
    if not source.is_file():
        pytest.skip("source reference is not installed")
    tree = ast.parse(source.read_text())
    call = next(node for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "MujocoCfg")
    expected = {item.arg: ast.literal_eval(item.value) for item in call.keywords}
    env = make_hand1_env(num_envs=1, backend_type="mujoco", play=play)
    try:
        for field in ("iterations", "ls_iterations", "timestep"):
            assert getattr(env.backend.model.opt, field) == expected[field]
    finally:
        env.close()


def test_source_dimensions_limits_and_reward_capture(hand1):
    env, reader = hand1
    observations = Hand1Observations(reader)
    state, memory, auxiliary = reader.state(), reader.memory(), observations.auxiliary()
    assert state.shape == (2, 58)
    assert flatten_memory(memory).shape == (2, 100)
    assert reader.contact().shape == (2, 23)
    assert reader.xi.shape == (2, 46)
    torch.testing.assert_close(reader.xi[:, 4], torch.zeros(2))
    assert observations.context(state, memory, auxiliary, reader.xi).shape == (2, 211)
    dynamics = observations.dynamics(state, reader.xi, observations.disturbance(auxiliary))
    assert dynamics.shape == (2, 113)  # plus 20D executed target = WM input 133
    hard = torch.tensor(reader.model.jnt_range[1:], dtype=torch.float32)
    center, half = hard.mean(-1), (hard[:, 1] - hard[:, 0]) * 0.45
    torch.testing.assert_close(reader.mdp.lower, center - half)
    torch.testing.assert_close(reader.mdp.upper, center + half)
    reader.attach_capture()
    obs, reward, *_ = env.step(torch.zeros(2, 20, device=env.device))
    assert obs["policy"].shape == (2, 207)
    torch.testing.assert_close(reader.reward_snapshot["reward"], reward)
    summed = sum(reader.reward_snapshot["terms"].values())
    torch.testing.assert_close(summed, reward)
    assert reader.reward_snapshot["contact"].shape == (2, 23)


def test_controller_executes_encoder_corrected_target(hand1):
    env, reader = hand1
    env.scene["robot"].data.encoder_bias_tensor.fill_(0.007)
    action = env.action_manager.get_term("joint_pos")
    env.step(torch.zeros(2, 20, device=env.device))
    control = env.native_env.backend.get_state_views("ctrl", device=env.device)["ctrl"]
    torch.testing.assert_close(control, action.target - 0.007)
    torch.testing.assert_close(reader.memory()["c1"], action.target)


def test_native_hand1_clean_actor_frame_matches_source(hand1):
    env, reader = hand1
    expected = reader.mdp.observation(reader.state(), reader.memory())
    actual = env.native_env.obs_buf["critic"][:, :69]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_torque_reward_uses_source_xml_actuator_units(hand1):
    env, reader = hand1
    # Pinned mjlab EntityData has no optional actuator_effort_limit. Source
    # rewards therefore use Nm squared, independent of a joint force clamp.
    assert reader.mdp.effort_limit is None
    assert reader.command.mdp.effort_limit is None
    reader.attach_capture()
    env.step(torch.zeros(2, 20, device=env.device))
    snapshot = reader.reward_snapshot
    expected = -24.0 * snapshot["torque"].square().sum(-1) * env.step_dt
    torch.testing.assert_close(snapshot["terms"]["torque"], expected)


def test_source_startup_randomization_survives_selected_reset(hand1):
    env, reader = hand1
    assert env.event_manager.get_term_cfg("randomize").mode == "startup"
    pool = env.native_env.backend._pool
    fields = (
        "body_mass", "body_ipos", "body_inertia", "geom_size",
        "geom_friction", "geom_rbound", "geom_aabb", "actuator_gainprm",
        "actuator_biasprm", "dof_armature",
    )
    before = {name: np.array(pool.expand(name), copy=True) for name in fields}
    env.reset(env_ids=torch.tensor([1], device=env.device))
    for name in fields:
        np.testing.assert_array_equal(pool.expand(name), before[name])
    # Source size DR resizes the collision geometry as well as its broad
    # phase bounds. Resolve the executor id because visual geoms are omitted
    # by UniLab's physical model materialization.
    gid = env.native_env.backend._model.geom("object/cube").id
    size = before["geom_size"][:, gid]
    np.testing.assert_allclose(before["geom_rbound"][:, gid], np.linalg.norm(size, axis=-1))
    np.testing.assert_allclose(before["geom_aabb"].reshape(2, -1, 2, 3)[:, gid, 1], size)
    assert reader.xi.dtype == torch.float32


def test_source_fk_matches_mujoco_and_has_finite_gradients(hand1):
    _, reader = hand1
    model = reader.model
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    q = torch.tensor(_HOME_JOINT_POS, dtype=torch.float32)[None].requires_grad_(True)
    qd = torch.zeros_like(q)
    positions = reader.kin.forward(q, qd)[0]
    mujoco.mj_forward(model, data)
    palm = model.body("robot/right_palm_link").id
    expected = (data.xpos[reader.kin.body_ids] - data.xpos[palm]) @ data.xmat[palm].reshape(3, 3)
    torch.testing.assert_close(positions[0], torch.tensor(expected, dtype=torch.float32), atol=1e-6, rtol=1e-5)
    positions.square().sum().backward()
    assert torch.isfinite(q.grad).all()
    # The reader must use final generalized freejoint state, independently
    # of whether a backend refreshes frame sensors after its final substep.
    state = reader.state()
    _, _, object_position, *_ = unpack(state)
    actual = reader.env.native_env.backend.get_state_views("qpos", device=reader.device)["qpos"][:, :3]
    expected = torch.einsum("bij,bj->bi", reader.palm_rot.transpose(-1, -2), actual-reader.palm_pos)
    torch.testing.assert_close(object_position, expected)


def test_source_size_and_contact_randomization_share_per_world_scale():
    class Rng:
        def uniform(self, low, high, size):
            return torch.rand(size) * (high - low) + low

    event = object.__new__(FixedRandomization)
    event.device, event.samples, event.ratios = "cpu", {}, {}
    env = SimpleNamespace(num_envs=4, rng=Rng())
    event._sample(env, "cube_size", torch.full((1, 3), 0.027), {"cube_size": (0.85, 1.15)})
    cube = event.samples["cube_size"]
    torch.testing.assert_close(cube[..., 0], cube[..., 1])
    torch.testing.assert_close(cube[..., 1], cube[..., 2])
    event._sample(env, "hand_size", torch.ones(10, 3), {"hand_size": (0.97, 1.03)})
    sizes = event.samples["hand_size"]
    torch.testing.assert_close(sizes, sizes[:, :1, :1].expand_as(sizes))
    event._sample(env, "solref_palm", torch.ones(5, 2), {"solref": {"0": (1, 2), "1": (0.8, 1.2)}})
    contact = event.samples["solref_palm"]
    torch.testing.assert_close(contact, contact[:, :1].expand_as(contact))


def test_play_disables_source_policy_noise_and_outlier_injection():
    cfg = make_env_cfg(play=True)
    assert cfg["max_episode_seconds"] == 60.0
    policy = cfg["observations"]["policy"]
    assert not policy["enable_corruption"]
    assert policy["terms"]["position"]["params"]["injection_prob"] == 0.0
    assert policy["terms"]["orientation"]["params"]["injection_prob"] == 0.0
    # Training retains the source 2% outlier distribution.
    train_policy = make_env_cfg()["observations"]["policy"]
    assert train_policy["enable_corruption"]
    assert train_policy["terms"]["position"]["params"]["injection_prob"] == 0.02
    assert train_policy["terms"]["orientation"]["params"]["injection_prob"] == 0.02


def test_cube_height_uses_hand1_palm_normal(hand1):
    env, reader = hand1
    command = reader.command
    palm_position, palm_quaternion = command.palm_pose()
    inverse = quat_to_mat(palm_quaternion).transpose(-1, -2)
    displacement = command.cube_position() - palm_position
    local = torch.einsum("bij,bj->bi", inverse, displacement)
    actual = metric(env.native_env, "cube_height_above_palm")
    torch.testing.assert_close(actual, local[:, 0])
    assert not torch.allclose(actual, local[:, 2])


def test_joint_acceleration_metric_reads_nonzero_native_qacc(hand1):
    env, _ = hand1
    native = env.native_env
    # Reset must bind the derived CPU data before the first physics step;
    # binding mjbatch qacc only after stepping would initially return zeros.
    assert getattr(native, "_hand1_acceleration_metric", None) is not None
    env.step(torch.zeros(2, 20, device=env.device))
    qacc = native.backend._pool.bind("qacc", native.backend._np_dtype)
    # Hand1's free cube occupies DOFs 0:6; measure only the 20 hand joints.
    expected = torch.from_numpy(np.array(qacc[:, 6:], copy=True)).float().square().mean(-1).sqrt()
    assert torch.all(expected > 0.1)
    torch.testing.assert_close(metric(native, "joint_acceleration_rms"), expected)


def test_joint_acceleration_reports_missing_backend_capability():
    env = SimpleNamespace(backend=SimpleNamespace(backend_type="unsupported"))
    with pytest.raises(RuntimeError, match="requires a backend with native qacc reads"):
        _joint_acceleration(env)

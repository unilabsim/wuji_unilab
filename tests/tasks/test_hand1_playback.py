"""Hand1 actor-only playback, native state sync, and interactive controls."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from unisupermimic.rl.algos.dex_supermimic.networks import Actor
from unisupermimic.tasks.dex_reorient.hand1 import factory, play


def checkpoint(actor):
    return dict(
        format_version=3,
        config=dict(task="hand1-reorient", algorithm="world_model", control_hz=50, seed=42),
        modules=dict(actor=actor.state_dict()),
        joint_names=[f"robot/right_finger{finger}_joint{joint}"
                     for finger in range(1, 6) for joint in range(1, 5)],
        iteration=8000,
    )


def test_actor_restores_weights_and_normalization_without_writing(tmp_path):
    actor = Actor(207).eval()
    actor.norm.update(torch.randn(11, 207) + torch.linspace(-2, 2, 207))
    path = tmp_path / "model.pt"
    torch.save(checkpoint(actor), path)
    before = path.read_bytes()
    loaded, metadata = play.load_actor(path)
    observations = torch.randn(4, 207)
    torch.testing.assert_close(loaded(observations), actor(observations), rtol=0, atol=0)
    for name, value in actor.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)
    assert not loaded.training and metadata["iteration"] == 8000
    assert path.read_bytes() == before
    assert [item.name for item in tmp_path.iterdir()] == ["model.pt"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (("task", "hand2-pen-spin", "Hand1 SuperMimic"),
     ("algorithm", "ppo", "Hand1 SuperMimic"),
     ("control_hz", 30, "control frequency")),
)
def test_loader_rejects_incompatible_checkpoint_contract(tmp_path, field, value, message):
    payload = checkpoint(Actor(207))
    payload["config"][field] = value
    path = tmp_path / "model.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match=message):
        play.load_actor(path)


def test_loader_rejects_wrong_observation_layout_and_partial_weights(tmp_path):
    path = tmp_path / "model.pt"
    torch.save(checkpoint(Actor(155)), path)
    with pytest.raises(ValueError, match="207 features"):
        play.load_actor(path)
    payload = checkpoint(Actor(207))
    del payload["modules"]["actor"]["net.0.weight"]
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        play.load_actor(path)


class NativeEnv:
    """Bridge-level double exposing a goal reached on a terminal transition."""

    device, num_envs = "cpu", 1

    def __init__(self):
        self.cfg = SimpleNamespace(auto_reset=True)
        self.reset_calls = self.step_calls = 0
        self.actions = []
        self.command = SimpleNamespace(goal_count=torch.zeros(1, dtype=torch.int64))
        self.command_manager = SimpleNamespace(get_term=lambda _: self.command)

    def set_autoreset(self, value):
        self.cfg.auto_reset = value

    def reset(self, **kwargs):
        self.reset_calls += 1
        self.command.goal_count.zero_()
        return dict(policy=torch.full((1, 207), 0.5)), {}

    def step(self, action):
        self.step_calls += 1
        self.actions.append(action.clone())
        self.command.goal_count.add_(1)
        return SimpleNamespace(
            obs=dict(policy=torch.full((1, 207), 0.5)), reward=torch.ones(1),
            terminated=torch.tensor([False]), truncated=torch.tensor([True]), info={},
        )


@pytest.mark.parametrize("manual", (False, True))
def test_direct_actor_keeps_normalization_fixed_and_preserves_terminal_goal(manual):
    actor = Actor(207).eval()
    actor.norm.update(torch.randn(8, 207))
    before = {name: value.clone() for name, value in actor.state_dict().items()}
    native = NativeEnv()
    session = play.Hand1Playback(native, actor, manual_reset=manual)
    session.reset()
    expected = actor(session.obs)
    assert session.advance()
    torch.testing.assert_close(native.actions[0], expected, rtol=0, atol=0)
    assert session.goal_reaches == 1
    assert session.episodes == 1 and session.failures == 0
    assert native.reset_calls == (1 if manual else 2)
    assert session.ended == manual
    if manual:
        assert not session.advance() and native.step_calls == 1
        session.reset()
        assert not session.ended and native.reset_calls == 2
        assert session.advance()
        assert session.goal_reaches == 2
    for name, value in actor.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_playback_without_training_reader_uses_current_reward_and_termination_pose(monkeypatch):
    """Physical motion must reach task terms before the command-update phase."""
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/unisupermimic-test-cache")
    native = factory.make_hand1_env(num_envs=1, backend_type="mujoco", play=True)
    try:
        actor = Actor(207).eval()
        with torch.no_grad():
            actor.net[-1].weight.zero_()
            actor.net[-1].bias.fill_(0.25)
        session = play.Hand1Playback(native, actor)
        session.reset()
        assert session.env.training_reader is None
        command = session.command
        initial_position = command.cube_position().clone()
        observed = []

        def check_current_pose(stage):
            current = native.scene.read_plan.body_tensor_view("object", (command.cube_name,))
            cached_position = command.cube_position().clone()
            torch.testing.assert_close(cached_position, current.pos_w[:, 0], rtol=0, atol=0)
            torch.testing.assert_close(command._cube_quat, current.quat_w[:, 0], rtol=0, atol=0)
            observed.append((stage, cached_position))

        original_termination = native._compute_termination_and_reset_flags
        original_reward = native._compute_rewards

        def termination():
            check_current_pose("termination")
            return original_termination()

        def reward():
            check_current_pose("reward")
            return original_reward()

        monkeypatch.setattr(native, "_compute_termination_and_reset_flags", termination)
        monkeypatch.setattr(native, "_compute_rewards", reward)
        for _ in range(5):
            assert session.advance()
        assert [stage for stage, _ in observed] == ["termination", "reward"] * 5
        # The cube actually moved, so equal cached/current poses above cannot
        # pass merely because reset left the simulation at a stationary pose.
        assert max(float((position - initial_position).abs().max()) for _, position in observed) > 1e-4
        assert native.action_manager.action.abs().sum() > 0
    finally:
        native.close()


def test_native_mocap_and_generalized_state_are_synced_without_physics_mutation(monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/unisupermimic-test-cache")
    env = factory.make_hand1_env(num_envs=1, backend_type="mujoco", play=True)
    try:
        env.reset()
        oracle = env.get_playback_model(0)
        visual = copy.deepcopy(oracle)
        data = mujoco.MjData(visual)
        physical = np.array(env.get_physics_state_snapshot(), copy=True)
        play.sync_viewer_data(env, visual, data)
        np.testing.assert_array_equal(env.get_physics_state_snapshot(), physical)
        np.testing.assert_allclose(data.qpos, physical[0, 1:1 + visual.nq])
        np.testing.assert_allclose(data.qvel, physical[0, 1 + visual.nq:1 + visual.nq + visual.nv])
        palm = env.scene.read_plan.body_tensor_view("robot", ("robot/right_palm_link",))
        np.testing.assert_allclose(data.mocap_pos, palm.pos_w[0].cpu().numpy())
        np.testing.assert_allclose(data.mocap_quat, palm.quat_w[0].cpu().numpy())
        np.testing.assert_allclose(data.xpos[visual.body("robot/right_palm_link").id],
                                   palm.pos_w[0, 0].cpu().numpy(), atol=1e-6)
    finally:
        env.close()


@pytest.mark.parametrize("unsupported", (False, True))
def test_snapshot_without_mocap_tail_reads_actual_backend_or_command_pose(unsupported):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="robot/right_palm_link" mocap="true">'
        '<geom type="sphere" size="0.02"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    position = np.array([[0.1, 0.2, 0.3]])
    quaternion = np.array([[np.cos(0.15), 0., np.sin(0.15), 0.]])

    def getter(_):
        if unsupported:
            raise NotImplementedError("backend has no mocap playback capability")
        return position, quaternion

    command = SimpleNamespace(
        palm_name="robot/right_palm_link",
        palm_pose=lambda: (torch.tensor(position), torch.tensor(quaternion)),
    )
    env = SimpleNamespace(
        get_physics_state_snapshot=lambda: np.zeros((1, 1 + model.nq + model.nv)),
        get_playback_mocap_state=getter,
        command_manager=SimpleNamespace(get_term=lambda _: command),
    )
    play.sync_viewer_data(env, model, data)
    np.testing.assert_allclose(data.mocap_pos, position)
    np.testing.assert_allclose(data.mocap_quat, quaternion)


def test_native_viewer_keyboard_loop_preserves_checkpoint_and_training_directory(tmp_path, monkeypatch):
    """Run real Hand1 physics and overlays through a window double, no OpenGL."""
    import mujoco.viewer

    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/unisupermimic-test-cache")
    path = tmp_path / "model.pt"
    torch.save(checkpoint(Actor(207)), path)
    before = path.read_bytes()
    monkeypatch.setattr(play.time, "sleep", lambda _: None)
    windows = []

    class Window:
        def __init__(self, model, data, key_callback):
            self.user_scn = mujoco.MjvScene(model, maxgeom=512)
            self.cam = mujoco.MjvCamera()
            self.opt = mujoco.MjvOption()
            self.callback = key_callback
            self.data = data
            self.times = []
            windows.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def lock(self):
            return nullcontext()

        def is_running(self):
            assert len(self.times) < 10, "viewer loop failed to advance"
            return True

        def sync(self):
            assert self.user_scn.ngeom >= 8  # Ghost, six colored faces, lamp.
            assert self.opt.frame == mujoco.mjtFrame.mjFRAME_NONE
            self.times.append(self.data.time)
            # Start paused -> single step -> reset -> run -> pause -> step.
            controls = {1: "N", 2: "R", 3: " ", 4: " ", 5: "N"}
            if len(self.times) in controls:
                self.callback(ord(controls[len(self.times)]))

    monkeypatch.setattr(mujoco.viewer, "launch_passive", Window)
    play.main(["--checkpoint", str(path), "--backend", "mujoco", "--steps", "3", "--start-paused"])
    np.testing.assert_allclose(windows[0].times, [0.0, 0.02, 0.0, 0.02, 0.02, 0.04])
    assert path.read_bytes() == before
    assert sorted(item.name for item in tmp_path.iterdir()) == ["model.pt"]

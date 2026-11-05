"""Actor/checkpoint and episode contracts of native Pen playback."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from unisupermimic.rl.algos.dex_supermimic.networks import Actor
from unisupermimic.tasks.pen_spin.play import (
    PenPlayback,
    load_actor,
    reference_paths,
)


def checkpoint(actor):
    return dict(format_version=3, config=dict(task="hand2-pen-spin", algorithm="world_model",
                control_hz=50, seed=42), modules=dict(actor=actor.state_dict()),
                joint_names=[f"j{i}" for i in range(20)], reference_signature="hash", iteration=8000)


def test_actor_restores_normalization_and_rejects_wrong_task(tmp_path):
    actor = Actor(155).eval()
    actor.norm.update(torch.randn(9, 155) + torch.linspace(-1, 1, 155))
    payload = checkpoint(actor)
    path = tmp_path / "model.pt"
    torch.save(payload, path)
    loaded, metadata = load_actor(path)
    obs = torch.randn(1, 155)
    torch.testing.assert_close(loaded(obs), actor(obs), rtol=0, atol=0)
    assert not loaded.training and metadata["iteration"] == 8000
    payload["config"]["task"] = "hand2-reorient"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="Pen Spin SuperMimic"):
        load_actor(path)


def test_paths_use_actual_saved_hydra_recipe_without_writing(tmp_path):
    import json

    file = tmp_path / "environment_config.json"
    file.write_text(json.dumps(dict(commands=dict(motion=dict(recipe_path="actual.json", clips_root="clips")))))
    metadata = dict(config=dict(recipe_path=None, clips_root=None))
    before = file.read_bytes()
    assert reference_paths(tmp_path / "model.pt", metadata) == ("actual.json", "clips")
    assert reference_paths(tmp_path / "model.pt", metadata, "override.json") == ("override.json", "clips")
    assert file.read_bytes() == before


class NativeEnv:
    """Small bridge-level double for manual versus automatic episode reset."""

    device, num_envs = "cpu", 1

    def __init__(self):
        self.cfg = SimpleNamespace(auto_reset=True)
        self.reset_calls = self.step_calls = 0
        self.actions = []
        self.termination_manager = SimpleNamespace(get_term=lambda _: torch.tensor([True]))

    def set_autoreset(self, value):
        self.cfg.auto_reset = value

    def reset(self, **kwargs):
        self.reset_calls += 1
        return dict(obs=torch.full((1, 155), 0.5)), {}

    def step(self, action):
        self.step_calls += 1
        self.actions.append(action.clone())
        return SimpleNamespace(obs=dict(obs=torch.full((1, 155), 0.5)), reward=torch.ones(1),
                               terminated=torch.tensor([False]), truncated=torch.tensor([True]), info={})


@pytest.mark.parametrize("manual", (False, True))
def test_playback_direct_actor_freezes_norm_and_resets_only_on_done(manual):
    actor = Actor(155).eval()
    before = {key: value.clone() for key, value in actor.state_dict().items()}
    native = NativeEnv()
    session = PenPlayback(native, actor, manual_reset=manual)
    session.reset()
    expected = actor(session.obs)
    assert session.advance()
    torch.testing.assert_close(native.actions[0], expected, rtol=0, atol=0)
    assert session.completed_clips == 1 and session.failures == 0
    assert native.reset_calls == (1 if manual else 2)
    assert session.ended == manual
    if manual:
        assert not session.advance() and native.step_calls == 1
        session.reset()
        assert not session.ended and native.reset_calls == 2
    for key, value in actor.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_native_viewer_loop_uses_real_state_overlay_and_keyboard_reset(tmp_path, monkeypatch):
    """Exercise the actual viewer loop with a window double, without OpenGL."""
    from contextlib import nullcontext

    import mujoco
    import mujoco.viewer

    from unisupermimic.tasks.pen_spin import play, play_config
    from unisupermimic.tasks.pen_spin.commands import JOINT_NAMES

    payload = checkpoint(Actor(155))
    payload["joint_names"] = list(JOINT_NAMES)
    path = tmp_path / "model.pt"
    torch.save(payload, path)
    before = path.read_bytes()
    original_builder = play_config.build_play_config

    def build(**kwargs):
        cfg = original_builder(**kwargs)
        cfg.env.commands.motion.synthetic_reference = True
        cfg.env.commands.motion.recipe_path = ""
        cfg.env.commands.motion.clips_root = None
        return cfg

    monkeypatch.setattr(play_config, "build_play_config", build)
    monkeypatch.setattr(play.time, "sleep", lambda _: None)
    windows = []

    class Window:
        def __init__(self, model, data, key_callback):
            self.user_scn = mujoco.MjvScene(model, maxgeom=512)
            self.cam = mujoco.MjvCamera()
            self.callback = key_callback
            self.data = data
            self.syncs = 0
            windows.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def lock(self):
            return nullcontext()

        def is_running(self):
            assert self.syncs < 10, "viewer loop failed to advance"
            return True

        def sync(self):
            self.syncs += 1
            assert self.user_scn.ngeom > 33
            assert self.data.time >= 0
            if self.syncs == 1:
                self.callback(ord("R"))

    monkeypatch.setattr(mujoco.viewer, "launch_passive", Window)
    play.main(["--checkpoint", str(path), "--backend", "mujoco", "--steps", "3"])
    assert windows[0].syncs == 4  # Three physics steps plus the requested reset.
    assert path.read_bytes() == before
    assert sorted(item.name for item in tmp_path.iterdir()) == ["model.pt"]

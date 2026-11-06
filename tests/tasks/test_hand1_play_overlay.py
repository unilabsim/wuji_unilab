"""Real Hand 1 goal rendering contracts without opening an OpenGL window."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from unisupermimic.tasks.dex_reorient.hand1.factory import make_hand1_env
from unisupermimic.tasks.dex_reorient.hand1.play_overlay import Hand1GoalOverlay
from unisupermimic.tasks.dex_reorient.hand2.overlay import (
    GOAL_FACE_HALF_SIZE,
    GOAL_FACE_RGBA,
    GOAL_FACE_THICKNESS,
    GOAL_LAMP_AMBER_RGBA,
    GOAL_LAMP_GREEN_RGBA,
    GOAL_LAMP_RED_RGBA,
    GOAL_VIS_Z_OFFSET,
)


@pytest.fixture
def playback(monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/unisupermimic-test-cache")
    env = make_hand1_env(num_envs=1, backend_type="mujoco", play=True)
    try:
        env.reset()
        source = env.get_playback_model(0)
        model = mujoco.MjModel.from_xml_path(str(source)) if isinstance(source, (str, Path)) else source
        command = env.command_manager.get_term("reorient")
        yield env, command, model, mujoco.MjData(model)
    finally:
        env.close()


def test_hand1_goal_faces_follow_world_goal_without_touching_physics(playback):
    env, command, model, data = playback
    overlay = Hand1GoalOverlay(model, command)
    scene = mujoco.MjvScene(model, maxgeom=64)
    # A 90 degree rotation about Z maps the +X face onto world +Y.
    command.goal[:] = torch.tensor([[2 ** -0.5, 0., 0., 2 ** -0.5]], device=env.device)
    physical_before = env.get_physics_state_snapshot().copy()
    qpos_before = data.qpos.copy()
    target_before = command.goal.clone()
    overlay.update(scene, data)
    assert scene.ngeom == 8  # One ghost, six faces, and one lamp; no axes.
    faces = list(scene.geoms[1:7])
    for geom, rgba in zip(faces, GOAL_FACE_RGBA, strict=True):
        assert geom.type == mujoco.mjtGeom.mjGEOM_BOX
        np.testing.assert_allclose(geom.rgba, rgba)
    center = command.cube_position()[0].detach().cpu().numpy() + (0., 0., GOAL_VIS_Z_OFFSET)
    radius = GOAL_FACE_HALF_SIZE + GOAL_FACE_THICKNESS
    np.testing.assert_allclose(faces[0].pos, center + (0., radius, 0.), atol=1e-7)
    np.testing.assert_allclose(faces[1].pos, center + (0., -radius, 0.), atol=1e-7)
    np.testing.assert_array_equal(env.get_physics_state_snapshot(), physical_before)
    np.testing.assert_array_equal(data.qpos, qpos_before)
    torch.testing.assert_close(command.goal, target_before, rtol=0, atol=0)
    # Repeated frames replace markers instead of accumulating viewer geoms.
    overlay.update(scene, data)
    assert scene.ngeom == 8


@pytest.mark.parametrize(
    ("window", "within", "expected"),
    [(0, False, GOAL_LAMP_RED_RGBA), (0, True, GOAL_LAMP_AMBER_RGBA), (1, False, GOAL_LAMP_GREEN_RGBA)],
)
def test_hand1_goal_lamp_follows_alignment_and_hold(playback, window, within, expected):
    _, command, model, data = playback
    command.window.fill_(window)
    command.within.fill_(within)
    scene = mujoco.MjvScene(model, maxgeom=64)
    Hand1GoalOverlay(model, command).update(scene, data)
    lamp = scene.geoms[scene.ngeom - 1]
    assert lamp.type == mujoco.mjtGeom.mjGEOM_SPHERE
    np.testing.assert_allclose(lamp.rgba, expected)

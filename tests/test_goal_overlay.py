from types import SimpleNamespace

import numpy as np
import pytest
from unilab.envs import ManagerBasedRlEnv
from unisim.backend.base import DebugPrimitive

from wuji_unilab.tasks.reorient import overlay
from wuji_unilab.tasks.reorient.commands import ReorientCommand
from wuji_unilab.tasks.reorient.math import multiply


def _fake_env(num_envs, cube_pos, tag_quat, goal):
    state = SimpleNamespace(
        cube=SimpleNamespace(
            data=SimpleNamespace(root_link_pos_w=np.asarray(cube_pos, dtype=np.float32))
        ),
        goal=np.asarray(goal, dtype=np.float32),
        tag_pose=lambda: (
            np.zeros((num_envs, 3), dtype=np.float32),
            np.asarray(tag_quat, dtype=np.float32),
        ),
    )
    env = SimpleNamespace(
        num_envs=num_envs,
        command_manager=SimpleNamespace(get_term=lambda name: state),
    )
    return env, state


def test_goal_overlay_shape_and_kinds_per_env():
    env, _ = _fake_env(
        3,
        cube_pos=[[0.0, 0.0, 0.5], [0.1, 0.0, 0.5], [0.0, 0.1, 0.5]],
        tag_quat=[[1.0, 0.0, 0.0, 0.0]] * 3,
        goal=[[1.0, 0.0, 0.0, 0.0]] * 3,
    )
    getter = overlay.goal_overlay_getter(env)
    overlays = getter()
    assert len(overlays) == 3
    for env_primitives in overlays:
        assert [primitive.kind for primitive in env_primitives] == ["ghost_geom", "frame"]
        assert all(isinstance(primitive, DebugPrimitive) for primitive in env_primitives)


def test_goal_overlay_pose_matches_source_marker_semantics():
    # 90 deg about z (tag) composed with 90 deg about x (tag-relative goal).
    tag_quat = [2**-0.5, 0.0, 0.0, 2**-0.5]
    goal = [2**-0.5, 2**-0.5, 0.0, 0.0]
    cube_pos = [0.1, -0.2, 0.5]
    env, state = _fake_env(1, [cube_pos], [tag_quat], [goal])
    (primitives,) = overlay.goal_overlay_getter(env)()
    ghost, frame = primitives

    expected_quat = multiply(
        np.asarray([tag_quat], dtype=np.float64), np.asarray([goal], dtype=np.float64)
    )[0]
    assert ghost.mesh_asset == overlay.GOAL_MESH_ASSET == "object/cube_mesh"
    assert ghost.pos == pytest.approx((0.1, -0.2, 0.5 + overlay.GOAL_VIS_Z_OFFSET))
    assert ghost.quat == pytest.approx(tuple(expected_quat))
    assert ghost.rgba == pytest.approx(overlay.GOAL_GHOST_RGBA)
    assert ghost.rgba[3] == pytest.approx(0.6)
    assert frame.pos == pytest.approx(ghost.pos)
    assert frame.quat == pytest.approx(ghost.quat)
    assert frame.size == (overlay.GOAL_FRAME_AXIS_LENGTH,)
    assert state.goal.shape == (1, 4)


def test_goal_overlay_reflects_live_state_between_calls():
    env, state = _fake_env(1, [[0.0, 0.0, 0.5]], [[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]])
    getter = overlay.goal_overlay_getter(env)
    first = getter()[0][0]
    state.cube.data.root_link_pos_w[:] = [0.3, 0.0, 0.6]
    state.goal[:] = [0.0, 1.0, 0.0, 0.0]
    second = getter()[0][0]
    assert second.pos == pytest.approx((0.3, 0.0, 0.6 + overlay.GOAL_VIS_Z_OFFSET))
    assert second.quat == pytest.approx((0.0, 1.0, 0.0, 0.0))
    assert first.pos != second.pos


def _fake_term(state):
    return SimpleNamespace(
        playback_debug_overlay_getter=lambda: overlay.goal_overlay_getter_for_command(state)
    )


def test_command_term_opts_into_playback_overlay_discovery():
    assert callable(getattr(ReorientCommand, "playback_debug_overlay_getter", None))


def test_manager_env_discovers_goal_overlay_from_command_term():
    env, state = _fake_env(1, [[0.0, 0.0, 0.5]], [[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]])
    term = _fake_term(state)
    mock_env = SimpleNamespace(
        num_envs=1,
        command_manager=SimpleNamespace(
            active_terms=["reorient_command"], get_term=lambda name: term
        ),
    )
    getter = ManagerBasedRlEnv.get_playback_debug_overlays(mock_env)
    assert getter is not None
    overlays = getter()
    assert [primitive.kind for primitive in overlays[0]] == ["ghost_geom", "frame"]
    assert overlays[0][0].mesh_asset == overlay.GOAL_MESH_ASSET


def test_manager_env_returns_none_without_overlay_providers():
    mock_env = SimpleNamespace(
        num_envs=1,
        command_manager=SimpleNamespace(
            active_terms=["reorient_command"], get_term=lambda name: SimpleNamespace()
        ),
    )
    assert ManagerBasedRlEnv.get_playback_debug_overlays(mock_env) is None

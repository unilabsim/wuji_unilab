"""Physical wrist resets and curriculum state must match the source task."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from unisim.backend.base import MocapPoseBinding

from unisupermimic.rl.algos.dex_supermimic.env_bridge import SourceEnvBridge
from unisupermimic.tasks.dex_reorient.hand1.factory import make_hand1_env
from unisupermimic.tasks.dex_reorient.hand1.reset_bridge import bind_mocap_reset, write_mocap_reset
from unisupermimic.tasks.dex_reorient.hand1.rewards import adaptive_episode_curriculum


def _source_wrist_pose(default, pitch):
    # Independent MuJoCo quaternion math for the source's local Euler-Y reset.
    rotation, orientation = np.empty(4), np.empty(4)
    mujoco.mju_axisAngle2Quat(rotation, np.array([0.0, 1.0, 0.0]), pitch)
    mujoco.mju_mulQuat(orientation, default[3:].astype(np.float64), rotation)
    return np.concatenate((default[:3], orientation))


@pytest.mark.parametrize("play", [False, True])
def test_wrist_pitch_survives_reset_commit_and_preserves_live_worlds(monkeypatch, play):
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/unisupermimic-test-cache")
    env = make_hand1_env(num_envs=3, backend_type="mujoco", play=play)
    try:
        reset_cfg = env.event_manager.get_term_cfg("reset")
        reset_cfg.params["wrist_pitch"] = (-0.3, -0.3)
        event = reset_cfg.func
        pool, backend = env.backend._pool, env.backend
        mocap_position = pool.bind("mocap_pos", backend._np_dtype)
        mocap_quaternion = pool.bind("mocap_quat", backend._np_dtype)
        body_quaternion = pool.bind("xquat")
        palm = backend.model.body("robot/right_palm_link").id
        mocap = int(backend.model.body_mocapid[palm])
        env.reset()
        expected = _source_wrist_pose(event.default_pose.cpu().numpy(), -0.3)
        np.testing.assert_allclose(event.pose.raw.read(), np.tile(expected, (3, 1)), atol=1e-7)
        np.testing.assert_allclose(mocap_position[:, mocap], np.tile(expected[:3], (3, 1)))
        np.testing.assert_allclose(body_quaternion[:, palm], np.tile(expected[3:], (3, 1)), atol=1e-7)

        env.step(torch.zeros(3, 20, device=env.device))
        qpos = backend.get_state_views("qpos")["qpos"].clone()
        qvel = backend.get_state_views("qvel")["qvel"].clone()
        poses = event.pose.raw.read()
        reset_cfg.params["wrist_pitch"] = (0.1, 0.1)
        env.reset(env_ids=torch.tensor([1], device=env.device))
        changed = _source_wrist_pose(event.default_pose.cpu().numpy(), 0.1)
        np.testing.assert_allclose(event.pose.raw.read()[1], changed, atol=1e-7)
        np.testing.assert_allclose(mocap_quaternion[1, mocap], changed[3:], atol=1e-7)
        np.testing.assert_allclose(body_quaternion[1, palm], changed[3:], atol=1e-7)
        np.testing.assert_array_equal(event.pose.raw.read()[[0, 2]], poses[[0, 2]])
        torch.testing.assert_close(backend.get_state_views("qpos")["qpos"][[0, 2]], qpos[[0, 2]], rtol=0, atol=0)
        torch.testing.assert_close(backend.get_state_views("qvel")["qvel"][[0, 2]], qvel[[0, 2]], rtol=0, atol=0)
        position, quaternion = env.command_manager.get_term("reorient").palm_pose()
        torch.testing.assert_close(position[1], torch.tensor(changed[:3], dtype=position.dtype), atol=1e-7, rtol=0)
        torch.testing.assert_close(quaternion[1], torch.tensor(changed[3:], dtype=quaternion.dtype), atol=1e-7, rtol=0)
        env.step(torch.zeros(3, 20, device=env.device))
        np.testing.assert_allclose(event.pose.raw.read()[1], changed, atol=1e-7)
    finally:
        env.close()


def test_adaptive_curriculum_matches_source_and_restored_runner_scale(monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/unisupermimic-test-cache")
    source = (Path(__file__).parents[2] / "references/Dex-Supermimic/src/dex_supermimic"
              / "tasks/dex_reorient/mdp/curriculums.py")
    if not source.is_file():
        pytest.skip("source reference is not installed")
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "adaptive_episode_curriculum")
    namespace = {"torch": torch, "resolve_env_ids": lambda env, ids: ids}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    env = make_hand1_env(num_envs=3, backend_type="mujoco")
    try:
        env.reset()
        bridge = SourceEnvBridge(env)
        # Checkpoint restore sets the scalar consumed by imagined rollouts.
        bridge._adaptive_episode_curriculum = 0.4
        command = env.command_manager.get_term("reorient")
        command.adaptive = 0.05  # stale command state must not undo restoration
        env.episode_length_buf[:] = torch.tensor([800, 1, 200])
        env.termination_manager.terminated[:] = torch.tensor([False, True, True])
        rows = torch.tensor([0, 2])
        reference = SimpleNamespace(
            num_envs=3, episode_length_buf=env.episode_length_buf.clone(),
            termination_manager=SimpleNamespace(terminated=env.termination_manager.terminated.clone()),
            _adaptive_episode_curriculum=0.4,
        )
        params = env.curriculum_manager.get_term_cfg("adaptive_episode").params
        expected = namespace["adaptive_episode_curriculum"](reference, rows, **params)
        env.curriculum_manager.compute(rows)
        actual = env.curriculum_manager._curriculum_state["adaptive_episode"]
        assert actual == pytest.approx(expected)
        assert bridge._adaptive_episode_curriculum == pytest.approx(expected["value"])
        assert command.adaptive == pytest.approx(expected["value"])
        # Even an empty reset consumes a restored scalar without changing it.
        bridge._adaptive_episode_curriculum = 0.7
        assert adaptive_episode_curriculum(env, torch.empty(0, dtype=torch.long))["value"] == 0.7
        assert command.adaptive == 0.7
    finally:
        env.close()


def test_missing_packed_mocap_capability_defers_and_clears_selected_writes():
    """MJWarp compatibility contract, without claiming a live CUDA run."""
    poses = np.tile(np.array([0, 0, 0.5, 1, 0, 0, 0], dtype=np.float32), (3, 1))
    default = poses[0].copy()
    calls = []

    class Transaction:
        _backend = SimpleNamespace(get_tensor_capabilities=lambda: SimpleNamespace(packed_reset_mocap=False))
        _tensor_active = True
        _tensor_active_mask = torch.tensor([False, True, False])
        fail = False

        def bind_mocap_pose(self, body_name):
            return raw

        def commit(self):
            calls.append("ordinary")
            poses[1] = default  # backend ordinary reset restores mocap defaults
            if self.fail:
                raise RuntimeError("ordinary reset failed")

        commit_tensor = commit

        def abort(self):
            calls.append("abort")

        def _require_active(self):
            pass

        def _validate_ids(self, ids, **kwargs):
            return ids

        def _validate_values(self, values, **kwargs):
            return values

        def _validate_quaternions(self, values, **kwargs):
            pass

    def writer(rows, values):
        calls.append("mocap")
        poses[rows] = values

    raw = MocapPoseBinding("mjwarp", "robot/right_palm_link", 3, default,
                          lambda: poses.copy(), writer)
    transaction = Transaction()
    entity = SimpleNamespace(_reset_state=transaction)
    binding = bind_mocap_reset(entity, raw.body_name, term_name="reset")
    # Rebinding must not wrap either commit callback a second time.
    bind_mocap_reset(entity, raw.body_name, term_name="reset")
    target = torch.tensor(poses.copy())
    target[1, :3] += 0.1
    rows = torch.tensor([1])
    write_mocap_reset(binding, rows, target)
    np.testing.assert_array_equal(poses, np.tile(default, (3, 1)))
    transaction.commit_tensor()
    assert calls == ["ordinary", "mocap"]
    np.testing.assert_array_equal(poses[1], target[1].numpy())
    np.testing.assert_array_equal(poses[[0, 2]], np.tile(default, (2, 1)))
    write_mocap_reset(binding, rows, target)
    transaction.abort()
    transaction.commit()
    assert calls[-2:] == ["abort", "ordinary"]
    write_mocap_reset(binding, rows, target)
    transaction.fail = True
    with pytest.raises(RuntimeError, match="ordinary reset failed"):
        transaction.commit_tensor()
    transaction.fail = False
    transaction.commit_tensor()
    assert calls[-2:] == ["ordinary", "ordinary"]
    assert not transaction._hand1_pending_mocap
    with pytest.raises(ValueError, match="outside the active reset"):
        write_mocap_reset(binding, torch.tensor([0]), target)

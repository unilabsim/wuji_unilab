"""Compare native Pen controls with source and exercise camera tick timing."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import unisupermimic.tasks.pen_spin.actions as actions
import unisupermimic.tasks.pen_spin.observations as observations
from unisupermimic.tasks.pen_spin.commands import JOINT_NAMES

ROOT = Path(__file__).parents[2]


def _native_action_env():
    bias = torch.full((4, 20), .01)
    robot = SimpleNamespace(data=SimpleNamespace(
        default_joint_pos=lambda device: torch.zeros(20, device=device), encoder_bias_tensor=bias,
    ))
    motion = SimpleNamespace(joint_names=JOINT_NAMES, joint_pos=torch.full((4, 20), .3))
    env = SimpleNamespace(
        num_envs=4, device="cpu", scene={"robot": robot},
        command_manager=SimpleNamespace(get_term=lambda name: motion),
    )
    return env, motion


def test_native_action_matches_source_freeze_controller(monkeypatch):
    path = ROOT / "references/Dex-Supermimic/src/dex_supermimic/tasks/pen_spin/mdp/actions.py"
    if not path.is_file():
        pytest.skip("source reference is not installed")
    env, motion = _native_action_env()
    monkeypatch.setattr(actions, "task", lambda env: motion)
    cfg = actions.PenSpinActionCfg(
        entity_name="robot", mask_prob=.8, mask_num_dofs=3,
        mask_duration_min=1, mask_duration_max=5,
    )
    migrated = cfg.build(env)

    class SourceBaseAction:
        def __init__(self, cfg, env):
            self.cfg, self._env, self.device, self.num_envs = cfg, env, env.device, env.num_envs
            self._entity = SimpleNamespace(data=SimpleNamespace(encoder_bias=env.scene["robot"].data.encoder_bias_tensor))
            self._target_names, self._target_ids = JOINT_NAMES, torch.arange(20)
            self._raw_actions = torch.zeros((4, 20))
            self._processed_actions = torch.zeros_like(self._raw_actions)
            self._scale, self.action_dim = .5, 20

    tree = ast.parse(path.read_text())
    source_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SingleHandResidualEMAAction")
    namespace = dict(torch=torch, BaseAction=SourceBaseAction, SingleHandResidualEMAActionCfg=object)
    exec(compile(ast.Module(body=[source_class], type_ignores=[]), str(path), "exec"), namespace)
    source_cfg = SimpleNamespace(
        command_name="motion", alpha=.5, residual_clip=(-.6, .6), offset=0, clip=None,
        mask_prob=.8, mask_num_dofs=3, mask_duration_min=1, mask_duration_max=5,
        mask_curriculum_term=None,
    )
    source = namespace["SingleHandResidualEMAAction"](source_cfg, env)
    generator = torch.Generator().manual_seed(1)
    for step in range(12):
        raw = torch.rand((4, 20), generator=generator) * 2 - 1
        torch.manual_seed(step + 31)
        source.process_actions(raw)
        torch.manual_seed(step + 31)
        migrated.process_actions(raw)
        torch.testing.assert_close(migrated.processed_action, source.processed_action)
        torch.testing.assert_close(migrated.target, source._targets)
        torch.testing.assert_close(migrated._mask_dofs, source._mask_dofs)
        torch.testing.assert_close(migrated._mask_remaining, source._mask_remaining)
        torch.testing.assert_close(migrated._mask_active, source._mask_active)


def _camera(monkeypatch, **params):
    env = SimpleNamespace(num_envs=2, device="cpu", cfg=SimpleNamespace(auto_reset=False), scene=SimpleNamespace(read_plan=object()))
    env.position = torch.tensor([[0., 0., .05], [.02, 0., .05]])
    env.quaternion = torch.tensor([[1., 0., 0., 0.]]).expand(2, -1)
    motion = SimpleNamespace(
        cfg=SimpleNamespace(object_position_tanh_scale_m=.15),
        axis_points=torch.tensor([[.1, 0., 0.], [-.1, 0., 0.]]),
    )
    monkeypatch.setattr(observations, "task", lambda env: motion)
    monkeypatch.setattr(observations, "_runtime_pose", lambda env, motion: (env.position, env.quaternion))
    defaults = dict(bias_position_m=(.01, .01), bias_orientation_rad=(0., 0.),
                    noise_position_m=(0., 0.), noise_orientation_rad=(0., 0.),
                    lag_ticks=(1, 1), dropout_enter_prob=0., dropout_continue_prob=0.)
    camera = observations.PenObjectPoseCameraModel(SimpleNamespace(params={**defaults, **params}), env)
    return env, camera


def test_camera_reset_first_frame_bias_and_one_tick_latency(monkeypatch):
    env, camera = _camera(monkeypatch)
    camera.reset()
    first = camera(env)
    points = torch.stack((env.position + .01 + torch.tensor([.1, 0., 0.]),
                          env.position + .01 - torch.tensor([.1, 0., 0.])), 1)
    torch.testing.assert_close(first, (.15 * torch.tanh(points / .15)).flatten(1))
    env.position += .02
    torch.testing.assert_close(camera(env), first)
    second_raw = camera.memory_state()["camera_raw"].clone()
    env.position += .02
    torch.testing.assert_close(camera(env), second_raw)


def test_camera_dropout_cap_and_partial_reset_backfill(monkeypatch):
    env, camera = _camera(monkeypatch, dropout_enter_prob=1., dropout_continue_prob=1., dropout_max_ticks=2)
    first = camera(env)
    env.position += .02
    torch.testing.assert_close(camera(env), first)
    env.position += .02
    camera(env)
    assert not camera._holding.any()  # capped after the two held ticks
    before = {name: value.clone() for name, value in camera.memory_state().items()}
    history = camera._history[:, 1].clone()
    age = camera._age[1].clone()
    camera.reset(torch.tensor([0]))
    camera(env)
    torch.testing.assert_close(camera._history[:, 1], history)
    torch.testing.assert_close(camera._age[1], age)
    after = camera.memory_state()
    for name in before:
        torch.testing.assert_close(after[name][1], before[name][1])

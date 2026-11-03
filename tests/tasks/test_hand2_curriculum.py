"""CPU tests for the Hand 2 curriculum/reset-row contract."""

from types import SimpleNamespace

import pytest
import torch

from unisupermimic.tasks.dex_reorient.hand2 import rewards


def _state(goal_count: torch.Tensor):
    return SimpleNamespace(
        goal_count=goal_count,
        difficulty=0.0,
        adaptive=0.5,
        metrics={"goal_reach_count": goal_count.float()},
    )


def test_success_curriculum_uses_selected_rows_and_publishes_value(monkeypatch):
    state = _state(torch.tensor([3, 0, 4, 0], dtype=torch.float32))
    env = SimpleNamespace(
        num_envs=4,
        device=torch.device("cpu"),
        command_manager=SimpleNamespace(get_term=lambda _name: state),
    )
    monkeypatch.setattr(rewards, "task", lambda _env: state)

    value = rewards.success_curriculum(
        env,
        torch.tensor([0, 2]),
        count_threshold=3,
        delta_per_loop=0.08,
    )

    # Both selected episodes succeeded: 2 * (0.08 / 4).
    assert value["value"] == pytest.approx(0.04)
    assert value["success_env_frac"] == 1.0
    assert value["delta"] == pytest.approx(0.04)
    assert state.difficulty == pytest.approx(0.04)
    assert env._reorient_success_curriculum == pytest.approx(0.04)


def test_adaptive_episode_curriculum_ignores_live_rows(monkeypatch):
    state = _state(torch.zeros(4))
    env = SimpleNamespace(
        num_envs=4,
        device=torch.device("cpu"),
        episode_length_buf=torch.tensor([800, 400, 0, 900], dtype=torch.float32),
        termination_manager=SimpleNamespace(
            terminated=torch.tensor([False, True, True, False])
        ),
    )
    monkeypatch.setattr(rewards, "task", lambda _env: state)

    value = rewards.adaptive_episode_curriculum(
        env,
        torch.tensor([1, 3]),
        target_steps=800,
        inc_rate=0.1,
        dec_rate=0.2,
        min_scale=0.05,
    )

    # Row 1 is an early drop (progress=.5), row 3 reaches the target.  The
    # delta is (-.2*.5 + .1) / 4 == 0, while rows 0 and 2 are ignored.
    assert value["delta"] == 0.0
    assert value["mean_episode_steps"] == 650.0
    assert value["reached_target_frac"] == 0.5
    assert value["early_drop_frac"] == 0.5
    assert state.adaptive == 0.5


def test_adaptive_episode_empty_reset_is_a_noop(monkeypatch):
    state = _state(torch.zeros(2))
    env = SimpleNamespace(num_envs=2, device=torch.device("cpu"))
    monkeypatch.setattr(rewards, "task", lambda _env: state)

    value = rewards.adaptive_episode_curriculum(env, torch.empty(0, dtype=torch.long))

    assert value == {"value": 0.5}


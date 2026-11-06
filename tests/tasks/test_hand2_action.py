from types import SimpleNamespace

import pytest
import torch

from unisupermimic.tasks.dex_reorient.hand2.actions import WujiAction, WujiActionCfg


def _env(num_envs: int):
    robot = SimpleNamespace(
        data=SimpleNamespace(
            default_joint_pos=lambda device: torch.zeros(20, device=device),
            soft_joint_pos_limits=lambda device: torch.tile(
                torch.tensor([-1.0, 1.0], device=device), (20, 1)
            ),
        )
    )
    return SimpleNamespace(
        scene={"robot": robot},
        num_envs=num_envs,
        episode_length_buf=torch.full((num_envs,), 20),
        step_dt=0.05,
    )


def test_wuji_action_rejects_nonfinite_policy_output():
    action = WujiAction(WujiActionCfg(entity_name="robot"), _env(2))
    with pytest.raises(FloatingPointError, match="action contains NaN or Inf"):
        action.process_actions(
            torch.tensor([[float("nan"), float("inf")] + [0.0] * 18, [0.0] * 20])
        )


def test_wuji_action_tracks_previous_ema_target():
    action = WujiAction(WujiActionCfg(entity_name="robot", ema_alpha=0.5), _env(1))
    action.process_actions(torch.ones(1, 20))
    first = action.target.clone()
    action.process_actions(torch.zeros(1, 20))
    torch.testing.assert_close(action.previous_target, first)

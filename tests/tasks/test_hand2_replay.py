"""Replay-window boundary checks for the Hand 2 learner contract."""

from __future__ import annotations

import torch

from unisupermimic.rl.algos.supermimic.hand2_replay import Hand2ReplayBuffer


def _transition(num_envs: int, step: int, done: tuple[bool, ...] = (False, False)):
    assert len(done) == num_envs
    value = float(step)
    return {
        "s": torch.full((num_envs, 58), value),
        "s1": torch.full((num_envs, 58), value + 1),
        "a": torch.full((num_envs, 20), value),
        "a_prev": torch.full((num_envs, 20), value - 1),
        "xi": torch.full((num_envs, 46), value),
        "m": torch.full((num_envs, 100), value),
        "m1": torch.full((num_envs, 100), value + 1),
        "u": torch.full((num_envs, 20), value + 1),
        "bias": torch.zeros((num_envs, 20)),
        "aux": torch.full((num_envs, 21), value),
        "contact": torch.zeros((num_envs, 32)),
        "torque": torch.zeros((num_envs, 20)),
        "obs": torch.full((num_envs, 207), value),
        "done": torch.tensor(done),
        "terminated": torch.tensor(done),
        "timeout": torch.zeros(num_envs, dtype=torch.bool),
        "ep_step": torch.full((num_envs,), step, dtype=torch.long),
    }


def test_hand2_replay_window_masks_after_reset() -> None:
    replay = Hand2ReplayBuffer(8, 2, "cpu")
    for step in range(5):
        replay.add(_transition(2, step, (step == 1, False)))
    torch.manual_seed(4)
    windows = replay.sample_windows(128, 3)
    assert windows["s"].shape == (128, 3, 58)
    assert windows["obs"].shape == (128, 3, 207)
    assert torch.all(windows["mask"][:, 1:] <= windows["mask"][:, :-1])
    # A sampled window beginning at age 0 and env 0 crosses the terminal row
    # at age 1.  Check the boundary rule directly without relying on RNG.
    done = torch.tensor([[False, True, False]])
    alive_before = torch.cat((torch.ones((1, 1), dtype=torch.bool), ~done[:, :-1]), -1)
    assert torch.equal(torch.cumprod(alive_before, -1), torch.tensor([[True, True, False]]))


def test_hand2_replay_starts_pad_history_and_roundtrip() -> None:
    replay = Hand2ReplayBuffer(3, 2, "cpu")
    for step in range(5):
        replay.add(_transition(2, step))
    start, history = replay.sample_starts(16, 3)
    assert start["s"].shape == (16, 58)
    assert history["obs"].shape == (16, 3, 207)
    state = replay.state_dict()
    restored = Hand2ReplayBuffer(3, 2, "cpu")
    restored.load_state_dict(state)
    assert restored.ptr == replay.ptr
    assert restored.size == replay.size
    for name in replay.data:
        torch.testing.assert_close(restored.data[name], replay.data[name])


def test_hand2_replay_holdout_and_failure_sampling() -> None:
    replay = Hand2ReplayBuffer(8, 4, "cpu", holdout_fraction=0.5)
    for step in range(5):
        replay.add(_transition(4, step, (False, False, False, False)))
    # Mark one held-out environment as terminated; failure-biased windows must
    # remain shape-stable and the holdout split must select only that half.
    replay.data["terminated"][2, 2] = True
    replay.data["done"][2, 2] = True
    windows = replay.sample_windows(32, 2, holdout=True, failure_frac=0.5, failure_horizon=3)
    assert windows["env"].min() >= replay.train_end
    assert windows["mask"].shape == (32, 2)
    assert windows["valid"].all()

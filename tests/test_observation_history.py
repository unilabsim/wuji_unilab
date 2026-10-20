"""CPU contracts for the Wuji-owned observation history stack."""

import pytest
import torch

from wuji_unilab.tasks.reorient.observations import (
    HistoryFeature,
    NoisyHistoryFeature,
    TipsFeature,
)


class FakeRng:
    def __init__(self):
        self.generator = torch.Generator().manual_seed(7)
        self.device = torch.device("cpu")

    def uniform(self, low, high, size, dtype=None):
        unit = torch.rand(size, dtype=dtype or torch.float32, generator=self.generator)
        return unit * (high - low) + low


class FakeFeatureEnv:
    def __init__(self, batch: int):
        self.num_envs = batch
        self.device = torch.device("cpu")
        self.rng = FakeRng()
        self.frame = torch.zeros((batch, 2), dtype=torch.float32)
        self.command_manager = type(
            "Commands",
            (),
            {
                "get_term": lambda self, name: type(
                    "Term", (), {"tip_names": ("tip0", "tip1", "tip2", "tip3", "tip4")}
                )()
            },
        )()

    def next_frame(self) -> torch.Tensor:
        self.frame = self.frame + 1.0
        return self.frame


@pytest.fixture
def replace_feature(monkeypatch):
    def apply():
        monkeypatch.setattr(
            "wuji_unilab.tasks.reorient.observations.feature",
            lambda task_env, kind, **params: task_env.next_frame(),
        )

    return apply


def test_history_flattens_oldest_to_newest_and_rejects_bad_length(replace_feature):
    with pytest.raises(ValueError, match="positive"):
        HistoryFeature(None, FakeFeatureEnv(2), kind="joint", history_length=0)

    replace_feature()
    env = FakeFeatureEnv(2)
    term = HistoryFeature(None, env, kind="joint", history_length=3)
    assert term.tensor_state_entity == "robot"
    first = term(None)
    second = term(None)
    third = term(None)
    torch.testing.assert_close(
        torch.stack((first, second, third))[:, 0, 0:6:2],
        torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 2.0],
                [1.0, 2.0, 3.0],
            ]
        ),
    )
    torch.testing.assert_close(third[:, 0:2], torch.full((2, 2), 1.0))
    torch.testing.assert_close(third[:, 2:4], torch.full((2, 2), 2.0))
    torch.testing.assert_close(third[:, 4:6], torch.full((2, 2), 3.0))


def test_history_reset_backfills_only_selected_rows(replace_feature):
    replace_feature()
    env = FakeFeatureEnv(2)
    term = HistoryFeature(None, env, kind="joint", history_length=2)
    for _ in range(3):
        term(None)
    before = term.buffer.buffer.clone()
    term.reset(torch.tensor([True, False]))
    assert torch.equal(term.buffer.buffer[0], torch.zeros(2, 2))
    assert torch.equal(term.buffer.buffer[1], before[1])

    history_before_append = term.buffer.buffer[1].clone()
    term(None)
    assert torch.equal(term.buffer.buffer[0, 0], term.buffer.buffer[0, 1])
    assert torch.equal(term.buffer.buffer[1, 0], history_before_append[1])
    assert torch.equal(term.buffer.buffer[1, 1], term.buffer.buffer[1, 0] + 1)


def test_history_noise_is_drawn_per_frame_not_per_stack(replace_feature):
    replace_feature()
    env = FakeFeatureEnv(1)
    term = NoisyHistoryFeature(
        None,
        env,
        kind="joint",
        history_length=3,
        low=-0.06,
        high=0.06,
    )
    first = term(None)
    second = term(None)
    assert torch.equal(first[:, :2], first[:, 2:4])
    assert torch.equal(first[:, 4:], first[:, 2:4])
    assert torch.equal(second[:, :2], first[:, 4:])
    assert not torch.equal(second[:, 4:], first[:, 4:])


def test_tips_declares_history_body_reads_and_scales_each_frame(replace_feature):
    replace_feature()
    env = FakeFeatureEnv(2)
    term = TipsFeature(None, env, history_length=3, scale=5.0)
    assert term.tensor_state_entity == "robot"
    assert term.tensor_body_names == ("tip0", "tip1", "tip2", "tip3", "tip4")
    output = term(None)
    assert torch.equal(output, torch.full((2, 6), 5.0))

"""Checkpoint and CSV logging contracts for the Hand 2 runner."""

from __future__ import annotations

import csv

import torch

from unisupermimic.rl.algos.supermimic.hand2_checkpoint import (
    Hand2CSVLogger,
    load_hand2_checkpoint,
    save_hand2_checkpoint,
)


def test_hand2_checkpoint_roundtrip_preserves_modules_optimizer_and_replay(tmp_path):
    torch.manual_seed(7)
    module = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
    loss = module(torch.ones(1, 3)).square().sum()
    loss.backward()
    optimizer.step()
    expected = {name: value.detach().clone() for name, value in module.state_dict().items()}
    path = save_hand2_checkpoint(
        tmp_path / "last.pt",
        cfg={"name": "test"},
        modules={"actor": module},
        optimizers={"actor": optimizer},
        iteration=3,
        sim_steps=12,
        replay={"s": [torch.ones(4), torch.zeros(4)]},
    )
    with torch.no_grad():
        for value in module.parameters():
            value.zero_()
    restored = load_hand2_checkpoint(
        path,
        modules={"actor": module},
        optimizers={"actor": optimizer},
    )
    for name, value in module.state_dict().items():
        torch.testing.assert_close(value, expected[name])
    assert restored["iteration"] == 3
    assert restored["sim_steps"] == 12
    assert len(restored["replay"]["s"]) == 2


def test_hand2_csv_logger_expands_fields(tmp_path):
    path = tmp_path / "train_log.csv"
    logger = Hand2CSVLogger(path)
    logger.log({"iteration": 0, "wm_loss": torch.tensor(1.25)})
    logger.log({"iteration": 1, "pi_loss": -0.5})
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["wm_loss"] == "1.25"
    assert rows[1]["pi_loss"] == "-0.5"
    assert rows[0]["pi_loss"] == ""

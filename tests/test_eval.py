import argparse

import pytest

from wuji_unilab.eval import TrialOutcome, _parser, summarize


def test_eval_defaults_match_source_trial_protocol(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr("sys.argv", ["wuji-eval", "--checkpoint-file", str(checkpoint)])
    args: argparse.Namespace = _parser().parse_args()
    assert args.num_trials == 100
    assert args.trial_timeout == pytest.approx(14.0)
    assert args.success_threshold == pytest.approx(0.2)
    assert args.success_hold_steps == 5
    assert args.drop_height == pytest.approx(-0.15)


def test_eval_summary_uses_trial_outcomes_not_training_goal_count():
    trials = [
        TrialOutcome(0, "success", 1.5, 1, 0.1, 0.05),
        TrialOutcome(1, "drop", None, 0, 1.0, 0.4),
        TrialOutcome(2, "timeout", None, 0, 0.8, 0.3),
        TrialOutcome(3, "success", 2.5, 1, 0.15, 0.1),
    ]
    result = summarize(trials)
    assert result["success_rate"] == pytest.approx(0.5)
    assert result["drop_rate"] == pytest.approx(0.25)
    assert result["timeout_rate"] == pytest.approx(0.25)
    assert result["mean_goal_reaches"] == pytest.approx(0.5)
    assert result["mean_time_to_first_success_s"] == pytest.approx(2.0)
    assert result["mean_min_orientation_error_rad"] == pytest.approx(0.2125)

import argparse
from types import SimpleNamespace

import pytest

from wuji_unilab.eval import TrialOutcome, _parser, _require_record_capabilities, summarize


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


def test_eval_record_cli_defaults_to_pure_data_mode(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr("sys.argv", ["wuji-eval", "--checkpoint-file", str(checkpoint)])
    args: argparse.Namespace = _parser().parse_args()
    assert args.record is False
    assert args.video_output is None


def test_eval_record_cli_parses_video_output(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        "sys.argv",
        [
            "wuji-eval",
            "--checkpoint-file",
            str(checkpoint),
            "--record",
            "--video-output",
            str(tmp_path / "out.mp4"),
        ],
    )
    args: argparse.Namespace = _parser().parse_args()
    assert args.record is True
    assert args.video_output == tmp_path / "out.mp4"


def _env_with_capabilities(playback, overlay):
    return SimpleNamespace(
        play_capabilities=SimpleNamespace(
            supports_physics_state_playback=playback,
            supports_debug_overlay=overlay,
        )
    )


def test_record_capability_gate_accepts_full_support():
    _require_record_capabilities(_env_with_capabilities(True, True))


def test_record_capability_gate_rejects_missing_playback():
    with pytest.raises(RuntimeError, match="physics-state playback"):
        _require_record_capabilities(_env_with_capabilities(False, True))
    with pytest.raises(RuntimeError, match="physics-state playback"):
        _require_record_capabilities(SimpleNamespace())


def test_record_capability_gate_rejects_missing_overlay():
    with pytest.raises(RuntimeError, match="debug overlay"):
        _require_record_capabilities(_env_with_capabilities(True, False))

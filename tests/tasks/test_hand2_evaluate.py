import pytest
import torch

from unisupermimic.tasks.dex_reorient.hand2.evaluate import Hand2EvalMetrics


def _transition(
    *, done, term=None, timeout=None, goal_count=None, error=None, before=None, success=None
):
    size = len(done)
    value = {
        "done": torch.tensor(done, dtype=torch.bool),
        "term": torch.tensor(term or [False] * size, dtype=torch.bool),
        "timeout": torch.tensor(timeout or [False] * size, dtype=torch.bool),
    }
    if goal_count is not None:
        value["goal_count"] = torch.tensor(goal_count, dtype=torch.long)
    if error is not None:
        value["orientation_error"] = torch.tensor(error, dtype=torch.float32)
    if before is not None:
        value["orientation_error_before"] = torch.tensor(before, dtype=torch.float32)
    if success is not None:
        value["success"] = torch.tensor(success, dtype=torch.bool)
    return value


def test_hand2_eval_metrics_accumulate_terminal_rows_and_reset_them():
    metrics = Hand2EvalMetrics(2)
    metrics.update(
        torch.tensor([1.0, 2.0]),
        _transition(
            done=[False, False], goal_count=[0, 0], error=[1.0, 0.8], before=[1.2, 1.0]
        ),
    )
    metrics.update(
        torch.tensor([3.0, 4.0]),
        _transition(
            done=[False, True],
            term=[False, True],
            goal_count=[1, 0],
            error=[0.1, 0.7],
            before=[0.2, 0.9],
            success=[True, False],
        ),
    )
    metrics.update(
        torch.tensor([5.0, 6.0]),
        _transition(
            done=[True, False],
            term=[True, False],
            timeout=[True, False],
            goal_count=[1, 0],
            error=[0.2, 0.6],
            before=[0.15, 0.65],
        ),
    )

    summary = metrics.summary()
    assert summary["steps"] == 6
    assert summary["episodes"] == 2
    assert summary["total_reward"] == pytest.approx(21.0)
    assert summary["goal_reaches"] == 1
    assert summary["mean_goal_reaches"] == pytest.approx(0.5)
    assert summary["success_rate"] == pytest.approx(0.5)
    assert summary["cage_drop_count"] == 1
    assert summary["cage_drop_rate"] == pytest.approx(0.5)
    assert summary["timeout_count"] == 1
    assert summary["mean_episode_length_steps"] == pytest.approx(2.5)
    assert summary["mean_min_orientation_error_rad"] == pytest.approx(0.4)
    assert summary["mean_final_orientation_error_rad"] == pytest.approx(0.45)
    assert summary["mean_target_progress_rad"] == pytest.approx(0.1166667)
    assert summary["active_episode_steps_mean"] == 0.5


def test_hand2_eval_metrics_reports_none_for_open_episode_aggregates():
    metrics = Hand2EvalMetrics(1)
    metrics.update(
        torch.tensor([2.0]),
        _transition(done=[False], goal_count=[0], error=[0.3]),
    )
    summary = metrics.summary()
    assert summary["mean_reward_per_step"] == 2.0
    assert summary["episodes"] == 0
    assert summary["success_rate"] is None
    assert summary["mean_episode_length_steps"] is None
    assert summary["mean_orientation_error_rad"] == pytest.approx(0.3)
    assert summary["mean_min_orientation_error_rad"] == pytest.approx(0.3)
    assert summary["mean_final_orientation_error_rad"] == pytest.approx(0.3)
    assert summary["active_episode_steps_mean"] == 1.0

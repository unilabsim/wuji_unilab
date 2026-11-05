"""Hand1 forwards native completed-episode reductions without redefining them."""

import csv

import pytest
import torch

from unisupermimic.rl.algos.dex_supermimic.config import DexSuperMimicCfg
from unisupermimic.rl.algos.dex_supermimic.env_bridge import SourceEnvBridge
from unisupermimic.rl.algos.dex_supermimic.runner import DexSuperMimicRunner, organize_log_scalars
from unisupermimic.tasks.dex_reorient.hand1.factory import make_hand1_env


def test_hand1_collect_preserves_native_episode_metrics_and_batch_weights(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/unisupermimic-test-cache")
    # Existing runs gain the new native metric columns without losing rows
    # or any user-retained column when their CSV is reopened for append.
    (tmp_path / "train_log.csv").write_text("iteration,wm/wm_loss,custom/retained\n12,0.2,9\n")
    native = make_hand1_env(num_envs=2, backend_type="mujoco", play=True)
    runner = None
    try:
        runner = DexSuperMimicRunner(SourceEnvBridge(native), DexSuperMimicCfg(
            task="hand1-reorient", logger="none", device="cpu", num_envs=2,
            buffer_steps=8, warmup_steps=8, wm_window=1, wm_batch=2, wm_batches=1,
            wm_hidden=(8,), pi_batch=2, horizon=1, max_horizon=1,
            critic_minibatches=1, compile=False, compile_policy=False,
        ), tmp_path)
        published = []
        original_reset = native.metrics_manager.reset

        def capture(env_ids=None):
            values = original_reset(env_ids)
            published.append((env_ids.numel(), dict(values)))
            return values

        monkeypatch.setattr(native.metrics_manager, "reset", capture)
        calls = 0

        def selected_terminal_rows(env, **params):
            nonlocal calls
            calls += 1
            return torch.tensor([True, calls > 1], device=env.device)

        term = native.termination_manager.get_term_cfg("cage_drop")
        monkeypatch.setattr(term, "func", selected_terminal_rows)
        stats = runner.collect(2)
        assert [count for count, _ in published] == [1, 2]
        assert "Episode_Metrics/goal_reach_count" in published[0][1]
        assert "Episode_Metrics/orientation_error" in published[0][1]
        assert "Episode_Metrics/joint_acceleration_rms" in published[0][1]
        for name in published[0][1]:
            tag = "task/episode_" + name.split("/", 1)[1]
            expected = sum(count * values[name] for count, values in published) / 3
            assert stats[tag] == pytest.approx(expected)
        runner.writer.log_scalars(organize_log_scalars(stats, 0), 0)
        with (tmp_path / "train_log.csv").open(newline="") as stream:
            old, row = tuple(csv.DictReader(stream))
        assert old["iteration"] == "12"
        assert old["wm/wm_loss"] == "0.2"
        assert old["custom/retained"] == "9"
        assert old["task/episode_goal_reach_count"] == ""
        for name in published[0][1]:
            tag = "task/episode_" + name.split("/", 1)[1]
            assert float(row[tag]) == pytest.approx(stats[tag])
        # No completed episode means no misleading zero-valued reductions.
        monkeypatch.setattr(term, "func", lambda env, **params: torch.zeros(2, dtype=torch.bool))
        assert not any(key.startswith("task/episode_") for key in runner.collect(1))
    finally:
        if runner is not None:
            runner.close()
        native.close()

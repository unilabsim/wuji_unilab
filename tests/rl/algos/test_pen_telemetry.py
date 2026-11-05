"""Native completed-episode metrics survive the source learner boundary."""

import csv
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from unilab.base import registry
from unilab.base.config_adapter import EnvConfigAdapter
from unilab.training import ensure_registries

from unisupermimic.rl.algos.dex_supermimic.config import DexSuperMimicCfg
from unisupermimic.rl.algos.dex_supermimic.env_bridge import SourceEnvBridge
from unisupermimic.rl.algos.dex_supermimic.runner import DexSuperMimicRunner, organize_log_scalars

ROOT = Path(__file__).parents[3]


def test_collect_forwards_native_episode_metrics_with_correct_weighting(tmp_path, monkeypatch):
    ensure_registries()
    with initialize_config_dir(
        config_dir=str(ROOT / "src/unisupermimic/conf/supermimic"), version_base="1.3"
    ):
        cfg = compose("config_pen_spin_supermimic", overrides=[
            "task=dex_hand2_pen_spin/mujoco", "algo.num_envs=2",
            "env.commands.motion.synthetic_reference=true",
            "env.commands.motion.recipe_path=", "env.commands.motion.clips_root=null",
        ])
    override = EnvConfigAdapter(cfg, root_dir=ROOT, algo_name="supermimic").build_task_env_cfg_override()
    native = registry.make("DexHand2PenSpin", sim_backend="mujoco", num_envs=2, env_cfg_override=override)
    runner = None
    try:
        env = SourceEnvBridge(native)
        runner = DexSuperMimicRunner(env, DexSuperMimicCfg(
            task="hand2-pen-spin", logger="none", device="cpu", num_envs=2,
            buffer_steps=8, warmup_steps=8, wm_window=1, wm_batch=2, wm_batches=1,
            wm_hidden=(8,), pi_batch=2, horizon=1, max_horizon=1,
            critic_minibatches=1,
            compile=False, compile_policy=False,
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
            # Exercise differently sized completed-episode batches: one row
            # first, then two rows. Native MetricsManager owns all reductions.
            calls += 1
            return torch.tensor([True, calls > 1], device=env.device)

        term = native.termination_manager.get_term_cfg("object_position_error")
        monkeypatch.setattr(term, "func", selected_terminal_rows)
        stats = runner.collect(2)
        assert [count for count, _ in published] == [1, 2]
        assert len(published[0][1]) == 7
        for name in published[0][1]:
            tag = "task/episode_" + name.split("/", 1)[1]
            expected = sum(count * values[name] for count, values in published) / 3
            assert stats[tag] == pytest.approx(expected)
        runner.writer.log_scalars(organize_log_scalars(stats, 0), 0)
        with (tmp_path / "train_log.csv").open(newline="") as stream:
            (row,) = tuple(csv.DictReader(stream))
        for name in published[0][1]:
            tag = "task/episode_" + name.split("/", 1)[1]
            assert float(row[tag]) == pytest.approx(stats[tag])
    finally:
        if runner is not None:
            runner.close()
        native.close()

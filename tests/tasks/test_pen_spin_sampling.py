"""Compare the migrated sampler with the original pure source algorithms."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from unisupermimic.tasks.pen_spin.commands import PenSpinCommand, PenSpinCommandCfg


def _source_sampler():
    path = (
        Path(__file__).parents[2]
        / "references/Dex-Supermimic/src/dex_supermimic/tasks/pen_spin/mdp/commands.py"
    )
    if not path.is_file():
        pytest.skip("Dex-SuperMimic reference checkout is unavailable")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    source = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SingleHandTrackingCommand"
    )
    methods = {
        "_configure_adaptive_sampling_bins", "_bin_ids_at", "_current_bin_ids",
        "_record_bin_outcomes", "_update_bin_failure_scores",
        "_smoothed_bin_failure_scores", "_adaptive_sampling_probabilities",
        "_set_sampling_distribution_metrics", "prepare_reset", "_hold_first_frame", "_update_command",
    }
    source.body = [
        node for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name in methods
    ]
    # Execute the original functions without importing mjlab or its registry.
    module = ast.Module(body=[tree.body[0], source], type_ignores=[])
    namespace = {"CommandTerm": object, "torch": torch, "dist": dist, "math": math}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["SingleHandTrackingCommand"]


def _sampler_fixture(command_type=PenSpinCommand, *, mode="adaptive"):
    command = command_type.__new__(command_type)
    command.cfg = PenSpinCommandCfg(
        trajectory_sampling_mode=mode,
        adaptive_sampling_bin_seconds=0.5,
        adaptive_sampling_ema_decay=0.5,
        adaptive_sampling_update_steps=2,
        adaptive_sampling_kernel_size=3,
        adaptive_sampling_kernel_decay=0.8,
        first_frame_hold_prob=0.25,
        first_frame_hold_s=(0.04, 0.08),
    )
    device = torch.device("cpu")
    command._device = device
    if command_type is not PenSpinCommand:
        command.device = device
    unstable = torch.tensor([False, False, False, True, False, False])
    command._env = SimpleNamespace(
        num_envs=6,
        episode_length_buf=torch.tensor([5, 5, 5, 5, 0, 0]),
        termination_manager=SimpleNamespace(
            terminated=torch.tensor([True, False, False, True, True, False]),
            time_outs=torch.tensor([False, True, True, False, False, False]),
            get_term=lambda name: unstable,
        ),
    )
    command.group = SimpleNamespace(frame_dt=0.02, num_trajectories=3)
    command._trajectory_lengths = torch.tensor([151, 88, 2])
    command._trajectory_starts = torch.tensor([0, 151, 239])
    command._trajectory_ends = torch.tensor([151, 239, 241])
    command._configure_adaptive_sampling_bins()
    command.trajectory_ids = torch.tensor([0, 0, 1, 1, 2, 1])
    command.time_steps = torch.tensor([65, 110, 86, 70, 1, 12])
    command._episode_start_steps = torch.tensor([0, 27, 0, 50, 0, -5])
    command._reset_prepared = torch.zeros(6, dtype=torch.bool)
    command.reference_end = torch.ones(6, dtype=torch.bool)
    command.bin_failure_scores = torch.linspace(0, 1, 11)
    command._pending_bin_outcomes = torch.zeros(2, 11)
    command._steps_since_score_update = 0
    for name in (
        "_sampling_probability", "_sampled_bin_failure_score", "_sampling_entropy",
        "_sampling_top1_prob", "_sampling_top1_bin",
    ):
        setattr(command, name, torch.zeros(6))
    return command


def _assert_tensor_state_equal(actual, expected):
    for name, value in vars(actual).items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, getattr(expected, name), rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["uniform", "adaptive"])
def test_sampler_matches_source_with_partial_resets_and_hold(mode):
    actual = _sampler_fixture(mode=mode)
    expected = _sampler_fixture(_source_sampler(), mode=mode)
    _assert_tensor_state_equal(actual, expected)
    selected = torch.tensor([0, 2, 5])
    before_trajectory = actual.trajectory_ids.clone()
    before_steps = actual.time_steps.clone()
    torch.manual_seed(324)
    actual.prepare_reset(selected)
    torch.manual_seed(324)
    expected.prepare_reset(selected)
    _assert_tensor_state_equal(actual, expected)
    torch.testing.assert_close(actual.trajectory_ids[[1, 3, 4]], before_trajectory[[1, 3, 4]])
    torch.testing.assert_close(actual.time_steps[[1, 3, 4]], before_steps[[1, 3, 4]])

    # A reset event prepares and writes this sample. The manager must consume
    # it without drawing another sample and invalidating the physical reset.
    prepared_trajectory = actual.trajectory_ids.clone()
    prepared_steps = actual.time_steps.clone()
    rng = torch.random.get_rng_state()
    actual._resample_command(torch.tensor([True, False, True, False, False, True]))
    torch.testing.assert_close(torch.random.get_rng_state(), rng)
    torch.testing.assert_close(actual.trajectory_ids, prepared_trajectory)
    torch.testing.assert_close(actual.time_steps, prepared_steps)
    assert not actual._reset_prepared.any()
    assert actual.reference_end.tolist() == [False, True, False, True, True, False]


def test_bin_outcomes_record_survivals_and_exclude_unstable_resets():
    actual = _sampler_fixture()
    expected = _sampler_fixture(_source_sampler())
    env_ids = torch.arange(6)
    actual._record_bin_outcomes(env_ids)
    expected._record_bin_outcomes(env_ids)
    _assert_tensor_state_equal(actual, expected)
    torch.testing.assert_close(
        actual._pending_bin_outcomes[0],
        torch.tensor([1, 2, 2, 1, 1, 0, 1, 1, 1, 1, 0], dtype=torch.float32),
    )
    torch.testing.assert_close(
        actual._pending_bin_outcomes[1],
        torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32),
    )
    # Score updates happen on the same fixed control step on every rank,
    # independent of which environments happen to reset.
    before = actual.bin_failure_scores.clone()
    actual._update_bin_failure_scores()
    expected._update_bin_failure_scores()
    torch.testing.assert_close(actual.bin_failure_scores, before)
    actual._update_bin_failure_scores()
    expected._update_bin_failure_scores()
    _assert_tensor_state_equal(actual, expected)
    assert not actual._pending_bin_outcomes.any()
    torch.testing.assert_close(
        actual._adaptive_sampling_probabilities(),
        expected._adaptive_sampling_probabilities(),
        rtol=0, atol=0,
    )
    # The forward smoothing kernel must stop at each trajectory's final bin.
    assert actual._bin_kernel_indices[5].tolist() == [5, 5, 5]
    assert actual._bin_kernel_indices[9].tolist() == [9, 9, 9]
    assert actual._bin_kernel_indices[10].tolist() == [10, 10, 10]


def test_reset_refresh_matches_source_selected_advance_and_reference_end():
    actual = _sampler_fixture(mode="uniform")
    expected = _sampler_fixture(_source_sampler(), mode="uniform")
    # Include a terminal frame and both sides of the first-frame hold boundary.
    actual.time_steps[:] = torch.tensor([149, 110, -50, -1, 0, 12])
    expected.time_steps.copy_(actual.time_steps)
    selected = torch.tensor([0, 2, 3, 4])
    mask = torch.zeros(6, dtype=torch.bool)
    mask[selected] = True
    before_latch = actual.reference_end.clone()
    actual.refresh_reset(mask)
    expected._update_command(selected)
    torch.testing.assert_close(actual.time_steps, expected.time_steps, rtol=0, atol=0)
    assert actual.time_steps[2] == -49 and actual.time_steps[3] == 0
    path = (Path(__file__).parents[2]
            / "references/Dex-Supermimic/src/dex_supermimic/tasks/pen_spin/mdp/terminations.py")
    tree = ast.parse(path.read_text())
    end = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name == "reference_end")
    namespace = {}
    exec(compile(ast.Module(body=[end], type_ignores=[]), str(path), "exec"), namespace)
    expected.trajectory_lengths = expected._trajectory_lengths[expected.trajectory_ids]
    oracle_env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: expected))
    source_end = namespace["reference_end"](oracle_env)
    torch.testing.assert_close(actual.reference_end[mask], source_end[mask])
    torch.testing.assert_close(actual.reference_end[~mask], before_latch[~mask])


def test_reference_control_frequency_mismatch_fails_explicitly():
    command = PenSpinCommand.__new__(PenSpinCommand)
    command._env = SimpleNamespace(step_dt=0.01)
    with pytest.raises(ValueError, match="time-warps the reference"):
        command._validate_control_frequency(0.02)


def test_adaptive_partial_reset_writes_the_selected_reference_frame():
    """Regress event-before-command ordering on actual MuJoCo state."""
    from hydra import compose, initialize_config_dir
    from unilab.base import registry
    from unilab.base.config_adapter import EnvConfigAdapter
    from unilab.training import ensure_registries

    root = Path(__file__).parents[2]
    recipe = root / "src/unisupermimic/tasks/pen_spin/data/recipe/01_single_axis.json"
    clips = recipe.parents[1] / "rawdata/clips/01_single_axis/index.json"
    if not clips.is_file():
        pytest.skip("upstream pen clip catalog is not installed")
    ensure_registries()
    with initialize_config_dir(
        config_dir=str(root / "src/unisupermimic/conf/supermimic"), version_base="1.3"
    ):
        cfg = compose("config_pen_spin", overrides=[
            "algo.num_envs=2",
            f"env.commands.motion.recipe_path={recipe}",
            "env.commands.motion.synthetic_reference=false",
            "env.commands.motion.first_frame_hold_prob=0.0",
            "env.commands.motion.adaptive_sampling_uniform_mix=0.0",
            # Isolate reference-reset ordering from model randomization and
            # subsequent step disturbances, which have separate integration tests.
            "env.events.source_physics=null",
            "env.events.source_disturbances=null",
        ])
    override = EnvConfigAdapter(cfg, root_dir=root, algo_name="ppo").build_task_env_cfg_override()
    env = registry.make("DexHand2PenSpin", sim_backend="mujoco", env_cfg_override=override, num_envs=2)
    try:
        env.reset()
        motion = env.command_manager.get_term("motion")
        prepared = {}
        source_sampler = motion.prepare_reset

        def capture_prepared(ids):
            source_sampler(ids)
            prepared.update(ids=ids.clone(), time=motion.time_steps[ids].clone(),
                            joint_pos=motion.joint_pos[ids].clone(),
                            joint_vel=motion.joint_vel[ids].clone())

        motion.prepare_reset = capture_prepared
        motion.bin_failure_scores.zero_()
        motion.bin_failure_scores[2] = 1.0
        env.reset()
        torch.testing.assert_close(motion.time_steps, prepared["time"] + 1)
        actual = env.scene.read_plan.joint_tensor_view("robot")
        torch.testing.assert_close(actual.joint_pos, prepared["joint_pos"], rtol=0, atol=1e-7)
        torch.testing.assert_close(actual.joint_vel, prepared["joint_vel"], rtol=0, atol=1e-7)
        torch.testing.assert_close(env.obs_buf["obs"][:, :20], motion.joint_pos)
        previous_frames = motion.frame_ids.clone()
        before = env.backend.get_state_views(("qpos", "qvel"), device="cpu")
        before = {key: value.clone() for key, value in before.items()}
        motion.bin_failure_scores.zero_()
        motion.bin_failure_scores[3] = 1.0
        env.reset(env_ids=torch.tensor([0]))
        assert motion.frame_ids[0] != previous_frames[0]
        assert motion.frame_ids[1] == previous_frames[1]
        actual = env.scene.read_plan.joint_tensor_view("robot")
        torch.testing.assert_close(actual.joint_pos[0], prepared["joint_pos"][0], rtol=0, atol=1e-7)
        torch.testing.assert_close(actual.joint_vel[0], prepared["joint_vel"][0], rtol=0, atol=1e-7)
        torch.testing.assert_close(motion.time_steps[:1], prepared["time"] + 1)
        torch.testing.assert_close(env.obs_buf["obs"][0, :20], motion.joint_pos[0])
        after = env.backend.get_state_views(("qpos", "qvel"), device="cpu")
        for key, value in before.items():
            torch.testing.assert_close(after[key][1], value[1], rtol=0, atol=0)
    finally:
        env.close()

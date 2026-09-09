"""Source-compatible trial evaluation for a Wuji checkpoint on mjwarp."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import warp as wp
from uni_rl.algos.rsl_rl import normalize_ppo_train_cfg
from uni_rl.algos.rsl_rl_training_state import TrainingStateOnPolicyRunner
from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.envs import ManagerBasedRlEnv
from unilab.training import algo_config_dict
from unilab.visualization.playback_session import SnapshotPlaybackSession
from unisim.backend.base import CameraCfg

from wuji_unilab.config import compose_task
from wuji_unilab.rl.runtime import WujiWrapper
from wuji_unilab.tasks.reorient.commands import task
from wuji_unilab.tasks.reorient.math import angle_error, random_quaternions
from wuji_unilab.tasks.reorient.overlay import goal_overlay_getter


@dataclass(frozen=True)
class TrialOutcome:
    trial_idx: int
    status: str
    time_to_first_success_s: float | None
    goal_reaches: int
    final_orientation_error_rad: float
    min_orientation_error_rad: float


def summarize(trials: list[TrialOutcome]) -> dict[str, float | int | None]:
    """Return the same aggregate fields as the source Wuji evaluator."""
    if not trials:
        raise ValueError("At least one evaluation trial is required")
    times = [trial.time_to_first_success_s for trial in trials]
    valid_times = [value for value in times if value is not None]
    count = len(trials)
    return {
        "num_trials": count,
        "success_rate": sum(trial.status == "success" for trial in trials) / count,
        "drop_rate": sum(trial.status == "drop" for trial in trials) / count,
        "timeout_rate": sum(trial.status == "timeout" for trial in trials) / count,
        "mean_goal_reaches": float(np.mean([trial.goal_reaches for trial in trials])),
        "mean_time_to_first_success_s": (float(np.mean(valid_times)) if valid_times else None),
        "mean_min_orientation_error_rad": float(
            np.mean([trial.min_orientation_error_rad for trial in trials])
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wuji-eval")
    parser.add_argument("--checkpoint-file", required=True, type=Path)
    parser.add_argument(
        "--task",
        choices=("WujiHand_Reorient", "WujiHand_Reorient_Light"),
        default="WujiHand_Reorient",
    )
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--trial-timeout", type=float, default=14.0)
    parser.add_argument("--success-threshold", type=float, default=0.2)
    parser.add_argument("--success-hold-steps", type=int, default=5)
    parser.add_argument("--drop-height", type=float, default=-0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record the evaluated trials to an mp4 with the goal-pose overlay",
    )
    parser.add_argument(
        "--video-output",
        type=Path,
        help="Recording output path (default: <checkpoint dir>/eval_video.mp4)",
    )
    return parser


def _require_record_capabilities(env: ManagerBasedRlEnv) -> None:
    """Fail fast when --record is requested on an env without playback support."""
    capabilities = getattr(env, "play_capabilities", None)
    if capabilities is None or not capabilities.supports_physics_state_playback:
        raise RuntimeError(
            "wuji-eval --record requires physics-state playback support; "
            f"{type(env).__name__} does not advertise it"
        )
    if not capabilities.supports_debug_overlay:
        raise RuntimeError(
            "wuji-eval --record draws the goal-pose overlay; "
            f"{type(env).__name__} does not advertise debug overlay support"
        )


def _sample_separated_goal(rng: np.random.Generator, current: np.ndarray) -> np.ndarray:
    for _ in range(1000):
        goal = random_quaternions(rng, 1)[0]
        if angle_error(goal[None], current[None])[0] >= np.pi / 2:
            return goal
    raise RuntimeError("Failed to sample an SO(3) goal at least 90 degrees away")


def _refresh_policy_observation(
    env: ManagerBasedRlEnv, wrapper: WujiWrapper, *, reset_history: bool
):
    """Refresh history after a manually selected trial goal."""
    env_ids = np.array([0], dtype=np.int32) if reset_history else None
    if env_ids is not None:
        env.observation_manager.reset(env_ids)
    groups = env.observation_manager.compute(update_history=True, env_ids=env_ids)
    env_cfg = cast(Any, env.cfg)
    policy_group = groups[env_cfg.policy_observation_group]
    critic_group = groups[env_cfg.critic_observation_group]
    if not isinstance(policy_group, np.ndarray) or not isinstance(critic_group, np.ndarray):
        raise TypeError("Wuji evaluation requires concatenated policy and critic observations")
    assert env.state is not None
    env.state.obs["obs"][:] = policy_group
    env.state.obs["critic"][:] = critic_group
    return wrapper.get_observations()


def run(args: argparse.Namespace) -> dict[str, object]:
    if not args.checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_file}")
    if args.num_trials <= 0 or args.trial_timeout <= 0 or args.success_hold_steps <= 0:
        raise ValueError("Trial count, timeout and hold steps must be positive")
    if not 0 < args.success_threshold <= np.pi:
        raise ValueError("Success threshold must be in (0, pi]")

    wp.init()
    wp.set_device("cuda:0")
    cfg = compose_task(
        args.task,
        overrides=["algo.num_envs=1", "training.play_only=true"],
    )
    override = BackendAdapter(
        cfg, root_dir=Path.cwd(), algo_name="ppo"
    ).build_play_env_cfg_override()
    # Trial outcomes own timeout/drop decisions. Environment auto-termination
    # would apply the training cage definition and change the source protocol.
    override["terminations"] = {}
    env = cast(
        ManagerBasedRlEnv,
        registry.make(
            cfg.training.task_name,
            sim_backend="mjwarp",
            env_cfg_override=override,
            num_envs=1,
        ),
    )
    env.set_autoreset(False)
    wrapper = WujiWrapper(env, device="cuda:0")
    rng = np.random.default_rng(args.seed)
    trials: list[TrialOutcome] = []
    record_video: str | None = None
    try:
        session: SnapshotPlaybackSession | None = None
        if args.record:
            _require_record_capabilities(env)
            session = SnapshotPlaybackSession(env, overlay_getter=goal_overlay_getter(env))
        runner = TrainingStateOnPolicyRunner(
            wrapper, normalize_ppo_train_cfg(algo_config_dict(cfg)), device="cuda:0"
        )
        runner.load(str(args.checkpoint_file), map_location="cuda:0")
        policy = runner.get_inference_policy(device="cuda:0")
        obs, _ = wrapper.reset()
        reset_before_trial = True
        state = task(env)
        state.cfg.success_threshold = args.success_threshold
        state.cfg.success_hold_steps = args.success_hold_steps
        drop_reference_z = float(state.cube.data.root_link_pos_w[0, 2])
        for trial_idx in range(args.num_trials):
            did_reset = reset_before_trial
            if did_reset:
                obs, _ = wrapper.reset()
                drop_reference_z = float(task(env).cube.data.root_link_pos_w[0, 2])
            state = task(env)
            current = state.cube_tag()[1][0]
            state.goal[0] = _sample_separated_goal(rng, current)
            state.hold[0] = 0
            state.goal_timer[0] = 0
            state.window[0] = 0
            state.success[0] = False
            obs = _refresh_policy_observation(env, wrapper, reset_history=did_reset)

            timeout_steps = int(np.ceil(args.trial_timeout / env.step_dt))
            min_error = float("inf")
            final_error = float("inf")
            goal_reaches = 0
            first_success: float | None = None
            status = "timeout"
            hold_steps = 0
            with torch.inference_mode():
                if session is not None:
                    session.snapshot()
                for step in range(1, timeout_steps + 1):
                    actions = policy(obs)
                    obs, _, _, _ = wrapper.step(actions)
                    if session is not None:
                        session.snapshot()
                    final_error = float(state.error()[0])
                    min_error = min(min_error, final_error)
                    hold_steps = hold_steps + 1 if final_error < args.success_threshold else 0
                    if hold_steps >= args.success_hold_steps:
                        goal_reaches += 1
                        first_success = step * env.step_dt
                        status = "success"
                        break
                    if float(state.cube.data.root_link_pos_w[0, 2]) < (
                        drop_reference_z + args.drop_height
                    ):
                        status = "drop"
                        break
            reset_before_trial = status != "success"
            trials.append(
                TrialOutcome(
                    trial_idx=trial_idx,
                    status=status,
                    time_to_first_success_s=first_success,
                    goal_reaches=goal_reaches,
                    final_orientation_error_rad=final_error,
                    min_orientation_error_rad=min_error,
                )
            )
            print(
                f"trial {trial_idx + 1}/{args.num_trials}: {status}, min_error={min_error:.4f} rad"
            )
        if session is not None:
            video_output = args.video_output or args.checkpoint_file.with_name("eval_video.mp4")
            video_output.parent.mkdir(parents=True, exist_ok=True)
            record_video = session.render_snapshots(
                output_video=video_output,
                camera=CameraCfg(cam_distance=0.6, cam_lookat=(0.0, 0.0, 0.5)),
            )
    finally:
        env.close()

    result: dict[str, object] = {
        "schema": "wuji_unilab_eval",
        "version": 1,
        "task": args.task,
        "physics_backend": "mjwarp",
        "checkpoint": str(args.checkpoint_file.resolve()),
        "protocol": {
            "num_trials": args.num_trials,
            "trial_timeout_s": args.trial_timeout,
            "success_threshold_rad": args.success_threshold,
            "success_hold_steps": args.success_hold_steps,
            "control_dt_s": env.step_dt,
            "minimum_goal_separation_rad": float(np.pi / 2),
            "drop_height_from_reset_m": args.drop_height,
            "reset_on": ["drop", "timeout"],
            "carry_scene_after_success": True,
            "seed": args.seed,
        },
        **summarize(trials),
        "trials": [asdict(trial) for trial in trials],
    }
    if args.record:
        result["record_video"] = record_video
    output = args.output or args.checkpoint_file.with_name("eval_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "trials"}, indent=2))
    print(f"Wrote {output}")
    return result


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()

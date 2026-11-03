"""Deterministic Hand 2 rollout evaluator for smoke runs and checkpoints."""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class Hand2EvalMetrics:
    """Accumulate episode metrics from vectorized Hand 2 transitions.

    The evaluator may run with automatic resets, so command-owned counters
    cannot be read after ``env.step``.  ``update`` consumes the transition
    snapshot, records the terminal episode first, and then starts the next
    episode row.  The class is tensor-only at the step boundary and exposes a
    small JSON-ready summary for scripts.
    """

    num_envs: int
    device: torch.device | str = "cpu"
    _episode_reward: torch.Tensor = field(init=False, repr=False)
    _episode_steps: torch.Tensor = field(init=False, repr=False)
    _episode_goals: torch.Tensor = field(init=False, repr=False)
    _min_orientation: torch.Tensor = field(init=False, repr=False)
    _final_orientation: torch.Tensor = field(init=False, repr=False)
    _total_reward: float = field(init=False, default=0.0)
    _total_steps: int = field(init=False, default=0)
    _goal_reaches: int = field(init=False, default=0)
    _completed_episodes: int = field(init=False, default=0)
    _success_episodes: int = field(init=False, default=0)
    _drop_episodes: int = field(init=False, default=0)
    _timeout_episodes: int = field(init=False, default=0)
    _episode_reward_sum: float = field(init=False, default=0.0)
    _episode_length_sum: int = field(init=False, default=0)
    _episode_goal_sum: int = field(init=False, default=0)
    _min_orientation_sum: float = field(init=False, default=0.0)
    _final_orientation_sum: float = field(init=False, default=0.0)
    _orientation_sum: float = field(init=False, default=0.0)
    _orientation_samples: int = field(init=False, default=0)
    _orientation_values: list[float] = field(init=False, default_factory=list, repr=False)
    _target_progress_sum: float = field(init=False, default=0.0)
    _target_progress_samples: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self._episode_reward = torch.zeros(self.num_envs, device=self.device)
        self._episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_goals = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._min_orientation = torch.full(
            (self.num_envs,), float("inf"), device=self.device
        )
        self._final_orientation = torch.full(
            (self.num_envs,), float("nan"), device=self.device
        )

    @staticmethod
    def _batch(value: Any, *, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=device)
        if tensor.ndim == 0:
            tensor = tensor.expand(1)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor.reshape(-1)

    def update(self, reward: torch.Tensor, transition: dict[str, Any]) -> None:
        """Consume one vectorized transition, including terminal rows."""
        reward = self._batch(reward, device=self._episode_reward.device, dtype=torch.float32)
        if reward.numel() != self.num_envs:
            raise ValueError("reward batch does not match num_envs")
        done = self._batch(
            transition["done"], device=reward.device, dtype=torch.bool
        )
        terminated = self._batch(
            transition.get("term", torch.zeros_like(done)),
            device=reward.device,
            dtype=torch.bool,
        )
        timeout = self._batch(
            transition.get("timeout", torch.zeros_like(done)),
            device=reward.device,
            dtype=torch.bool,
        )
        orientation = transition.get("orientation_error")
        if orientation is not None:
            orientation = self._batch(orientation, device=reward.device, dtype=torch.float32)
            if orientation.numel() != self.num_envs:
                raise ValueError("orientation_error batch does not match num_envs")
            finite_orientation = torch.isfinite(orientation)
            self._orientation_sum += float(orientation[finite_orientation].sum().item())
            self._orientation_samples += int(finite_orientation.sum().item())
            self._orientation_values.extend(orientation[finite_orientation].detach().cpu().tolist())
            orientation = torch.nan_to_num(orientation, nan=float("inf"), posinf=float("inf"))
            self._min_orientation = torch.minimum(self._min_orientation, orientation)
            self._final_orientation = orientation
            before = transition.get("orientation_error_before")
            if before is not None:
                before = self._batch(before, device=reward.device, dtype=torch.float32)
                progress = before - orientation
                finite_progress = torch.isfinite(progress)
                self._target_progress_sum += float(progress[finite_progress].sum().item())
                self._target_progress_samples += int(finite_progress.sum().item())

        success = transition.get("success")
        if success is not None:
            success = self._batch(success, device=reward.device, dtype=torch.bool)
        else:
            success = torch.zeros_like(done)
        # ``success`` is an event emitted by Hand2ReorientCommand.  A
        # fallback goal-count delta keeps this helper useful with older
        # transition producers.
        goal_count = transition.get("goal_count")
        if goal_count is not None:
            goal_count = self._batch(goal_count, device=reward.device, dtype=torch.long)
            delta = torch.clamp(goal_count - self._episode_goals, min=0)
            success = success | (delta > 0)
            self._episode_goals.copy_(torch.maximum(self._episode_goals, goal_count))
            goal_events = delta
        else:
            goal_events = success.to(torch.long)
            self._episode_goals.add_(goal_events)
        self._goal_reaches += int(goal_events.sum().item())

        self._episode_reward += reward
        self._episode_steps += 1
        self._total_reward += float(reward.sum().item())
        self._total_steps += self.num_envs

        completed = done.nonzero(as_tuple=False).flatten()
        if completed.numel() == 0:
            return
        dropped = terminated & ~timeout
        for index in completed.tolist():
            self._completed_episodes += 1
            goals = int(self._episode_goals[index].item())
            self._success_episodes += int(goals > 0)
            self._drop_episodes += int(bool(dropped[index]))
            self._timeout_episodes += int(bool(timeout[index]))
            self._episode_reward_sum += float(self._episode_reward[index].item())
            self._episode_length_sum += int(self._episode_steps[index].item())
            self._episode_goal_sum += goals
            self._min_orientation_sum += float(self._min_orientation[index].item())
            self._final_orientation_sum += float(self._final_orientation[index].item())
        self._episode_reward[completed] = 0
        self._episode_steps[completed] = 0
        self._episode_goals[completed] = 0
        self._min_orientation[completed] = float("inf")
        self._final_orientation[completed] = float("nan")

    def summary(self) -> dict[str, float | int | None]:
        """Return scalar metrics suitable for logs and JSON serialization."""
        completed = self._completed_episodes
        active_min = self._min_orientation[torch.isfinite(self._min_orientation)]
        active_final = self._final_orientation[torch.isfinite(self._final_orientation)]
        mean_min_orientation = (
            self._min_orientation_sum / completed
            if completed
            else (float(active_min.mean().item()) if active_min.numel() else None)
        )
        mean_final_orientation = (
            self._final_orientation_sum / completed
            if completed
            else (float(active_final.mean().item()) if active_final.numel() else None)
        )
        orientation_p95 = (
            float(torch.tensor(self._orientation_values).quantile(0.95).item())
            if self._orientation_values
            else None
        )
        return {
            "steps": self._total_steps,
            "episodes": completed,
            "total_reward": self._total_reward,
            "mean_reward_per_step": self._total_reward / max(self._total_steps, 1),
            "mean_episode_reward": (
                self._episode_reward_sum / completed if completed else None
            ),
            "success_rate": self._success_episodes / completed if completed else None,
            "goal_reaches": self._goal_reaches,
            "mean_goal_reaches": self._episode_goal_sum / completed if completed else None,
            "cage_drop_count": self._drop_episodes,
            "cage_drop_rate": self._drop_episodes / completed if completed else None,
            "timeout_count": self._timeout_episodes,
            "timeout_rate": self._timeout_episodes / completed if completed else None,
            "mean_episode_length_steps": (
                self._episode_length_sum / completed if completed else None
            ),
            "mean_orientation_error_rad": (
                self._orientation_sum / self._orientation_samples
                if self._orientation_samples
                else None
            ),
            "orientation_error_p95_rad": orientation_p95,
            "mean_target_progress_rad": (
                self._target_progress_sum / self._target_progress_samples
                if self._target_progress_samples
                else None
            ),
            "mean_min_orientation_error_rad": mean_min_orientation,
            "mean_final_orientation_error_rad": mean_final_orientation,
            "active_episode_steps_mean": float(self._episode_steps.float().mean().item()),
        }


def _run_hand2_interactive(
    *,
    env: Any,
    initialize,
    step,
    num_steps: int,
    frame_state_getter,
    debug_overlay_getter=None,
    camera_kwargs: dict[str, Any] | None = None,
) -> None:
    """Run a local MuJoCo window with an explicit keyboard reset.

    UniSim's generic mjwarp playback loop intentionally has no task-specific
    key bindings.  Hand2 debugging needs a reset that does not close the
    window, so keep the callback as a small evaluator-owned loop.  ``R`` only
    sets an event from the viewer callback; the actual environment reset runs
    on the physics thread between steps.
    """
    import mujoco
    import mujoco.viewer
    from unisim.backend.base import CameraCfg
    from unisim.backend.mjwarp.playback import _inject_interactive_debug_overlays

    model_or_path = env.get_playback_model(0)
    if isinstance(model_or_path, (str, Path)):
        model = mujoco.MjModel.from_xml_path(str(model_or_path))
    else:
        model = model_or_path
    data = mujoco.MjData(model)
    camera = CameraCfg.from_kwargs(camera_kwargs)
    reset_requested = threading.Event()

    def on_key(key: int) -> None:
        if int(key) in (ord("r"), ord("R")):
            reset_requested.set()

    def update(snapshot) -> None:
        row = np.asarray(snapshot(), dtype=np.float32)[0]
        data.time = float(row[0])
        data.qpos[:] = row[1 : 1 + model.nq]
        data.qvel[:] = row[1 + model.nq : 1 + model.nq + model.nv]
        if model.nmocap:
            base = 1 + model.nq + model.nv
            if row.shape[0] < base + 7 * model.nmocap:
                raise ValueError("Hand2 interactive snapshot is missing the mocap pose tail")
            tail = row[base : base + 7 * model.nmocap]
            data.mocap_pos[:] = tail[: 3 * model.nmocap].reshape(model.nmocap, 3)
            data.mocap_quat[:] = tail[3 * model.nmocap :].reshape(model.nmocap, 4)
        mujoco.mj_forward(model, data)

    history = initialize()
    update(frame_state_getter)
    try:
        viewer = mujoco.viewer.launch_passive(model, data, key_callback=on_key)
    except Exception as exc:
        raise RuntimeError(
            "Hand2 interactive playback could not open the MuJoCo viewer; "
            "check GLFW/DISPLAY access."
        ) from exc
    with viewer:
        with viewer.lock():
            viewer.cam.distance = camera.cam_distance
            viewer.cam.elevation = camera.cam_elevation
            viewer.cam.azimuth = camera.cam_azimuth
            if camera.cam_lookat is not None:
                viewer.cam.lookat[:] = camera.cam_lookat
        mesh_id_cache: dict[str, int] = {}
        mesh_mat_cache: dict[str, int] = {}
        if debug_overlay_getter is not None and viewer.user_scn is None:
            raise RuntimeError("Hand2 goal overlay requires viewer.user_scn support")
        viewer.sync()
        print("Hand2 playback: press R to reset all environments; close the window to stop.", flush=True)
        ctrl_dt = float(getattr(getattr(env, "cfg", None), "ctrl_dt", 0.02))
        for _ in range(num_steps):
            if not viewer.is_running():
                break
            started = time.monotonic()
            if reset_requested.is_set():
                reset_requested.clear()
                history = initialize()
            else:
                history = step(history)
            update(frame_state_getter)
            with viewer.lock():
                if debug_overlay_getter is not None:
                    _inject_interactive_debug_overlays(
                        user_scn=viewer.user_scn,
                        overlays=debug_overlay_getter(),
                        world=0,
                        num_envs=env.num_envs,
                        model=model,
                        mesh_id_cache=mesh_id_cache,
                        mesh_mat_cache=mesh_mat_cache,
                    )
            viewer.sync()
            time.sleep(max(0.0, ctrl_dt - (time.monotonic() - started)))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mjwarp", "mujoco"), default="mujoco")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--render",
        choices=("none", "record", "interactive"),
        default="none",
        help="visualize the rollout, or export it as a video",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=Path("hand2_rollout.mp4"),
        help="output path used with --render record",
    )
    parser.add_argument(
        "--show-goal",
        action="store_true",
        help="overlay the target cube pose and axes during rendered playback",
    )
    # The Hand2 scene is calibrated around the palm/cube at z≈0.56 m.  The
    # generic renderer's origin-centered camera leaves the hand tiny or below
    # the frame, so use a close three-quarter view by default.
    parser.add_argument("--camera-distance", type=float, default=0.65)
    parser.add_argument("--camera-elevation", type=float, default=-5.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument(
        "--camera-lookat",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(-0.05, 0.0, 0.62),
    )
    args = parser.parse_args(argv)
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    from hydra import compose, initialize_config_dir
    from unilab.base import registry
    from unilab.base.config_adapter import EnvConfigAdapter
    from unilab.training import ensure_registries

    from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_TRANSITION_KEY

    # The evaluator is also a standalone console entry point.  Importing only
    # ``unilab.base.registry`` does not load this package's entry-point module
    # in every UniLab installation, so explicitly initialize registrations
    # before asking the registry to construct the task.
    ensure_registries()

    # evaluate.py lives below ``src/unisupermimic/tasks/dex_reorient/hand2``;
    # the packaged Hydra root is ``src/unisupermimic/conf/supermimic``.  Use
    # the Hand2 training config so playback has the same EMA/action and goal
    # timing parameters as the learner; composing the generic G1 config here
    # silently changes those controller semantics.
    conf = Path(__file__).parents[3] / "conf/supermimic"
    with initialize_config_dir(config_dir=str(conf), version_base="1.3"):
        cfg = compose(
            "config_hand2",
            overrides=[
                f"task=dex_hand2_reorient/{args.backend}",
                f"training.sim_backend={args.backend}",
                f"algo.num_envs={args.num_envs}",
            ],
        )
    override = EnvConfigAdapter(
        cfg, root_dir=Path.cwd(), algo_name="supermimic_hand2"
    ).build_task_env_cfg_override()
    env = registry.make(
        "DexHand2Reorient",
        sim_backend=args.backend,
        env_cfg_override=override,
        num_envs=args.num_envs,
    )
    # Evaluation must expose terminal states to the interactive viewer.  The
    # keyboard reset is handled by the evaluator, so a backend-level autoreset
    # would hide the state we are trying to inspect.
    set_autoreset = getattr(env, "set_autoreset", None)
    if callable(set_autoreset):
        set_autoreset(False)
    try:
        import torch

        from unisupermimic.rl.algos.supermimic.hand2_runner import (
            Hand2Actor,
            Hand2SuperMimicRunner,
        )
        from unisupermimic.tasks.dex_reorient.hand2.overlay import goal_overlay_getter

        checkpoint = None
        actor = None
        if args.checkpoint is not None:
            # A training checkpoint also contains the full replay buffer
            # (about 3 GiB for the source 1024-env recipe).  Evaluation only
            # needs the actor weights, so keep the serialized payload on CPU
            # and copy that module to the runtime device below.
            checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            saved_cfg = checkpoint.get("config", {})
            hidden = tuple(saved_cfg.get("policy_hidden", (512, 256, 128)))
            actor = Hand2Actor(hidden).to(env.device)
            actor.load_state_dict(checkpoint["modules"]["actor"])
            actor.eval()
        num_envs = int(args.num_envs)
        metrics = Hand2EvalMetrics(num_envs, device=env.device)
        playback_done = False

        def reset_history() -> torch.Tensor:
            nonlocal num_envs, metrics, playback_done
            playback_done = False
            observation = env.reset()[0].get("policy")
            if observation is None:
                observation = env.observation_manager.compute(update_history=False).get("policy")
            if observation is None:
                raise RuntimeError("Hand2 evaluator could not obtain policy observation")
            if observation.ndim == 1:
                observation = observation.unsqueeze(0)
            num_envs = int(observation.shape[0])
            metrics = Hand2EvalMetrics(num_envs, device=env.device)
            # UniLab emits the three-frame policy group term-major (all
            # history slots for joint position, then all slots for target
            # error, ...).  Convert it to chronological complete frames.
            frame = Hand2SuperMimicRunner._single_frame_observation(observation)
            return frame[:, None].expand(-1, 3, -1).clone()

        # Keep the terminal state visible in the interactive viewer.  The
        # regular evaluator and video recorder still reset finished rows so
        # they can run for a fixed number of steps.
        def advance(history: torch.Tensor, *, reset_done: bool = True) -> torch.Tensor:
            nonlocal playback_done
            if not reset_done and playback_done:
                return history
            if actor is None:
                action = torch.zeros(
                    (num_envs, env.action_manager.total_action_dim), device=env.device
                )
            else:
                with torch.no_grad():
                    action = actor(history)
            state = env.step(action)
            reward = state.reward.to(env.device)
            metrics.update(reward, state.info[SUPERMIMIC_TRANSITION_KEY])
            observation = env.observation_manager.compute(update_history=False).get("policy")
            if observation is None:
                raise RuntimeError("Hand2 evaluator lost policy observation")
            if observation.ndim == 1:
                observation = observation.unsqueeze(0)
            frame = Hand2SuperMimicRunner._single_frame_observation(observation)
            # The training runner disables autoreset and explicitly resets
            # only terminal rows after publishing their transition.  Mirror
            # that lifecycle here so a long local playback does not continue
            # stepping a finished episode indefinitely.
            done = state.info[SUPERMIMIC_TRANSITION_KEY]["done"].to(env.device)
            if not reset_done and bool(done.any()):
                playback_done = True
                print("Hand2 episode ended; press R to reset.", flush=True)
            elif reset_done and bool(done.any()):
                reset_ids = done.nonzero(as_tuple=False).flatten()
                reset_obs, _ = env.reset(env_ids=reset_ids)
                reset_packed = reset_obs.get("policy", reset_obs.get("obs"))
                if reset_packed is None:
                    raise RuntimeError("Hand2 playback reset did not return policy/obs")
                reset_frame = Hand2SuperMimicRunner._single_frame_observation(
                    reset_packed.to(env.device)
                )
                frame = torch.where(done[:, None], reset_frame, frame)
            return torch.cat((history[:, 1:], frame[:, None]), dim=1)

        playback_model = env.get_playback_model(0)
        playback_data = None
        if isinstance(playback_model, (str, Path)):
            import mujoco

            playback_model = mujoco.MjModel.from_xml_path(str(playback_model))
        if int(playback_model.nmocap) > 0:
            import mujoco

            playback_data = mujoco.MjData(playback_model)
        playback_width = 1 + int(playback_model.nq) + int(playback_model.nv) + 7 * int(
            playback_model.nmocap
        )

        def playback_state() -> np.ndarray:
            """Return the backend snapshot in the renderer's full layout.

            The current MuJoCo host backend exposes ``[time, qpos, qvel]``
            for this mocap-rooted scene while its visual playback model also
            expects the mocap pose tail.  The palm is fixed in this task, so
            the compiled model pose is the correct missing tail.  mjwarp
            already publishes the complete layout and passes through here.
            """
            state = np.asarray(env.get_physics_state_snapshot(), dtype=np.float32)
            if state.ndim != 2:
                raise ValueError(f"Hand2 playback snapshot must be rank 2, got {state.shape}")
            if state.shape[1] == playback_width:
                return state.copy()
            base_width = 1 + int(playback_model.nq) + int(playback_model.nv)
            if (
                state.shape[1] == base_width
                and int(playback_model.nmocap) > 0
            ):
                assert playback_data is not None
                mocap_pos = np.asarray(playback_data.mocap_pos, dtype=np.float32).reshape(-1)
                mocap_quat = np.asarray(playback_data.mocap_quat, dtype=np.float32).reshape(-1)
                tail = np.broadcast_to(
                    np.concatenate((mocap_pos, mocap_quat)),
                    (state.shape[0], 7 * int(playback_model.nmocap)),
                )
                return np.concatenate((state, tail), axis=1).copy()
            raise ValueError(
                f"Hand2 playback snapshot width {state.shape[1]} does not match "
                f"renderer width {playback_width}"
            )

        overlay = goal_overlay_getter(env) if args.show_goal and args.render != "none" else None
        camera_kwargs = {
            "cam_distance": args.camera_distance,
            "cam_elevation": args.camera_elevation,
            "cam_azimuth": args.camera_azimuth,
        }
        if args.camera_lookat is not None:
            camera_kwargs["cam_lookat"] = tuple(args.camera_lookat)
        if args.render == "record":
            output = args.output_video.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            rendered = env.run_playback(
                initialize=reset_history,
                step=advance,
                num_steps=args.steps,
                output_video=str(output),
                headless=True,
                record_video=True,
                frame_state_getter=playback_state,
                debug_overlay_getter=overlay,
                camera_kwargs=camera_kwargs,
            )
            if rendered is not None:
                print(f"video={rendered}", flush=True)
        elif args.render == "interactive":
            _run_hand2_interactive(
                env=env,
                initialize=reset_history,
                # Keep terminal states visible until the user presses R;
                # ``advance`` otherwise performs the normal evaluator's
                # selected-row reset after publishing a terminal transition.
                step=lambda history: advance(history, reset_done=False),
                num_steps=args.steps,
                frame_state_getter=playback_state,
                debug_overlay_getter=overlay,
                camera_kwargs=camera_kwargs,
            )
        else:
            history = reset_history()
            for _ in range(args.steps):
                history = advance(history)
        label = "zero_action" if actor is None else str(args.checkpoint)
        summary = metrics.summary()
        print(
            f"DexHand2Reorient backend={args.backend} steps={args.steps} "
            f"num_envs={args.num_envs} policy={label} "
            f"mean_reward={summary['mean_reward_per_step']:.6g} "
            f"episodes={summary['episodes']} success_rate={summary['success_rate']} "
            f"goal_reaches={summary['goal_reaches']} "
            f"cage_drops={summary['cage_drop_count']} "
            f"mean_episode_length={summary['mean_episode_length_steps']} "
            f"mean_min_orientation_error={summary['mean_min_orientation_error_rad']} "
            f"orientation_error_p95={summary['orientation_error_p95_rad']} "
            f"target_progress={summary['mean_target_progress_rad']}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()

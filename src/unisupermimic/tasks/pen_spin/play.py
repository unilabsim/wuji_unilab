"""Actor-only Pen Spin playback adapted from Dex-Supermimic's cli.play.

The source play profile supplies clean observations, fixed hardware pen
dimensions and a held first reference frame. Only the environment/viewer
boundary changes for UniLab; no learner or training output is constructed.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
from pathlib import Path

import numpy as np
import torch

from unisupermimic.rl.algos.dex_supermimic.common import require_finite
from unisupermimic.rl.algos.dex_supermimic.env_bridge import SourceEnvBridge
from unisupermimic.rl.algos.dex_supermimic.networks import Actor


def load_actor(checkpoint_path: Path):
    """Keep the full checkpoint on CPU and restore its actor and normalization."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    if config.get("task") != "hand2-pen-spin" or config.get("algorithm") != "world_model":
        raise ValueError("Pen playback requires a Pen Spin SuperMimic checkpoint")
    if checkpoint.get("format_version") != 3 or config.get("control_hz") != 50:
        raise ValueError("Unsupported Pen checkpoint format or control frequency")
    state = checkpoint["modules"]["actor"]
    width = state["norm._mean"].shape[-1]
    if width != 155:
        raise ValueError(f"Pen actor must consume 155 features, got {width}")
    actor = Actor(width)
    actor.load_state_dict(state, strict=True)
    actor.eval()
    # Discard the world model and optimizers; playback only needs this metadata.
    metadata = {key: checkpoint.get(key) for key in ("config", "joint_names", "iteration")}
    return actor, metadata


def reference_paths(checkpoint_path: Path, metadata: dict, recipe_path=None, clips_root=None):
    """Recover the actual Hydra motion paths saved alongside a training run."""
    config = metadata["config"]
    saved_motion = {}
    saved_environment = checkpoint_path.parent / "environment_config.json"
    if saved_environment.is_file():
        saved_motion = json.loads(saved_environment.read_text())["commands"]["motion"]
    return (
        recipe_path or saved_motion.get("recipe_path") or config.get("recipe_path"),
        clips_root or saved_motion.get("clips_root") or config.get("clips_root"),
    )


class PenPlayback:
    """Deterministic source policy on the real UniLab Pen environment."""

    def __init__(self, env, actor, *, manual_reset=False):
        self.env = SourceEnvBridge(env)
        self.actor = actor.to(env.device).eval()
        self.manual_reset = manual_reset
        self.obs = None
        self.ended = False
        self.steps = self.episodes = self.failures = self.completed_clips = 0
        self.reward_sum = 0.0

    @torch.no_grad()
    def reset(self):
        self.obs = self.env.reset()[0]["policy"]
        if self.obs.shape != (1, 155):
            raise ValueError(f"Unexpected Pen policy observation shape: {tuple(self.obs.shape)}")
        self.ended = False

    @torch.no_grad()
    def advance(self):
        if self.ended:
            return False
        require_finite("Pen playback observation", self.obs)
        action = self.actor(self.obs)
        require_finite("Pen playback action", action)
        observations, reward, terminated, truncated, _ = self.env.step(action)
        self.obs = observations["policy"]
        self.steps += 1
        self.reward_sum += float(reward.mean())
        done = terminated | truncated
        if bool(done.any()):
            self.episodes += 1
            self.failures += int(bool(terminated.any()))
            reference_end = self.env.termination_manager.get_term("reference_end")
            self.completed_clips += int(bool(reference_end.any()) and not bool(terminated.any()))
            if self.manual_reset:
                self.ended = True
                print("Episode ended; press R to reset.", flush=True)
            else:
                self.reset()
        return True

    def summary(self):
        return dict(steps=self.steps, episodes=self.episodes, failures=self.failures,
                    completed_clips=self.completed_clips,
                    mean_reward=self.reward_sum / max(self.steps, 1))


def sync_viewer_data(env, model, data) -> None:
    """Copy the public physics snapshot into a separate visual mjData."""
    import mujoco

    row = np.asarray(env.get_physics_state_snapshot(), dtype=np.float64)[0]
    base = 1 + model.nq + model.nv
    if row.size not in (base, base + 7 * model.nmocap):
        raise ValueError("Pen playback snapshot does not match the visual model")
    data.time = row[0]
    data.qpos[:] = row[1:1 + model.nq]
    data.qvel[:] = row[1 + model.nq:base]
    if model.nmocap and row.size > base:
        data.mocap_pos[:] = row[base:base + 3 * model.nmocap].reshape(-1, 3)
        data.mocap_quat[:] = row[base + 3 * model.nmocap:].reshape(-1, 4)
    # Host snapshots omit mocap. The source play wrist is fixed at the same
    # calibrated pose as the model defaults, which mjData already contains.
    mujoco.mj_forward(model, data)


def run_viewer(session, args):
    import mujoco
    import mujoco.viewer

    from .play_config import prepare_playback_model
    from .play_overlay import PenReferenceOverlay

    env = session.env
    model = prepare_playback_model(env)
    data = mujoco.MjData(model)
    overlay = PenReferenceOverlay(model, env.command_manager.get_term("motion")) if args.show_reference else None
    reset_requested, pause_requested, step_requested = (threading.Event() for _ in range(3))

    def on_key(key):
        if key in (ord("r"), ord("R")):
            reset_requested.set()
        elif key == ord(" "):
            pause_requested.set()
        elif key in (ord("n"), ord("N")):
            step_requested.set()

    session.reset()
    sync_viewer_data(env, model, data)
    paused = args.start_paused
    print("Pen playback: R reset, Space pause/resume, N single step; close window to exit.", flush=True)
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = args.camera_lookat
            viewer.cam.distance = args.camera_distance
            viewer.cam.elevation = args.camera_elevation
            viewer.cam.azimuth = args.camera_azimuth
        while viewer.is_running() and (args.steps == 0 or session.steps < args.steps):
            started = time.monotonic()
            if pause_requested.is_set():
                pause_requested.clear()
                paused = not paused
            if reset_requested.is_set():
                reset_requested.clear()
                session.reset()
            elif (not paused or step_requested.is_set()) and not session.ended:
                session.advance()
            step_requested.clear()
            sync_viewer_data(env, model, data)
            if overlay is not None:
                with viewer.lock():
                    overlay.update(viewer.user_scn, data)
            viewer.sync()
            time.sleep(max(0.0, float(env.step_dt) / args.speed - (time.monotonic() - started)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backend", choices=("mjwarp", "mujoco"), default="mjwarp")
    parser.add_argument("--render", choices=("interactive", "none"), default="interactive")
    parser.add_argument("--steps", type=int, default=0, help="Control steps; 0 keeps the viewer open")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--recipe-path", type=str, default=None)
    parser.add_argument("--clips-root", type=str, default=None)
    parser.add_argument("--show-reference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--manual-reset", action="store_true", help="Hold terminal state until R is pressed")
    parser.add_argument("--start-paused", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--camera-lookat", type=float, nargs=3, default=(0.0, 0.0, 0.5))
    parser.add_argument("--camera-distance", type=float, default=0.45)
    parser.add_argument("--camera-elevation", type=float, default=-25.0)
    parser.add_argument("--camera-azimuth", type=float, default=-40.0)
    args = parser.parse_args(argv)
    if args.steps < 0 or args.speed <= 0 or args.camera_distance <= 0:
        parser.error("Require steps >= 0, speed > 0 and camera-distance > 0")
    if args.render == "none" and (args.steps == 0 or args.manual_reset):
        parser.error("Headless playback requires positive --steps and automatic resets")
    if args.backend == "mjwarp" and not torch.cuda.is_available():
        raise RuntimeError("MJWarp requires CUDA; choose --backend mujoco for CPU playback")
    from unilab.base.config_adapter import EnvConfigAdapter, create_env
    from unilab.training import ensure_registries

    from .play_config import build_play_config

    checkpoint = args.checkpoint.expanduser().resolve()
    actor, metadata = load_actor(checkpoint)
    seed = args.seed if args.seed is not None else int(metadata["config"].get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    recipe, clips = reference_paths(checkpoint, metadata, args.recipe_path, args.clips_root)
    ensure_registries()
    cfg = build_play_config(backend=args.backend, seed=seed, recipe_path=recipe, clips_root=clips)
    override = EnvConfigAdapter(cfg, root_dir=Path.cwd(), algo_name="supermimic").build_task_env_cfg_override()
    env = create_env(cfg, num_envs=1, env_cfg_override=override)
    try:
        if tuple(metadata["joint_names"]) != tuple(env.scene["robot"].joint_names):
            raise ValueError("Pen checkpoint joint order mismatch")
        session = PenPlayback(env, actor, manual_reset=args.manual_reset)
        print(f"Pen Spin SuperMimic checkpoint={checkpoint} iteration={metadata['iteration']} backend={args.backend}", flush=True)
        if args.render == "interactive":
            run_viewer(session, args)
        else:
            session.reset()
            for _ in range(args.steps):
                session.advance()
        print(json.dumps(session.summary(), indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()

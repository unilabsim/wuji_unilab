"""Actor-only Hand1 SuperMimic playback in a local MuJoCo window.

Uses the source Hand1 play profile and the same actor-only playback pattern
as Pen Spin. No learner, optimizer, video or training output is created.
"""

from __future__ import annotations

import argparse
import copy
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
    """Restore the Hand1 actor and its saved normalization on CPU."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    if config.get("task") != "hand1-reorient" or config.get("algorithm") != "world_model":
        raise ValueError("Hand1 playback requires a Hand1 SuperMimic checkpoint")
    if checkpoint.get("format_version") != 3 or config.get("control_hz") not in (20, 50):
        raise ValueError("Unsupported Hand1 checkpoint format or control frequency")
    state = checkpoint["modules"]["actor"]
    width = state["norm._mean"].shape[-1]
    if width != 207:
        raise ValueError(f"Hand1 actor must consume 207 features, got {width}")
    actor = Actor(width)
    actor.load_state_dict(state, strict=True)
    actor.eval()
    metadata = {key: checkpoint.get(key) for key in ("config", "joint_names", "iteration")}
    return actor, metadata


class Hand1Playback:
    """Run the saved policy directly, without exploration or norm updates."""

    def __init__(self, env, actor, *, manual_reset=False):
        self.env = SourceEnvBridge(env)
        self.actor = actor.to(env.device).eval()
        self.command = env.command_manager.get_term("reorient")
        self.manual_reset = manual_reset
        self.obs = None
        self.ended = False
        self.steps = self.episodes = self.failures = self.goal_reaches = 0
        self.reward_sum = 0.0
        self._episode_goals = 0

    @torch.no_grad()
    def reset(self):
        self.obs = self.env.reset()[0]["policy"]
        if self.obs.shape != (1, 207):
            raise ValueError(f"Unexpected Hand1 policy observation shape: {tuple(self.obs.shape)}")
        self._episode_goals = 0
        self.ended = False

    @torch.no_grad()
    def advance(self):
        if self.ended:
            return False
        require_finite("Hand1 playback observation", self.obs)
        action = self.actor(self.obs)
        require_finite("Hand1 playback action", action)
        observations, reward, terminated, truncated, _ = self.env.step(action)
        self.obs = observations["policy"]
        self.steps += 1
        self.reward_sum += float(reward.mean())
        # Autoreset is disabled; read the terminal command before resetting.
        goals = int(self.command.goal_count[0])
        self.goal_reaches += max(0, goals - self._episode_goals)
        self._episode_goals = goals
        if bool((terminated | truncated).any()):
            self.episodes += 1
            self.failures += int(bool(terminated.any()))
            if self.manual_reset:
                self.ended = True
                print("Episode ended; press R to reset.", flush=True)
            else:
                self.reset()
        return True

    def summary(self):
        return dict(steps=self.steps, episodes=self.episodes, failures=self.failures,
                    goal_reaches=self.goal_reaches,
                    mean_reward=self.reward_sum / max(self.steps, 1))


def sync_viewer_data(env, model, data):
    """Copy physics and the randomized wrist into independent visual data."""
    import mujoco

    row = np.asarray(env.get_physics_state_snapshot(), dtype=np.float64)[0]
    base = 1 + model.nq + model.nv
    if row.size not in (base, base + 7 * model.nmocap):
        raise ValueError("Hand1 playback snapshot does not match the visual model")
    data.time = row[0]
    data.qpos[:] = row[1:1 + model.nq]
    data.qvel[:] = row[1 + model.nq:base]
    if model.nmocap:
        if row.size > base:
            data.mocap_pos[:] = row[base:base + 3 * model.nmocap].reshape(-1, 3)
            data.mocap_quat[:] = row[base + 3 * model.nmocap:].reshape(-1, 4)
        else:
            try:
                position, quaternion = env.get_playback_mocap_state(0)
            except NotImplementedError:
                # Some host backends omit the mocap tail and advertise no
                # mocap playback capability. The Hand1 palm is itself the
                # sole mocap body, so its measured world pose is exact here.
                command = env.command_manager.get_term("reorient")
                mocap_id = int(model.body_mocapid[model.body(command.palm_name).id])
                if model.nmocap != 1 or mocap_id < 0:
                    raise ValueError("Hand1 playback requires one mocap palm") from None
                position, quaternion = command.palm_pose()
                data.mocap_pos[mocap_id] = position[0].detach().cpu().numpy()
                data.mocap_quat[mocap_id] = quaternion[0].detach().cpu().numpy()
            else:
                data.mocap_pos[:] = position
                data.mocap_quat[:] = quaternion
    mujoco.mj_forward(model, data)


def run_viewer(session, args):
    import mujoco
    import mujoco.viewer

    from .play_overlay import Hand1GoalOverlay

    env = session.env
    oracle = env.get_playback_model(0)
    model = (mujoco.MjModel.from_xml_path(str(oracle))
             if isinstance(oracle, (str, Path)) else copy.deepcopy(oracle))
    data = mujoco.MjData(model)
    overlay = Hand1GoalOverlay(model, session.command) if args.show_goal else None
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
    print("Hand1 playback: R reset, Space pause/resume, N single step; close window to exit.", flush=True)
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = args.camera_lookat
            viewer.cam.distance = args.camera_distance
            viewer.cam.elevation = args.camera_elevation
            viewer.cam.azimuth = args.camera_azimuth
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
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
    parser.add_argument("--show-goal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--manual-reset", action="store_true", help="Hold terminal state until R is pressed")
    parser.add_argument("--start-paused", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--camera-lookat", type=float, nargs=3, default=(-0.03, 0.0, 0.58))
    parser.add_argument("--camera-distance", type=float, default=0.5)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--camera-azimuth", type=float, default=90.0)
    args = parser.parse_args(argv)
    if args.steps < 0 or args.speed <= 0 or args.camera_distance <= 0:
        parser.error("Require steps >= 0, speed > 0 and camera-distance > 0")
    if args.render == "none" and (args.steps == 0 or args.manual_reset):
        parser.error("Headless playback requires positive --steps and automatic resets")
    if args.backend == "mjwarp" and not torch.cuda.is_available():
        raise RuntimeError("MJWarp requires CUDA; choose --backend mujoco for CPU playback")
    from unilab.training import ensure_registries

    from .factory import make_hand1_env

    checkpoint = args.checkpoint.expanduser().resolve()
    actor, metadata = load_actor(checkpoint)
    config = metadata["config"]
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    ensure_registries()
    env = make_hand1_env(num_envs=1, backend_type=args.backend,
                         control_hz=int(config["control_hz"]), seed=seed, play=True)
    try:
        if tuple(metadata["joint_names"]) != tuple(env.scene["robot"].joint_names):
            raise ValueError("Hand1 checkpoint joint order mismatch")
        session = Hand1Playback(env, actor, manual_reset=args.manual_reset)
        print(f"Hand1 SuperMimic checkpoint={checkpoint} iteration={metadata['iteration']} backend={args.backend}", flush=True)
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

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Reference command for the migrated Wuji Hand 2 pen-spin task.

The production task consumes the verified ``wuji_pen_spin_clips`` catalog.  A
small deterministic trajectory is available when no catalog is supplied so a
scene/manager smoke can run immediately after installation; passing
``recipe_path`` or ``clips_root`` always selects the strict catalog path.

Trajectory sampling and failure statistics are ported from Dex-SuperMimic's
``tasks/pen_spin/mdp/commands.py``. UniLab adaptation is confined to entity
access, manager reset masks, and scalar-first quaternion helpers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.distributed as dist
from unilab.base.entity import Entity
from unilab.managers import CommandTerm, CommandTermCfg

from .tooling.recipe import resolve_recipe

JOINT_NAMES = tuple(
    f"robot/right_finger{finger}_joint{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
)
LINK_NAMES = ("robot/right_palm_link",) + tuple(
    f"robot/right_finger{finger}_link{link}"
    for finger in range(1, 6)
    for link in range(1, 5)
)
FIXED_WRIST_POSE_W = (
    -0.028110361960901435,
    0.00300004,
    0.4952974018391883,
    0.5416752204197018,
    0.45451947767204376,
    0.45451947767204376,
    0.5416752204197018,
)


def _quat_normalize(value: torch.Tensor) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1e-8)


def _quat_conjugate(value: torch.Tensor) -> torch.Tensor:
    return torch.cat((value[..., :1], -value[..., 1:]), dim=-1)


def _quat_multiply(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = first.unbind(-1)
    w2, x2, y2, z2 = second.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _quat_rotate(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    pure = torch.cat((torch.zeros_like(vector[..., :1]), vector), dim=-1)
    return _quat_multiply(_quat_multiply(quat, pure), _quat_conjugate(quat))[..., 1:]


def _load_reference(
    path: Path, device: torch.device, frame_dt: float
) -> dict[str, torch.Tensor]:
    """Load one source-compatible NPZ trajectory and convert to wrist-local form."""
    with np.load(path, allow_pickle=False) as data:
        required = ("reference_qpos", "wrist_pose_w", "object_pose_w", "link_pos_w")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"pen clip {path} is missing required arrays: {missing}")
        qpos = torch.as_tensor(data["reference_qpos"], dtype=torch.float32, device=device)
        wrist = torch.as_tensor(data["wrist_pose_w"], dtype=torch.float32, device=device)
        obj = torch.as_tensor(data["object_pose_w"], dtype=torch.float32, device=device)
        links = torch.as_tensor(data["link_pos_w"], dtype=torch.float32, device=device)
    if qpos.ndim != 2 or qpos.shape[-1] != 20 or wrist.shape != (len(qpos), 7):
        raise ValueError(f"invalid pen clip shapes in {path}: qpos={tuple(qpos.shape)}, wrist={tuple(wrist.shape)}")
    if len(qpos) < 2:
        raise ValueError(f"pen clip {path} must contain at least two frames")
    if obj.shape != (len(qpos), 7) or links.shape != (len(qpos), 21, 3):
        raise ValueError(f"invalid pen clip object/link shapes in {path}")
    if not np.isfinite(frame_dt) or frame_dt <= 0:
        raise ValueError("pen clip frame_dt must be finite and positive")
    values = (qpos, wrist, obj, links)
    if not all(torch.isfinite(value).all() for value in values):
        raise ValueError(f"pen clip {path} contains non-finite values")
    wrist_norm = torch.linalg.vector_norm(wrist[:, 3:], dim=-1)
    object_norm = torch.linalg.vector_norm(obj[:, 3:], dim=-1)
    if (wrist_norm <= 1.0e-8).any() or (object_norm <= 1.0e-8).any():
        raise ValueError(f"pen clip {path} contains a zero-length quaternion")
    wrist_q = _quat_normalize(wrist[:, 3:])
    obj_q = _quat_normalize(obj[:, 3:])
    inv = _quat_conjugate(wrist_q)
    rel_pos = _quat_rotate(inv, obj[:, :3] - wrist[:, :3])
    rel_q = _quat_normalize(_quat_multiply(inv, obj_q))
    rel_links = _quat_rotate(inv[:, None, :], links - wrist[:, None, :3])
    joint_vel = torch.zeros_like(qpos)
    if len(qpos) > 1:
        joint_vel[:-1] = (qpos[1:] - qpos[:-1]) / float(frame_dt)
        joint_vel[-1] = joint_vel[-2]
    # The source task mounts every clip on the same calibrated wrist.  Keep
    # the clip's wrist-local object/link data, but publish that fixed world
    # frame instead of allowing small capture-mount differences to become an
    # artificial command input.
    fixed_wrist = torch.tensor(FIXED_WRIST_POSE_W, device=device, dtype=wrist.dtype)
    fixed_wrist = fixed_wrist.expand(len(qpos), -1).clone()
    fixed_q = fixed_wrist[:, 3:]
    object_pos_w = fixed_wrist[:, :3] + _quat_rotate(fixed_q, rel_pos)
    object_quat_w = _quat_normalize(_quat_multiply(fixed_q, rel_q))
    object_linvel_w, object_angvel_w = _pose_velocity(
        object_pos_w, object_quat_w, frame_dt
    )
    return {
        "joint_pos": qpos,
        "joint_vel": joint_vel,
        "wrist_pose_w": fixed_wrist,
        "object_pose": torch.cat((rel_pos, rel_q), -1),
        "link_pos": rel_links,
        "object_linvel_w": object_linvel_w,
        "object_angvel_w": object_angvel_w,
    }


def _pose_velocity(
    position: torch.Tensor, quaternion: torch.Tensor, frame_dt: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finite-difference world pose into the reset velocity contract."""
    linear = torch.zeros_like(position)
    angular = torch.zeros_like(position)
    if len(position) <= 1:
        return linear, angular
    linear[:-1] = (position[1:] - position[:-1]) / float(frame_dt)
    linear[-1] = linear[-2]
    delta = _quat_normalize(_quat_multiply(quaternion[1:], _quat_conjugate(quaternion[:-1])))
    delta = torch.where(delta[:, :1] < 0.0, -delta, delta)
    sin_half = torch.linalg.vector_norm(delta[:, 1:], dim=-1)
    axis = delta[:, 1:] / sin_half.clamp_min(1e-8).unsqueeze(-1)
    angle = 2.0 * torch.atan2(sin_half, delta[:, 0].clamp(-1.0, 1.0))
    angular[:-1] = axis * (angle / float(frame_dt)).unsqueeze(-1)
    angular[-1] = angular[-2]
    return linear, angular


def _synthetic_reference(default: torch.Tensor, device: torch.device, frame_dt: float) -> dict[str, torch.Tensor]:
    """Generate a finite smoke trajectory matching the source tensor contract."""
    frames = 500
    t = torch.arange(frames, device=device, dtype=torch.float32) * float(frame_dt)
    wrist = torch.tensor(FIXED_WRIST_POSE_W, device=device).expand(frames, -1).clone()
    angle = 2.0 * torch.pi * t / 5.0
    obj_q = torch.stack((torch.cos(angle / 2), torch.sin(angle / 2), torch.zeros_like(angle), torch.zeros_like(angle)), -1)
    obj_pos = torch.tensor((0.0, 0.0, 0.08), device=device).expand(frames, -1)
    links = torch.zeros((frames, 21, 3), device=device)
    joint_pos = default.expand(frames, -1).clone()
    object_quat_w = _quat_multiply(wrist[:, 3:], obj_q)
    object_linvel_w, object_angvel_w = _pose_velocity(
        wrist[:, :3] + _quat_rotate(wrist[:, 3:], obj_pos), object_quat_w, frame_dt
    )
    return {
        "joint_pos": joint_pos,
        "joint_vel": torch.zeros_like(joint_pos),
        "wrist_pose_w": wrist,
        "object_pose": torch.cat((obj_pos, obj_q), -1),
        "link_pos": links,
        "object_linvel_w": object_linvel_w,
        "object_angvel_w": object_angvel_w,
    }


@dataclass(frozen=True)
class TrajectoryGroup:
    """Command-owned trajectories for exactly one object asset."""

    frame_dt: float
    trajectory_end: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    wrist_pose_w: torch.Tensor
    object_pose: torch.Tensor
    link_pos: torch.Tensor

    @classmethod
    def from_reader_data(cls, data: dict[str, object]) -> TrajectoryGroup:
        trajectory_lengths = torch.as_tensor(data["trajectory_lengths"], dtype=torch.long)
        return cls(
            frame_dt=data["frame_dt"],
            trajectory_end=torch.cumsum(trajectory_lengths, dim=0),
            joint_pos=data["joint_pos"],
            joint_vel=data["joint_vel"],
            wrist_pose_w=data["wrist_pose_w"],
            object_pose=data["object_pose"],
            link_pos=data["link_pos"],
        )

    @property
    def num_trajectories(self) -> int:
        return self.trajectory_end.numel()


@dataclass(kw_only=True)
class PenSpinCommandCfg(CommandTermCfg):
    entity_name: str = "robot"
    object_name: str = "object"
    recipe_path: str = ""
    clips_root: str | None = None
    synthetic_reference: bool = True
    frame_dt: float = 0.02
    episode_frames: int = 500
    axis_point_radius_m: float = 0.10
    object_position_tanh_scale_m: float = 0.15
    trajectory_sampling_mode: str = "uniform"
    adaptive_sampling_ema_decay: float = 0.99
    adaptive_sampling_uniform_mix: float = 0.20
    adaptive_sampling_bin_seconds: float = 1.0
    adaptive_sampling_update_steps: int = 40
    adaptive_sampling_kernel_size: int = 1
    adaptive_sampling_kernel_decay: float = 0.8
    # The migrated YAML originally names source's "unstable" term nan_guard.
    instability_term_name: str = "nan_guard"
    joint_names: tuple[str, ...] = JOINT_NAMES
    link_names: tuple[str, ...] = LINK_NAMES
    fixed_wrist_pose_w: tuple[float, float, float, float, float, float, float] = FIXED_WRIST_POSE_W
    first_frame_hold_prob: float = 0.15
    first_frame_hold_s: tuple[float, float] = (0.5, 3.0)
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    def build(self, env):
        return PenSpinCommand(self, env)


class PenSpinCommand(CommandTerm):
    cfg: PenSpinCommandCfg

    def __init__(self, cfg: PenSpinCommandCfg, env):
        super().__init__(cfg, env)
        self.robot = cast(Entity, env.scene[cfg.entity_name])
        self.object = cast(Entity, env.scene[cfg.object_name])
        # The SuperMimic reader consumes actual self-contact and actuator-force
        # labels through the same packed read phase as the physical state.
        self._sensor_names = (
            *(f"finger_collision_{index}" for index in range(25)),
            *(f"actuator_force_{index}" for index in range(20)),
        )
        env.scene.bind_sensor_data(self._sensor_names)
        if not math.isfinite(cfg.frame_dt) or cfg.frame_dt <= 0:
            raise ValueError("PenSpin frame_dt must be finite and positive")
        if not cfg.joint_names:
            raise ValueError("joint_names must be configured by the concrete robot config")
        if not cfg.link_names:
            raise ValueError("link_names must be configured by the concrete robot config")
        if cfg.axis_point_radius_m <= 0:
            raise ValueError("axis_point_radius_m must be positive")
        if not (
            math.isfinite(cfg.object_position_tanh_scale_m)
            and cfg.object_position_tanh_scale_m > 0.0
        ):
            raise ValueError("bounded object mapping scale must be finite and positive")
        if cfg.trajectory_sampling_mode not in ("uniform", "adaptive"):
            raise ValueError(
                f"trajectory_sampling_mode must be 'uniform' or 'adaptive', got {cfg.trajectory_sampling_mode!r}"
            )
        if not (
            math.isfinite(cfg.adaptive_sampling_ema_decay)
            and 0.0 <= cfg.adaptive_sampling_ema_decay < 1.0
        ):
            raise ValueError("adaptive_sampling_ema_decay must be in [0, 1)")
        if not (
            math.isfinite(cfg.adaptive_sampling_uniform_mix)
            and 0.0 <= cfg.adaptive_sampling_uniform_mix <= 1.0
        ):
            raise ValueError("adaptive_sampling_uniform_mix must be in [0, 1]")
        if not (
            math.isfinite(cfg.adaptive_sampling_bin_seconds)
            and cfg.adaptive_sampling_bin_seconds > 0.0
        ):
            raise ValueError("adaptive_sampling_bin_seconds must be finite and positive")
        if cfg.adaptive_sampling_update_steps < 1:
            raise ValueError("adaptive_sampling_update_steps must be at least 1")
        if (
            not isinstance(cfg.adaptive_sampling_kernel_size, int)
            or isinstance(cfg.adaptive_sampling_kernel_size, bool)
            or cfg.adaptive_sampling_kernel_size <= 0
        ):
            raise ValueError("adaptive_sampling_kernel_size must be a positive integer")
        if not (
            math.isfinite(cfg.adaptive_sampling_kernel_decay)
            and 0.0 <= cfg.adaptive_sampling_kernel_decay <= 1.0
        ):
            raise ValueError("adaptive_sampling_kernel_decay must be in [0, 1]")

        if not (
            math.isfinite(cfg.first_frame_hold_prob)
            and 0.0 <= cfg.first_frame_hold_prob <= 1.0
        ):
            raise ValueError("first_frame_hold_prob must be in [0, 1]")
        hold_min_s, hold_max_s = cfg.first_frame_hold_s
        if not (math.isfinite(hold_max_s) and 0.0 <= hold_min_s <= hold_max_s):
            raise ValueError(
                "first_frame_hold_s must be a finite (min, max) with 0 <= min <= max"
            )

        if cfg.recipe_path or cfg.clips_root:
            self._frames = self._load_catalog(cfg)
        elif cfg.synthetic_reference:
            default = self.robot.data.default_joint_pos(self._device)
            default = default[0] if default.ndim >= 2 else default
            self._frames = _synthetic_reference(default, self._device, cfg.frame_dt)
        else:
            raise FileNotFoundError(
                "Pen spin requires a clip catalog: pass commands.motion.recipe_path and "
                "commands.motion.clips_root (or set synthetic_reference=true for a smoke run)"
            )
        self.frame_count = int(self._frames["joint_pos"].shape[0])
        if self.frame_count < 2:
            raise ValueError("Pen spin reference must contain at least two frames")
        lengths = getattr(self, "_catalog_lengths", None)
        if lengths is None:
            lengths = torch.tensor([self.frame_count], dtype=torch.long, device=self._device)
        self._trajectory_lengths = lengths
        self._trajectory_starts = torch.cat(
            (lengths.new_zeros(1), lengths.cumsum(0)[:-1])
        )
        self._trajectory_ends = self._trajectory_starts + lengths
        self.group = TrajectoryGroup.from_reader_data({
            "frame_dt": cfg.frame_dt,
            "trajectory_lengths": lengths,
            **self._frames,
        })
        self._validate_groups()
        self._configure_adaptive_sampling_bins()
        self._configure_fixed_wrist_reference()
        self.trajectory_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self._device
        )
        self.time_steps = torch.zeros_like(self.trajectory_ids)
        self._episode_start_steps = torch.zeros_like(self.trajectory_ids)
        self._reset_prepared = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self._device
        )
        self.bin_failure_scores = torch.zeros(
            self._bin_trajectory_ids.numel(),
            dtype=torch.float32,
            device=self._device,
        )
        # Episode outcomes since the last score update, [trials; failures] per bin.
        self._pending_bin_outcomes = torch.zeros(
            (2, self._bin_trajectory_ids.numel()),
            dtype=torch.float32,
            device=self._device,
        )
        self._steps_since_score_update = 0
        self._sampling_probability = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self._device
        )
        self._sampled_bin_failure_score = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self._device
        )
        self._sampling_entropy = torch.zeros_like(self._sampling_probability)
        self._sampling_top1_prob = torch.zeros_like(self._sampling_probability)
        self._sampling_top1_bin = torch.zeros_like(self._sampling_probability)
        self.metrics["sampling_probability"] = self._sampling_probability.clone()
        self.metrics["bin_failure_score"] = self._sampled_bin_failure_score.clone()
        self.metrics["sampling_entropy"] = self._sampling_entropy.clone()
        self.metrics["sampling_top1_prob"] = self._sampling_top1_prob.clone()
        self.metrics["sampling_top1_bin"] = self._sampling_top1_bin.clone()

        self.reference_end = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        self._read_plan = None
        self.metrics["reference_frame"] = torch.zeros(self.num_envs, device=self._device)
        self.axis_points = torch.tensor(
            ((cfg.axis_point_radius_m, 0.0, 0.0), (-cfg.axis_point_radius_m, 0.0, 0.0)),
            dtype=self._frames["object_pose"].dtype,
            device=self._device,
        )

    def _configure_adaptive_sampling_bins(self) -> None:
        """Build boundary-safe local start-frame bins for every trajectory."""
        trajectory_lengths = self._trajectory_ends - self._trajectory_starts
        bin_size_frames = max(
            1,
            round(self.cfg.adaptive_sampling_bin_seconds / self.group.frame_dt),
        )
        self._adaptive_sampling_bin_size_frames = bin_size_frames

        start_slots = (trajectory_lengths - 1).clamp_min(1)
        self._trajectory_bin_counts = torch.div(
            start_slots + bin_size_frames - 1,
            bin_size_frames,
            rounding_mode="floor",
        )
        self._trajectory_bin_offsets = torch.cat(
            (
                self._trajectory_bin_counts.new_zeros(1),
                self._trajectory_bin_counts.cumsum(0)[:-1],
            )
        )

        trajectory_ids = torch.arange(
            trajectory_lengths.numel(), device=self._device, dtype=torch.long
        )
        self._bin_trajectory_ids = torch.repeat_interleave(
            trajectory_ids, self._trajectory_bin_counts
        )
        bin_ids = torch.arange(
            self._bin_trajectory_ids.numel(), device=self._device, dtype=torch.long
        )
        local_bin_ids = bin_ids - self._trajectory_bin_offsets[self._bin_trajectory_ids]
        self._bin_start_steps = local_bin_ids * bin_size_frames
        self._bin_stop_steps = torch.minimum(
            self._bin_start_steps + bin_size_frames,
            start_slots[self._bin_trajectory_ids],
        )

        kernel_offsets = torch.arange(
            self.cfg.adaptive_sampling_kernel_size,
            device=self._device,
            dtype=torch.long,
        )
        final_bin_ids = (
            self._trajectory_bin_offsets[self._bin_trajectory_ids]
            + self._trajectory_bin_counts[self._bin_trajectory_ids]
            - 1
        )
        self._bin_kernel_indices = torch.minimum(
            bin_ids[:, None] + kernel_offsets[None, :],
            final_bin_ids[:, None],
        )
        kernel = self.cfg.adaptive_sampling_kernel_decay ** kernel_offsets.to(torch.float32)
        self._adaptive_sampling_kernel = kernel / kernel.sum()

    def _configure_fixed_wrist_reference(self) -> None:
        """Rebase every wrist-local trajectory onto one physical wrist mount."""
        fixed_pose = self.cfg.fixed_wrist_pose_w
        if fixed_pose is None:
            raise ValueError(
                "fixed_wrist_pose_w must be configured by the concrete robot config"
            )
        dtype = self._frames["wrist_pose_w"].dtype
        pose = torch.as_tensor(
            fixed_pose,
            device=self._device,
            dtype=dtype,
        )
        if pose.shape != (7,) or not torch.isfinite(pose).all():
            raise ValueError("fixed_wrist_pose_w must contain seven finite values")
        quat_norm = torch.linalg.vector_norm(pose[3:])
        if quat_norm <= 0:
            raise ValueError("fixed_wrist_pose_w must contain a non-zero quaternion")
        pose = pose.clone()
        pose[3:] /= quat_norm
        wrist_pose_w = (
            pose.unsqueeze(0).expand(self._frames["wrist_pose_w"].shape[0], -1).clone()
        )
        object_pose = self._frames["object_pose"]
        object_pos_w = wrist_pose_w[:, :3] + _quat_rotate(
            wrist_pose_w[:, 3:], object_pose[:, :3]
        )
        object_quat_w = _quat_multiply(wrist_pose_w[:, 3:], object_pose[:, 3:])
        object_pose_w = torch.cat((object_pos_w, object_quat_w), dim=-1)
        object_linvel_w = torch.zeros_like(object_pos_w)
        object_angvel_w = torch.zeros_like(object_pos_w)
        frame_dt = self.group.frame_dt
        for start, end in zip(
            self._trajectory_starts.tolist(),
            self._trajectory_ends.tolist(),
            strict=True,
        ):
            linear, angular = _pose_velocity(
                object_pose_w[start:end, :3], object_pose_w[start:end, 3:], frame_dt
            )
            object_linvel_w[start:end] = linear
            object_angvel_w[start:end] = angular
        self._frames.update(
            wrist_pose_w=wrist_pose_w,
            object_linvel_w=object_linvel_w,
            object_angvel_w=object_angvel_w,
        )

    def _validate_control_frequency(self, frame_dt: float) -> None:
        """The command advances one clip frame per control step, so frame_dt must equal step_dt."""
        control_dt = float(self._env.step_dt)
        frame_dt = float(frame_dt)
        if abs(frame_dt - control_dt) > 1e-3 * control_dt:
            raise ValueError(
                f"clip fps {1.0 / frame_dt:.4g} Hz (frame_dt={frame_dt:.6g}s) must match "
                f"the control frequency {1.0 / control_dt:.4g} Hz (step_dt={control_dt:.6g}s): "
                "a mismatch time-warps the reference."
            )

    def _validate_groups(self) -> None:
        group = self.group
        self._validate_control_frequency(group.frame_dt)
        expected_joints = len(self.cfg.joint_names)
        expected_links = tuple(self.cfg.link_names)
        if len(set(expected_links)) != len(expected_links):
            raise ValueError("link_names must be unique")
        if group.joint_pos.ndim != 2 or group.joint_pos.shape[1] != expected_joints:
            raise ValueError(
                "tracking joint width must match configured joint_names; "
                f"expected {expected_joints}, got {tuple(group.joint_pos.shape)}"
            )
        if group.link_pos.ndim != 3 or group.link_pos.shape[1:] != (
            len(expected_links),
            3,
        ):
            raise ValueError(
                "tracking link shape must match configured link_names; "
                f"expected (*, {len(expected_links)}, 3), got {tuple(group.link_pos.shape)}"
            )

    def _load_catalog(self, cfg: PenSpinCommandCfg) -> dict[str, torch.Tensor]:
        if not cfg.recipe_path:
            raise ValueError("clips_root requires recipe_path for pen-spin reference selection")
        resolved = resolve_recipe(
            cfg.recipe_path,
            clips_root=cfg.clips_root,
            verify_clip_hashes=True,
        )
        (clip_set,) = resolved.clip_sets
        if abs(float(clip_set.frame_dt) - float(cfg.frame_dt)) > 1e-3 * float(cfg.frame_dt):
            raise ValueError(
                "pen-spin clip frame_dt must match command frame_dt: "
                f"catalog={clip_set.frame_dt:.6g}, command={cfg.frame_dt:.6g}"
            )
        trajectories = [_load_reference(clip.npz_path, self._device, cfg.frame_dt) for clip in clip_set.clips]
        self._catalog_lengths = torch.tensor(
            [item["joint_pos"].shape[0] for item in trajectories],
            dtype=torch.long,
            device=self._device,
        )
        return {key: torch.cat([item[key] for item in trajectories], dim=0) for key in trajectories[0]}

    @property
    def tensor_sensor_names(self) -> tuple[str, ...]:
        return self._sensor_names

    @property
    def command(self) -> torch.Tensor:
        return torch.cat(
            (self.joint_pos, self.bounded_object_axis_points, self.link_pos.flatten(1)), -1
        )

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(self.cfg.joint_names)

    @property
    def trajectory_starts(self) -> torch.Tensor:
        return self._trajectory_starts[self.trajectory_ids]

    @property
    def trajectory_ends(self) -> torch.Tensor:
        return self._trajectory_ends[self.trajectory_ids]

    @property
    def trajectory_lengths(self) -> torch.Tensor:
        return self.trajectory_ends - self.trajectory_starts

    @property
    def holding_first_frame(self) -> torch.Tensor:
        return self.time_steps < 0

    @property
    def object_axis_points(self) -> torch.Tensor:
        pose = self.object_pose
        rotated = _quat_rotate(pose[:, None, 3:], self.axis_points)
        return (pose[:, None, :3] + rotated).flatten(1)

    @property
    def bounded_object_axis_points(self) -> torch.Tensor:
        return (self.cfg.object_position_tanh_scale_m * torch.tanh(
            self.object_axis_points / self.cfg.object_position_tanh_scale_m
        )).flatten(1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._frames["joint_pos"][self.frame_ids]

    @property
    def frame_ids(self) -> torch.Tensor:
        lengths = self._trajectory_lengths[self.trajectory_ids]
        local = self.time_steps.clamp_min(0).minimum(lengths - 1)
        return self._trajectory_starts[self.trajectory_ids] + local

    @property
    def joint_vel(self) -> torch.Tensor:
        value = self._frames["joint_vel"][self.frame_ids]
        return torch.where(self.time_steps[:, None] < 0, torch.zeros_like(value), value)

    @property
    def wrist_pose_w(self) -> torch.Tensor:
        pose = self._frames["wrist_pose_w"][self.frame_ids].clone()
        pose[:, :3] += self._env.scene.env_origins(self._device)
        return pose

    @property
    def object_pose(self) -> torch.Tensor:
        return self._frames["object_pose"][self.frame_ids]

    @property
    def link_pos(self) -> torch.Tensor:
        return self._frames["link_pos"][self.frame_ids]

    @property
    def object_linvel_w(self) -> torch.Tensor:
        value = self._frames["object_linvel_w"][self.frame_ids]
        return torch.where(self.time_steps[:, None] < 0, torch.zeros_like(value), value)

    @property
    def object_angvel_w(self) -> torch.Tensor:
        value = self._frames["object_angvel_w"][self.frame_ids]
        return torch.where(self.time_steps[:, None] < 0, torch.zeros_like(value), value)

    def bind_read_phase(self) -> None:
        self._read_plan = getattr(self._env.scene, "read_plan", None)

    def _bin_ids_at(
        self, env_ids: torch.Tensor, local_steps: torch.Tensor
    ) -> torch.Tensor:
        trajectory_ids = self.trajectory_ids[env_ids]
        trajectory_lengths = (
            self._trajectory_ends[trajectory_ids] - self._trajectory_starts[trajectory_ids]
        )
        final_start_steps = (trajectory_lengths - 2).clamp_min(0)
        local_steps = local_steps.clamp_min(0).minimum(final_start_steps)
        local_bin_ids = torch.div(
            local_steps,
            self._adaptive_sampling_bin_size_frames,
            rounding_mode="floor",
        ).minimum(self._trajectory_bin_counts[trajectory_ids] - 1)
        return self._trajectory_bin_offsets[trajectory_ids] + local_bin_ids

    def _current_bin_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self._bin_ids_at(env_ids, self.time_steps[env_ids])

    def _record_bin_outcomes(self, env_ids: torch.Tensor) -> None:
        if self.cfg.trajectory_sampling_mode != "adaptive":
            return
        termination_manager = self._env.termination_manager
        terminated = termination_manager.terminated[env_ids]
        timed_out = termination_manager.time_outs[env_ids]
        unstable = termination_manager.get_term(self.cfg.instability_term_name)[env_ids]
        failed = terminated & ~unstable
        succeeded = timed_out & ~terminated
        has_outcome = (
            (failed | succeeded) & ~unstable & (self._env.episode_length_buf[env_ids] > 0)
        )
        completed_env_ids = env_ids[has_outcome]
        if len(completed_env_ids) == 0:
            return

        # An episode is one trial in every bin it entered: it survived each bin
        # before the last, and the last one takes the outcome. Scoring only the
        # final bin never records survivals, and the sampler flattens to uniform.
        final_bin_ids = self._current_bin_ids(completed_env_ids)
        first_bin_ids = self._bin_ids_at(
            completed_env_ids, self._episode_start_steps[completed_env_ids]
        )
        outcomes = failed[has_outcome].to(self.bin_failure_scores.dtype)
        num_bins = self.bin_failure_scores.numel()
        entered = torch.bincount(first_bin_ids, minlength=num_bins + 1)
        entered -= torch.bincount(final_bin_ids + 1, minlength=num_bins + 1)
        counts = entered.cumsum(0)[:num_bins]
        failure_sums = torch.bincount(
            final_bin_ids,
            weights=outcomes,
            minlength=num_bins,
        )
        self._pending_bin_outcomes[0] += counts
        self._pending_bin_outcomes[1] += failure_sums

    def _update_bin_failure_scores(self) -> None:
        """Fold the pending outcomes of every rank into the scores.

        Runs on a fixed step count, never on a reset: ranks reset at different
        times, and a collective that only some ranks reach hangs the run.
        """
        self._steps_since_score_update += 1
        if self._steps_since_score_update < self.cfg.adaptive_sampling_update_steps:
            return
        self._steps_since_score_update = 0
        if dist.is_initialized():
            dist.all_reduce(self._pending_bin_outcomes)
        counts, failure_sums = self._pending_bin_outcomes
        updated = counts > 0
        batch_failure_rates = failure_sums[updated] / counts[updated]
        decay_powers = self.cfg.adaptive_sampling_ema_decay ** counts[updated]
        self.bin_failure_scores[updated] = (
            decay_powers * self.bin_failure_scores[updated]
            + (1.0 - decay_powers) * batch_failure_rates
        )
        self._pending_bin_outcomes.zero_()

    def _smoothed_bin_failure_scores(self) -> torch.Tensor:
        gathered = self.bin_failure_scores[self._bin_kernel_indices]
        return (gathered * self._adaptive_sampling_kernel).sum(dim=-1)

    def _adaptive_sampling_probabilities(self) -> torch.Tensor:
        scores = self._smoothed_bin_failure_scores()
        uniform = torch.full_like(scores, 1.0 / scores.numel())
        score_sum = scores.sum()
        proportional = torch.where(
            score_sum > 0.0,
            scores / score_sum.clamp_min(torch.finfo(scores.dtype).eps),
            uniform,
        )
        mix = self.cfg.adaptive_sampling_uniform_mix
        return mix * uniform + (1.0 - mix) * proportional

    def _set_sampling_distribution_metrics(
        self,
        selected: torch.Tensor,
        probabilities: torch.Tensor,
        *,
        adaptive: bool,
    ) -> None:
        if probabilities.numel() > 1:
            entropy = -(
                probabilities * probabilities.clamp_min(1.0e-12).log()
            ).sum() / math.log(probabilities.numel())
        else:
            entropy = probabilities.new_tensor(1.0)
        top_probability, top_index = probabilities.max(dim=0)
        self._sampling_entropy[selected] = entropy
        self._sampling_top1_prob[selected] = top_probability
        self._sampling_top1_bin[selected] = (
            top_index.to(probabilities.dtype) / probabilities.numel() if adaptive else 0.0
        )

    def prepare_reset(self, env_ids: torch.Tensor) -> None:
        self._record_bin_outcomes(env_ids)
        if self.cfg.trajectory_sampling_mode == "adaptive":
            probabilities = self._adaptive_sampling_probabilities()
            sampled_bins = torch.multinomial(probabilities, len(env_ids), replacement=True)
            self.trajectory_ids[env_ids] = self._bin_trajectory_ids[sampled_bins]
            starts = self._bin_start_steps[sampled_bins]
            widths = self._bin_stop_steps[sampled_bins] - starts
            random_offsets = (
                torch.rand((len(env_ids),), device=self._device) * widths.to(torch.float32)
            ).to(torch.long)
            self.time_steps[env_ids] = starts + random_offsets
            self._sampling_probability[env_ids] = probabilities[sampled_bins]
            self._sampled_bin_failure_score[env_ids] = self.bin_failure_scores[sampled_bins]
            self._set_sampling_distribution_metrics(env_ids, probabilities, adaptive=True)
        else:
            num_trajectories = self.group.num_trajectories
            probabilities = torch.full(
                (num_trajectories,),
                1.0 / num_trajectories,
                dtype=self.bin_failure_scores.dtype,
                device=self._device,
            )
            sampled_trajectories = torch.randint(
                num_trajectories, (len(env_ids),), device=self._device
            )
            self.trajectory_ids[env_ids] = sampled_trajectories
            self.time_steps[env_ids] = 0
            self._sampling_probability[env_ids] = probabilities[sampled_trajectories]
            self._sampled_bin_failure_score[env_ids] = 0.0
            self._set_sampling_distribution_metrics(env_ids, probabilities, adaptive=False)
        self._hold_first_frame(env_ids)
        self._episode_start_steps[env_ids] = self.time_steps[env_ids]
        self._reset_prepared[env_ids] = True

    def _hold_first_frame(self, env_ids: torch.Tensor) -> None:
        """Send a share of the resets to a held first frame (negative ``time_steps``)."""
        if self.cfg.first_frame_hold_prob <= 0.0:
            return
        held = env_ids[
            torch.rand(len(env_ids), device=self._device) < self.cfg.first_frame_hold_prob
        ]
        if len(held) == 0:
            return
        frame_dt = self.group.frame_dt
        hold_min, hold_max = (
            round(seconds / frame_dt) for seconds in self.cfg.first_frame_hold_s
        )
        self.time_steps[held] = -torch.randint(
            hold_min, hold_max + 1, (len(held),), device=self._device
        )

    def _update_metrics(self) -> None:
        self.metrics["reference_frame"].copy_(self.frame_ids.float())
        self.metrics["sampling_probability"].copy_(self._sampling_probability)
        self.metrics["bin_failure_score"].copy_(self._sampled_bin_failure_score)
        self.metrics["sampling_entropy"].copy_(self._sampling_entropy)
        self.metrics["sampling_top1_prob"].copy_(self._sampling_top1_prob)
        self.metrics["sampling_top1_bin"].copy_(self._sampling_top1_bin)

    def _resample_command(self, reset_mask: torch.Tensor) -> None:
        # UniLab dispatches a full-width reset mask; the source sampler owns
        # trajectory rows, so convert once at this manager boundary.
        env_ids = reset_mask.nonzero(as_tuple=False).squeeze(-1)
        pending = env_ids[~self._reset_prepared[env_ids]]
        if len(pending):
            self.prepare_reset(pending)
        self._reset_prepared[env_ids] = False
        self.reference_end.masked_fill_(reset_mask, False)

    def _update_command(self) -> None:
        self.time_steps.add_(1)
        if self.cfg.trajectory_sampling_mode == "adaptive":
            self._update_bin_failure_scores()
        lengths = self._trajectory_lengths[self.trajectory_ids]
        ended = self.time_steps >= lengths - 1
        self.time_steps.copy_(torch.minimum(self.time_steps, lengths - 1))
        self.reference_end.copy_(ended)

    def refresh_reset(self, reset_mask: torch.Tensor) -> None:
        """Match mjlab's selected command advance after the physical reset."""
        # The event writes sampled frame k into physics. Source mjlab then
        # calls command.compute(dt=0, env_ids=...) before reset observations,
        # advancing only those rows to k+1 for the actor's next target. This
        # also advances a held first frame from -50 to -49 at 50 Hz.
        self.time_steps.add_(reset_mask.to(self.time_steps.dtype))
        ended = self.time_steps + 1 >= self.trajectory_lengths
        self.reference_end.copy_(torch.where(reset_mask, ended, self.reference_end))

    def post_compute_reset(self) -> None:
        """No cached backend pose is retained between manager phases."""
        return None

    def reset(self, env_ids=None, *, publish_metrics: bool = True):
        return super().reset(env_ids, publish_metrics=publish_metrics)


def task(env) -> PenSpinCommand:
    command = env.command_manager.get_term("motion")
    if not isinstance(command, PenSpinCommand):
        raise TypeError("Pen spin expects a PenSpinCommand named 'motion'")
    return command


__all__ = ["PenSpinCommand", "PenSpinCommandCfg", "TrajectoryGroup", "task", "JOINT_NAMES", "LINK_NAMES"]

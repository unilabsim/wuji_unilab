"""Metric logging backends for Dex SuperMimic."""

from __future__ import annotations

import csv
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

LoggerBackend = Literal["wandb", "tensorboard", "none"]

CSV_FIELDS = (
    "iteration",
    "training/collected_sim_steps",
    "training/time",
    "training/lr",
    "wm/wm_loss",
    "wm/state_loss",
    "wm/contact_digit_loss",
    "wm/failure_loss",
    "wm/holdout_wm_loss",
    "wm/holdout_state_loss",
    "wm/holdout_contact_digit_loss",
    "wm/holdout_failure_loss",
    "wm/reward",
    "wm/reward_parity",
    "policy/actor_loss",
    "policy/critic_loss",
    "wm/survival",
    "training/horizon",
    "task/drop_rate",
    "task/rot_error",
    # Native per-episode reductions, averaged over completed episodes in the
    # logged collect call. Blank when no episode completed, never a fake zero.
    "task/episode_object_pos_tracking_error_cm",
    "task/episode_object_ori_tracking_error_deg",
    "task/episode_action_rate_sq",
    "task/episode_action_acc_sq",
    "task/episode_joint_vel_rms_rad_s",
    "task/episode_joint_acc_rms_rad_s2",
    "task/episode_actuator_saturation_share",
    # Hand1's native episode time averages retain their source metric names.
    # In particular, goal_reach_count is an averaged counter, not a success rate.
    "task/episode_goal_reach_count",
    "task/episode_orientation_error",
    "task/episode_cube_survival_steps",
    "task/episode_action_delta_rms",
    "task/episode_action_jerk_rms",
    "task/episode_cage_escape_frequency",
    "task/episode_fingertip_contact_count",
    "task/episode_torque_saturation_ratio",
    "task/episode_joint_acceleration_rms",
    "task/episode_cube_height_above_palm",
    "task/episode_finger_collision_frequency",
    "task/episode_palm_detach_frequency",
    "task/episode_success_interval",
)


class MetricLogger:
    """Small facade over W&B and TensorBoard.

    TensorBoard files remain in the run directory. W&B uses the same directory as
    its ``dir`` so offline runs and local metadata stay colocated with checkpoints.
    """

    def __init__(
        self,
        backend: LoggerBackend,
        *,
        log_dir: str | Path,
        config: Mapping[str, Any] | None = None,
        run_name: str | None = None,
        project: str,
        entity: str,
    ) -> None:
        self.backend = backend
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.log_dir / "train_log.csv"
        has_header = csv_path.is_file() and csv_path.stat().st_size > 0
        fieldnames = CSV_FIELDS
        if has_header:
            # Resuming a run written before new telemetry must keep a valid
            # CSV header and every existing value/column.
            with csv_path.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                existing = tuple(reader.fieldnames or ())
                fieldnames = tuple(dict.fromkeys((*existing, *CSV_FIELDS)))
                if existing != fieldnames:
                    temporary = csv_path.with_suffix(".csv.tmp")
                    with temporary.open("w", encoding="utf-8", newline="") as output:
                        writer = csv.DictWriter(output, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(reader)
                    temporary.replace(csv_path)
        self._csv_stream = csv_path.open("a", encoding="utf-8", newline="")
        self._csv = csv.DictWriter(self._csv_stream, fieldnames=fieldnames)
        if not has_header:
            self._csv.writeheader()
            self._csv_stream.flush()
        if backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self._tensorboard = SummaryWriter(str(self.log_dir))
            self._wandb = None
        elif backend == "wandb":
            import wandb

            if os.getenv("WANDB_MODE", "").lower() == "disabled":
                self.backend = "none"
                self._wandb = self._tensorboard = None
                return
            if not project or not entity:
                raise ValueError("project and entity are required for wandb logging")
            self._wandb = wandb.init(
                project=project,
                entity=entity,
                name=run_name or self.log_dir.name,
                dir=str(self.log_dir),
                config=dict(config or {}),
            )
            self._tensorboard = None
        elif backend == "none":
            self._wandb = self._tensorboard = None
        else:
            raise ValueError(f"Unknown logger backend: {backend!r}")

    @property
    def wandb_run(self):
        """The active ``wandb.Run`` when W&B is selected."""
        return self._wandb

    def log_scalars(self, scalars: Mapping[str, float], step: int) -> None:
        """Log one dictionary of scalar metrics at a common explicit step."""
        values = {name: float(value) for name, value in scalars.items()}
        self._csv.writerow({
            "iteration": step,
            **{key: values[key] for key in self._csv.fieldnames if key != "iteration" and key in values},
        })
        self._csv_stream.flush()
        if self.backend == "wandb":
            self._wandb.log(values, step=step)
        elif self.backend == "tensorboard":
            for name, value in values.items():
                self._tensorboard.add_scalar(name, value, step)

    def log_images(self, name: str, images, step: int) -> None:
        """Log a ``(N, C, H, W)`` image batch."""
        if self.backend == "wandb":
            self._wandb.log({name: wandb_image_batch(images)}, step=step)
        elif self.backend == "tensorboard":
            self._tensorboard.add_images(name, images, step)

    def flush(self) -> None:
        self._csv_stream.flush()
        if self.backend == "tensorboard":
            self._tensorboard.flush()

    def close(self) -> None:
        if not self._csv_stream.closed:
            self._csv_stream.close()
        if self._tensorboard is not None:
            self._tensorboard.close()
        if self._wandb is not None:
            self._wandb.finish()
        self._wandb = self._tensorboard = None


def wandb_image_batch(images) -> list:
    """Convert a tensor/ndarray image batch to W&B image objects."""
    import wandb

    if hasattr(images, "detach"):
        return [wandb.Image(image) for image in images.detach().cpu()]
    if hasattr(images, "transpose"):
        images = images.transpose(0, 2, 3, 1)
    return [wandb.Image(image) for image in images]

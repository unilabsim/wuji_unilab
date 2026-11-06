"""Isolated bridges for mjwarp reset capability gaps.

UniLab 1.5 owns the public ``env_ids`` reset contract and observation history.
The only remaining task bridges are mjwarp fields whose derived constants still
cannot travel through the device-resident packed reset commit: selected mocap
poses and geometry sizes.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import torch
from unisim.dr.types import TensorResetRandomizationPayload


class ResetEntity(Protocol):
    """The subset of Entity used by these bridges."""

    @property
    def _reset_state(self) -> Any: ...  # upstream-owned transaction storage


def bind_mocap_reset(entity: ResetEntity, body_name: str, *, term_name: str):
    """Bind the named mocap body to the active reset transaction."""
    reset_state = entity._reset_state
    if reset_state is None:
        raise RuntimeError("Wuji reset requires an active reset transaction")
    return reset_state.bind_mocap_pose(body_name)


def write_mocap_reset(binding, env_ids: torch.Tensor, poses: torch.Tensor) -> None:
    """Publish selected mocap rows while mjwarp lacks packed-reset mocap."""
    rows = env_ids.detach().cpu().numpy()
    binding.write(rows, poses.detach().cpu().numpy()[rows])


def reset_backend(entity: ResetEntity):
    """Return the backend owning the active fixed-width reset transaction."""
    reset_state = entity._reset_state
    if reset_state is None:
        raise RuntimeError("Wuji reset requires an active reset transaction")
    return reset_state._require_packed_transaction().backend


def write_geom_sizes(
    backend,
    *,
    backend_geom_ids: np.ndarray,
    reset_rows: np.ndarray,
    values: torch.Tensor,
) -> None:
    """Publish host-only geom sizes and rebuild derived geometry fields.

    UniLab exposes geom-size staging, but mjwarp does not include that field in
    its device-plane reset commit; derived bounds require the backend's model
    update boundary.  ``reset_rows`` carries only the env rows being
    written so validation and derived-bound recompute stay proportional to the
    reset set instead of the full batch.
    """
    if reset_rows.size == 0:
        return
    sizes = np.array(backend._dr_geom_size[reset_rows], copy=True)
    sizes[:, backend_geom_ids] = values.detach().cpu().numpy()[reset_rows]
    updates = backend._prepare_reset_randomization(
        reset_rows,
        TensorResetRandomizationPayload(geom_size=torch.from_numpy(sizes)),
    )
    backend._apply_model_updates(updates)

"""Isolated bridges for UniLab 1.4 mjwarp reset capability gaps.

Each function prefers the public Entity boundary. Mocap and geometry-size writes
fall back to backend-owned boundaries only where mjwarp 2.0 cannot express them
through the current packed reset API. Keeping those calls here makes the
documented gaps auditable and prevents upstream-private names from spreading
through task terms.
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


def write_mocap_reset(binding, reset_mask: torch.Tensor, poses: torch.Tensor) -> None:
    """Publish selected mocap rows while mjwarp lacks packed-reset mocap."""
    rows = reset_mask.detach().cpu().nonzero(as_tuple=False).flatten().numpy()
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
    model_ids: np.ndarray,
    backend_geom_ids: np.ndarray,
    reset_mask: torch.Tensor,
    values: torch.Tensor,
) -> None:
    """Publish host-only geom sizes and rebuild derived geometry fields.

    UniLab 1.4 exposes geom-size staging, but mjwarp 2.0 does not include that
    field in its device-plane reset commit; derived bounds require the backend's
    model update boundary.
    """
    rows = reset_mask.detach().cpu().numpy()
    host_sizes = backend._dr_geom_size.copy()
    host_sizes[np.ix_(rows, backend_geom_ids)] = values.detach().cpu().numpy()[rows]
    updates = backend._prepare_reset_randomization(
        np.arange(host_sizes.shape[0]),
        TensorResetRandomizationPayload(geom_size=torch.from_numpy(host_sizes)),
    )
    backend._apply_model_updates(updates)

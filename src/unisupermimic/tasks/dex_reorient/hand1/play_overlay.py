"""Hand 1 goal markers using the completed cube playback visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
from unisim.backend.mjwarp.playback import _inject_interactive_debug_overlays

from unisupermimic.tasks.dex_reorient.hand2.overlay import goal_debug_overlay_primitives

if TYPE_CHECKING:
    from .commands import Hand1ReorientCommand


class Hand1GoalOverlay:
    """Draw six target faces and a completion lamp in a detached viewer scene.

    Hand 1 and Hand 2 commands both publish a world-frame goal quaternion,
    cube position, alignment flag, and success window. Reuse those existing
    marker semantics; omit the coordinate frame for the local playback view.
    """

    def __init__(self, model: mujoco.MjModel, command: Hand1ReorientCommand):
        self.model = model
        self.command = command
        self._mesh_ids: dict[str, int] = {}
        self._mesh_materials: dict[str, int] = {}

    def update(self, user_scn: mujoco.MjvScene, actual_data: mujoco.MjData) -> None:
        """Replace viewer-only markers without writing to simulation state."""
        del actual_data
        overlays = [
            [primitive for primitive in row if primitive.kind != "frame"]
            for row in goal_debug_overlay_primitives(self.command)
        ]
        _inject_interactive_debug_overlays(
            user_scn=user_scn,
            overlays=overlays,
            world=0,
            num_envs=self.command.num_envs,
            model=self.model,
            mesh_id_cache=self._mesh_ids,
            mesh_mat_cache=self._mesh_materials,
        )


__all__ = ["Hand1GoalOverlay"]

# Adapted marker semantics from wuji-technology/wuji-mjlab 9410a3a,
# src/wuji_mjlab/tasks/reorient/tooling/scene_builder.py (set_goal_mocap).
# Copyright 2026 Wuji Technology Co., Ltd. Apache-2.0.
"""Goal-pose debug overlay primitives for Wuji reorient playback recording.

``ReorientCommand.goal`` is expressed in the palm tag frame (the command
compares it against the tag-relative cube quaternion), so the world-frame
goal orientation is ``tag_quat * goal``.  As in the source evaluator, the
goal marker floats above the live cube position.  Primitive poses are
env-local: the batched mjwarp simulation carries no per-env grid offset, and
the playback renderer applies grid offsets itself.
"""

from __future__ import annotations

import numpy as np
from unisim.backend.base import DebugOverlayGetter, DebugPrimitive

from .commands import ReorientCommand, task
from .math import multiply

GOAL_MESH_ASSET = "object/cube_mesh"
GOAL_VIS_Z_OFFSET = 0.15
GOAL_GHOST_RGBA = (1.0, 1.0, 1.0, 0.6)
GOAL_FRAME_RGBA = (1.0, 1.0, 1.0, 0.6)
GOAL_FRAME_AXIS_LENGTH = 0.05


def goal_pose_world(state: ReorientCommand) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-env world-frame goal cube pose ``(pos, quat_wxyz)``."""
    cube_pos = np.asarray(state.cube.data.root_link_pos_w, dtype=np.float64)
    _, tag_quat = state.tag_pose()
    goal_quat = multiply(
        np.asarray(tag_quat, dtype=np.float64), np.asarray(state.goal, dtype=np.float64)
    )
    goal_pos = cube_pos + np.array([0.0, 0.0, GOAL_VIS_Z_OFFSET], dtype=np.float64)
    return goal_pos, goal_quat


def goal_debug_overlay_primitives(state: ReorientCommand) -> list[list[DebugPrimitive]]:
    """Build the per-env goal ghost cube and RGB triad for one frame."""
    goal_pos, goal_quat = goal_pose_world(state)
    overlays: list[list[DebugPrimitive]] = []
    for pos, quat in zip(goal_pos, goal_quat, strict=True):
        pos_tuple = (float(pos[0]), float(pos[1]), float(pos[2]))
        quat_tuple = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        overlays.append(
            [
                DebugPrimitive(
                    kind="ghost_geom",
                    pos=pos_tuple,
                    quat=quat_tuple,
                    rgba=GOAL_GHOST_RGBA,
                    mesh_asset=GOAL_MESH_ASSET,
                ),
                DebugPrimitive(
                    kind="frame",
                    pos=pos_tuple,
                    quat=quat_tuple,
                    size=(GOAL_FRAME_AXIS_LENGTH,),
                    rgba=GOAL_FRAME_RGBA,
                ),
            ]
        )
    return overlays


def goal_overlay_getter(env) -> DebugOverlayGetter:
    """Return the per-frame goal overlay getter for a Wuji reorient env."""
    state = task(env)

    def _get_overlay() -> list[list[DebugPrimitive]]:
        return goal_debug_overlay_primitives(state)

    return _get_overlay

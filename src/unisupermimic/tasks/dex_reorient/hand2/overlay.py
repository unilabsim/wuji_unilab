# Adapted marker semantics from wuji-technology/wuji-mjlab 9410a3a,
# src/wuji_mjlab/tasks/reorient/tooling/scene_builder.py (set_goal_mocap).
# Copyright 2026 Wuji Technology Co., Ltd. Apache-2.0.
"""Goal-pose debug overlay primitives for Wuji reorient playback recording.

``Hand2ReorientCommand.goal`` is stored in world coordinates by the UniLab command
(it is composed with the tag pose when sampled), so it can be passed directly
to the renderer.  As in the source evaluator, the goal marker floats above
the live cube position.  Primitive poses are
env-local: the batched mjwarp simulation carries no per-env grid offset, and
the playback renderer applies grid offsets itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from unisim.backend.base import DebugOverlayGetter, DebugPrimitive

if TYPE_CHECKING:
    from .commands import Hand2ReorientCommand

GOAL_MESH_ASSET = "object/cube_mesh"
GOAL_VIS_Z_OFFSET = 0.15
# The six target faces follow the ``arucocube`` mapping in cube.xml:
# [+X, -X, +Y, -Y, +Z, -Z] = [purple, blue, red, cyan, green, black].
GOAL_GHOST_RGBA = (1.0, 1.0, 1.0, 0.18)
GOAL_FACE_HALF_SIZE = 0.027
GOAL_FACE_THICKNESS = 0.0012
GOAL_FACE_RGBA = (
    (0.68, 0.12, 0.88, 0.92),
    (0.05, 0.22, 0.95, 0.92),
    (0.95, 0.08, 0.08, 0.92),
    (0.02, 0.82, 0.92, 0.92),
    (0.08, 0.88, 0.18, 0.92),
    (0.03, 0.03, 0.03, 0.92),
)
GOAL_FRAME_RGBA = (1.0, 1.0, 1.0, 0.95)
GOAL_FRAME_AXIS_LENGTH = 0.08
GOAL_LAMP_OFFSET = (0.0, 0.0, 0.085)
GOAL_LAMP_SIZE = 0.014
GOAL_LAMP_RED_RGBA = (1.0, 0.04, 0.02, 1.0)
GOAL_LAMP_AMBER_RGBA = (1.0, 0.72, 0.02, 1.0)
GOAL_LAMP_GREEN_RGBA = (0.05, 1.0, 0.12, 1.0)


def _quat_to_matrix(quat: tuple[float, float, float, float]) -> np.ndarray:
    """Convert a wxyz quaternion to a 3x3 matrix without a backend import."""
    w, x, y, z = quat
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def goal_pose_world(state: Hand2ReorientCommand) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-env world-frame goal cube pose ``(pos, quat_wxyz)``."""
    cube_pos = state.cube_position().detach().cpu().numpy().astype(np.float64, copy=False)
    goal_quat = np.asarray(state.goal.detach().cpu(), dtype=np.float64)
    goal_pos = cube_pos + np.array([0.0, 0.0, GOAL_VIS_Z_OFFSET], dtype=np.float64)
    return goal_pos, goal_quat


def goal_debug_overlay_primitives(state: Hand2ReorientCommand) -> list[list[DebugPrimitive]]:
    """Build a high-contrast goal marker and completion lamp for one frame.

    The mesh ghost inherits the cube material in MuJoCo's user scene, so the
    six colored face panels guarantee a visible marker even when the mesh
    texture overrides ``rgba``.  ``window`` is the command's persistent
    success-window flag; ``within`` marks a currently aligned but not-yet-held
    target.  The lamp therefore stays red, turns amber while inside the
    threshold, and turns green after the hold requirement is satisfied.
    """
    goal_pos, goal_quat = goal_pose_world(state)
    def cpu_array(value, size: int) -> np.ndarray:
        if value is None:
            return np.zeros(size, dtype=np.float64)
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        return np.asarray(value, dtype=np.float64).reshape(-1)

    window = cpu_array(getattr(state, "window", None), len(goal_pos))
    within = cpu_array(getattr(state, "within", None), len(goal_pos))
    overlays: list[list[DebugPrimitive]] = []
    for index, (pos, quat) in enumerate(zip(goal_pos, goal_quat, strict=True)):
        pos_tuple = (float(pos[0]), float(pos[1]), float(pos[2]))
        quat_tuple = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        if index < len(window) and window[index] > 0.5:
            lamp_rgba = GOAL_LAMP_GREEN_RGBA
        elif index < len(within) and within[index] > 0.5:
            lamp_rgba = GOAL_LAMP_AMBER_RGBA
        else:
            lamp_rgba = GOAL_LAMP_RED_RGBA
        lamp_pos = (
            pos_tuple[0] + GOAL_LAMP_OFFSET[0],
            pos_tuple[1] + GOAL_LAMP_OFFSET[1],
            pos_tuple[2] + GOAL_LAMP_OFFSET[2],
        )
        rotation = _quat_to_matrix(quat_tuple)
        face_specs = (
            ((1.0, 0.0, 0.0), (GOAL_FACE_THICKNESS, GOAL_FACE_HALF_SIZE, GOAL_FACE_HALF_SIZE)),
            ((-1.0, 0.0, 0.0), (GOAL_FACE_THICKNESS, GOAL_FACE_HALF_SIZE, GOAL_FACE_HALF_SIZE)),
            ((0.0, 1.0, 0.0), (GOAL_FACE_HALF_SIZE, GOAL_FACE_THICKNESS, GOAL_FACE_HALF_SIZE)),
            ((0.0, -1.0, 0.0), (GOAL_FACE_HALF_SIZE, GOAL_FACE_THICKNESS, GOAL_FACE_HALF_SIZE)),
            ((0.0, 0.0, 1.0), (GOAL_FACE_HALF_SIZE, GOAL_FACE_HALF_SIZE, GOAL_FACE_THICKNESS)),
            ((0.0, 0.0, -1.0), (GOAL_FACE_HALF_SIZE, GOAL_FACE_HALF_SIZE, GOAL_FACE_THICKNESS)),
        )
        face_primitives = []
        for (normal, size), rgba in zip(face_specs, GOAL_FACE_RGBA, strict=True):
            local_center = np.asarray(normal, dtype=np.float64) * (
                GOAL_FACE_HALF_SIZE + GOAL_FACE_THICKNESS
            )
            face_center = np.asarray(pos, dtype=np.float64) + rotation @ local_center
            face_primitives.append(
                DebugPrimitive(
                    kind="box",
                    pos=(float(face_center[0]), float(face_center[1]), float(face_center[2])),
                    quat=quat_tuple,
                    size=tuple(float(value) for value in size),
                    rgba=rgba,
                )
            )
        overlays.append(
            [
                DebugPrimitive(
                    kind="ghost_geom",
                    pos=pos_tuple,
                    quat=quat_tuple,
                    rgba=GOAL_GHOST_RGBA,
                    mesh_asset=GOAL_MESH_ASSET,
                ),
                *face_primitives,
                DebugPrimitive(
                    kind="frame",
                    pos=pos_tuple,
                    quat=quat_tuple,
                    size=(GOAL_FRAME_AXIS_LENGTH,),
                    rgba=GOAL_FRAME_RGBA,
                ),
                DebugPrimitive(
                    kind="sphere",
                    pos=lamp_pos,
                    size=(GOAL_LAMP_SIZE,),
                    rgba=lamp_rgba,
                ),
            ]
        )
    return overlays


def goal_overlay_getter_for_command(state: Hand2ReorientCommand) -> DebugOverlayGetter:
    """Return the per-frame goal overlay getter bound to a command term."""

    def _get_overlay() -> list[list[DebugPrimitive]]:
        return goal_debug_overlay_primitives(state)

    return _get_overlay


def goal_overlay_getter(env) -> DebugOverlayGetter:
    """Return the per-frame goal overlay getter for a Wuji reorient env."""
    from .commands import task

    return goal_overlay_getter_for_command(task(env))

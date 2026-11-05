# SPDX-License-Identifier: Apache-2.0
# Adapted from Dex-Supermimic tasks/pen_spin/mdp/visualization.py.
# Copyright 2026 Wuji Technology Co., Ltd.
"""Native MuJoCo reference ghosts for source-style Pen Spin playback."""

from __future__ import annotations

import copy

import mujoco
import numpy as np

from .commands import JOINT_NAMES

REFERENCE_COLOR = (1.0, 0.70, 0.80)
HAND_ALPHA = 0.38
PEN_ALPHA = 0.42


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _rotation(quat: np.ndarray) -> np.ndarray:
    result = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(result, quat)
    return result.reshape(3, 3)


class PenReferenceOverlay:
    """Draw the reference without modifying the environment or actual data.

    Mesh IDs remain compatible with the viewer because the ghost owns a deep
    copy of its playback model. In the migrated asset, the pen's three visible
    components are also collision geoms; explicitly include those components
    rather than hiding the pen with a visual-only entity filter.
    """

    def __init__(self, model: mujoco.MjModel, motion):
        self.motion = motion
        self.model = copy.deepcopy(model)
        self.data = mujoco.MjData(self.model)
        self._options = mujoco.MjvOption()
        self._perturb = mujoco.MjvPerturb()
        self._object_q = int(model.jnt_qposadr[model.joint("object/cube_freejoint").id])
        self._joint_q = [int(model.jnt_qposadr[model.joint(name).id]) for name in JOINT_NAMES]
        self._palm_mocap = int(model.body_mocapid[model.body("robot/right_palm_link").id])
        if self._palm_mocap < 0:
            raise ValueError("Pen reference overlay requires the calibrated mocap palm")
        self.model.geom_rgba[:, 3] = 0.0
        self.model.geom_matid[:] = -1
        for index in range(self.model.ngeom):
            name = self.model.geom(index).name
            is_hand = (
                name.startswith("robot/") and name.endswith("_visual")
                and self.model.geom_contype[index] == 0
                and self.model.geom_conaffinity[index] == 0
            )
            is_pen = name in ("object/cube", "object/A_bottom", "object/A_top")
            if is_hand or is_pen:
                self.model.geom_rgba[index] = (*REFERENCE_COLOR, HAND_ALPHA if is_hand else PEN_ALPHA)

    @staticmethod
    def _geom(scene, geom_type, size, position, matrix, rgba):
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError("Pen reference overlay exceeds the viewer's geometry capacity")
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom, geom_type, np.asarray(size, dtype=np.float64),
            np.asarray(position, dtype=np.float64), np.asarray(matrix, dtype=np.float64).reshape(9),
            np.asarray(rgba, dtype=np.float32),
        )
        scene.ngeom += 1
        return geom

    def update(self, user_scn: mujoco.MjvScene, actual_data: mujoco.MjData) -> None:
        """Replace this overlay for the first environment's current reference."""
        del actual_data
        wrist = _numpy(self.motion.wrist_pose_w)[0]
        local_object = _numpy(self.motion.object_pose)[0]
        wrist_rotation = _rotation(wrist[3:])
        object_position = wrist[:3] + wrist_rotation @ local_object[:3]
        object_quaternion = np.empty(4)
        mujoco.mju_mulQuat(object_quaternion, wrist[3:], local_object[3:])
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[self._joint_q] = _numpy(self.motion.joint_pos)[0]
        self.data.qpos[self._object_q:self._object_q + 3] = object_position
        self.data.qpos[self._object_q + 3:self._object_q + 7] = object_quaternion
        self.data.mocap_pos[self._palm_mocap] = wrist[:3]
        self.data.mocap_quat[self._palm_mocap] = wrist[3:]
        mujoco.mj_forward(self.model, self.data)

        user_scn.ngeom = 0
        mujoco.mjv_addGeoms(
            self.model, self.data, self._options, self._perturb,
            mujoco.mjtCatBit.mjCAT_ALL, user_scn,
        )
        reference_links = _numpy(self.motion.link_pos)[0] @ wrist_rotation.T + wrist[:3]
        for position in reference_links:
            self._geom(user_scn, mujoco.mjtGeom.mjGEOM_SPHERE, (0.0045, 0., 0.),
                       position, np.eye(3), (*REFERENCE_COLOR, 0.8))


__all__ = ["PenReferenceOverlay"]

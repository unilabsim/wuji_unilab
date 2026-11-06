"""Reference poses are visualization-only and follow the calibrated wrist."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from unisupermimic.tasks.pen_spin.commands import FIXED_WRIST_POSE_W, JOINT_NAMES
from unisupermimic.tasks.pen_spin.play_overlay import (
    HAND_ALPHA,
    PEN_ALPHA,
    REFERENCE_COLOR,
    PenReferenceOverlay,
)
from unisupermimic.tasks.pen_spin.scene import materialize_scene


@pytest.fixture
def reference(monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/unisupermimic-test-cache")
    model = mujoco.MjModel.from_xml_path(str(materialize_scene()))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    qids = [model.jnt_qposadr[model.joint(name).id] for name in JOINT_NAMES]
    motion = SimpleNamespace(
        wrist_pose_w=torch.tensor([FIXED_WRIST_POSE_W], dtype=torch.float64),
        object_pose=torch.tensor([[0.03, 0.02, 0.08, 1., 0., 0., 0.]], dtype=torch.float64),
        joint_pos=torch.tensor(data.qpos[qids][None] + np.linspace(0.0, .02, 20)[None]),
        link_pos=torch.linspace(-.04, .04, 63, dtype=torch.float64).reshape(1, 21, 3),
    )
    return model, data, motion


def test_overlay_reference_pose_fk_and_scene_leave_actual_state_untouched(reference):
    model, actual, motion = reference
    original = {name: getattr(model, name).copy() for name in ("geom_rgba", "geom_matid", "geom_size")}
    actual_state = {name: getattr(actual, name).copy() for name in ("qpos", "qvel", "xpos", "mocap_pos", "mocap_quat")}
    overlay = PenReferenceOverlay(model, motion)
    scene = mujoco.MjvScene(model, maxgeom=256)
    overlay.update(scene, actual)

    palm = model.body("robot/right_palm_link").id
    expected_position = actual.xpos[palm] + actual.xmat[palm].reshape(3, 3) @ motion.object_pose[0, :3].numpy()
    obj = model.body("object/cube").id
    np.testing.assert_allclose(overlay.data.xpos[obj], expected_position, atol=1e-12)
    np.testing.assert_allclose(overlay.data.xquat[obj], motion.wrist_pose_w[0, 3:].numpy(), atol=1e-12)
    qids = [model.jnt_qposadr[model.joint(name).id] for name in JOINT_NAMES]
    np.testing.assert_allclose(overlay.data.qpos[qids], motion.joint_pos[0].numpy())
    # Reference hand meshes, pen and link dots remain; coordinate arrows are hidden.
    assert scene.ngeom > 21 + 3
    assert all(scene.geoms[index].type != mujoco.mjtGeom.mjGEOM_ARROW for index in range(scene.ngeom))
    count = scene.ngeom
    motion.object_pose[0, 0] += .01
    overlay.update(scene, actual)
    assert scene.ngeom == count  # Replace the old overlay, never append forever.
    np.testing.assert_allclose(
        overlay.data.xpos[obj],
        expected_position + actual.xmat[palm].reshape(3, 3)[:, 0] * .01,
        atol=1e-12,
    )
    for name, value in original.items():
        np.testing.assert_array_equal(getattr(model, name), value)
    for name, value in actual_state.items():
        np.testing.assert_array_equal(getattr(actual, name), value)


def test_ghost_uses_source_color_alpha_and_hides_collision_meshes(reference):
    model, actual, motion = reference
    overlay = PenReferenceOverlay(model, motion)
    for name in ("object/cube", "object/A_bottom", "object/A_top"):
        geom = overlay.model.geom(name).id
        np.testing.assert_allclose(overlay.model.geom_rgba[geom], (*REFERENCE_COLOR, PEN_ALPHA))
        assert overlay.model.geom_matid[geom] == -1
    hand = overlay.model.geom("robot/right_finger2_link2_visual").id
    np.testing.assert_allclose(overlay.model.geom_rgba[hand], (*REFERENCE_COLOR, HAND_ALPHA))
    for name in ("floor", "robot/right_finger2_link2_col_mesh_disabled", "robot/right_finger2_link2_col"):
        assert overlay.model.geom_rgba[overlay.model.geom(name).id, 3] == 0.0
    scene = mujoco.MjvScene(model, maxgeom=256)
    overlay.update(scene, actual)
    expected_links = (
        motion.link_pos[0].numpy() @ actual.xmat[model.body("robot/right_palm_link").id].reshape(3, 3).T
        + motion.wrist_pose_w[0, :3].numpy()
    )
    np.testing.assert_allclose(
        [scene.geoms[index].pos for index in range(scene.ngeom - 21, scene.ngeom)],
        expected_links,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        [scene.geoms[index].rgba for index in range(scene.ngeom - 21, scene.ngeom)],
        np.broadcast_to((*REFERENCE_COLOR, 0.8), (21, 4)),
    )

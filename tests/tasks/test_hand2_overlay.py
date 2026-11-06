from types import SimpleNamespace

import pytest
import torch

from unisupermimic.tasks.dex_reorient.hand2 import overlay


def _state(*, window=0, within=False):
    return SimpleNamespace(
        cube_position=lambda: torch.tensor([[0.0, 0.0, 0.5]]),
        goal=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        window=torch.tensor([window]),
        within=torch.tensor([within]),
    )


def test_hand2_goal_overlay_has_colored_marker_and_status_lamp():
    primitives = overlay.goal_debug_overlay_primitives(_state())[0]
    assert [item.kind for item in primitives] == [
        "ghost_geom",
        "box",
        "box",
        "box",
        "box",
        "box",
        "box",
        "frame",
        "sphere",
    ]
    ghost, *faces, frame, lamp = primitives
    assert ghost.rgba == overlay.GOAL_GHOST_RGBA
    assert [face.rgba for face in faces] == list(overlay.GOAL_FACE_RGBA)
    assert [face.size for face in faces] == [
        (overlay.GOAL_FACE_THICKNESS, overlay.GOAL_FACE_HALF_SIZE, overlay.GOAL_FACE_HALF_SIZE),
        (overlay.GOAL_FACE_THICKNESS, overlay.GOAL_FACE_HALF_SIZE, overlay.GOAL_FACE_HALF_SIZE),
        (overlay.GOAL_FACE_HALF_SIZE, overlay.GOAL_FACE_THICKNESS, overlay.GOAL_FACE_HALF_SIZE),
        (overlay.GOAL_FACE_HALF_SIZE, overlay.GOAL_FACE_THICKNESS, overlay.GOAL_FACE_HALF_SIZE),
        (overlay.GOAL_FACE_HALF_SIZE, overlay.GOAL_FACE_HALF_SIZE, overlay.GOAL_FACE_THICKNESS),
        (overlay.GOAL_FACE_HALF_SIZE, overlay.GOAL_FACE_HALF_SIZE, overlay.GOAL_FACE_THICKNESS),
    ]
    # The identity goal quaternion makes the face order directly observable:
    # [+X, -X, +Y, -Y, +Z, -Z].
    goal_center = (0.0, 0.0, 0.5 + overlay.GOAL_VIS_Z_OFFSET)
    face_offset = overlay.GOAL_FACE_HALF_SIZE + overlay.GOAL_FACE_THICKNESS
    assert faces[0].pos == pytest.approx((face_offset, 0.0, goal_center[2]))
    assert faces[1].pos == pytest.approx((-face_offset, 0.0, goal_center[2]))
    assert faces[2].pos == pytest.approx((0.0, face_offset, goal_center[2]))
    assert faces[3].pos == pytest.approx((0.0, -face_offset, goal_center[2]))
    assert faces[4].pos == pytest.approx((0.0, 0.0, goal_center[2] + face_offset))
    assert faces[5].pos == pytest.approx((0.0, 0.0, goal_center[2] - face_offset))
    assert all(face.quat == (1.0, 0.0, 0.0, 0.0) for face in faces)
    assert frame.size == (overlay.GOAL_FRAME_AXIS_LENGTH,)
    assert lamp.rgba == overlay.GOAL_LAMP_RED_RGBA
    assert lamp.pos[2] == pytest.approx(goal_center[2] + overlay.GOAL_LAMP_OFFSET[2])


@pytest.mark.parametrize(
    ("window", "within", "expected"),
    [
        (0, False, overlay.GOAL_LAMP_RED_RGBA),
        (0, True, overlay.GOAL_LAMP_AMBER_RGBA),
        (1, False, overlay.GOAL_LAMP_GREEN_RGBA),
    ],
)
def test_hand2_goal_overlay_lamp_tracks_persistent_goal_state(window, within, expected):
    lamp = overlay.goal_debug_overlay_primitives(_state(window=window, within=within))[0][-1]
    assert lamp.rgba == expected

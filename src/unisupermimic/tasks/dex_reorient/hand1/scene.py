"""Compose the source Hand 1 robot, task rig and cube for UniLab.

The packaged robot is the upstream ``body-with-soft`` model after the same
contact/site/exclusion edits as Dex-Supermimic's ``assets/wuji_hand.py``.
Hand 2's capsule overlay and measured actuator table are deliberately absent.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import mujoco

from wuji_unilab.assets import ASSET_ROOT, verify_assets

HAND_XML = ASSET_ROOT / "robots/wuji_hand/mjcf/right_mjlab.xml"
HAND_MESH_DIR = ASSET_ROOT / "robots/wuji_hand/meshes/right"
CUBE_ROOT = Path(__file__).resolve().parents[3] / "assets/objects/inhand_cube"
_HOME_CUBE_QPOS = (-0.0967, 0.0100, 0.5599, 1.0, 0.0, 0.0, 0.0)
_HOME_PALM_POS = (0.0, 0.0, 0.5)
_HOME_PALM_QUAT = (0.70710678, 0.0, -0.70710678, 0.0)
_HOME_JOINT_POS = (
    0.8, -0.0215, 0.285, 0.582, 0.441, 0.163, 0.822, 0.494,
    0.448, -0.0832, 0.883, 0.305, 0.594, -0.268, 1.13, 0.18,
    1.2, -0.228, 0.794, 0.407,
)
_TAG_POS = (0.0262, 0.0, -0.0563)
_TAG_QUAT = (0.70710678, 0.0, 0.70710678, 0.0)

# Read the robot's own position actuators; do not reuse Hand 2 calibration.
_ACTUATORS = ET.parse(HAND_XML).getroot().findall("./actuator/position")
_GAINS = tuple((float(a.get("kp")), float(a.get("kv"))) for a in _ACTUATORS)
_FORCE_LIMITS = tuple(max(abs(float(v)) for v in a.get("forcerange").split()) for a in _ACTUATORS)


@lru_cache(maxsize=1)
def materialize_scene() -> Path:
    hashes = verify_assets()
    key = hashlib.sha256(
        (str(ASSET_ROOT.resolve()) + repr(sorted(hashes.items()))).encode()
        + Path(__file__).read_bytes()
        + (CUBE_ROOT / "xmls/cube.xml").read_bytes()
    ).hexdigest()[:24]
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "unisupermimic"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"hand1_reorient_{key}.xml"
    if path.is_file():
        return path
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.01
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
    # Match the asset while attaching it; final task overrides are applied
    # after composition below, as in the source SimulationCfg.
    spec.option.iterations = 5
    spec.option.ls_iterations = 8
    spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
    cube = mujoco.MjSpec.from_file(str(CUBE_ROOT / "xmls/cube.xml"))
    cube.meshdir = str(CUBE_ROOT)
    cube.texturedir = str(CUBE_ROOT)
    cube.body("cube").pos = _HOME_CUBE_QPOS[:3]
    cube.body("cube").quat = _HOME_CUBE_QPOS[3:]
    spec.attach(cube, prefix="object/", frame=spec.worldbody.add_frame())
    hand = mujoco.MjSpec.from_file(str(HAND_XML))
    hand.meshdir = str(HAND_MESH_DIR)
    palm = hand.body("right_palm_link")
    palm.mocap = True
    palm.pos = _HOME_PALM_POS
    palm.quat = _HOME_PALM_QUAT
    tag = palm.add_body(name="right_wrist_tag_frame", pos=_TAG_POS, quat=_TAG_QUAT)
    tag.add_site(name="right_wrist_tag", type=mujoco.mjtGeom.mjGEOM_BOX,
                 size=(0.0252, 0.0252, 0.0006), rgba=(1, 0.9, 0, 0.5), group=1)
    spec.attach(hand, prefix="robot/", frame=spec.worldbody.add_frame())
    # The source task overrides the asset's 5/8 solver settings with 10/20.
    # This final setting must reach both the trainer and the playback model.
    spec.option.iterations = 10
    spec.option.ls_iterations = 20
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=(5, 5, 0.05), rgba=(0.15, 0.18, 0.22, 1))
    spec.worldbody.add_light(pos=(0, 0, 2), dir=(0, 0, -1))

    def contact(name, primary, mode="body", secondary="object/cube", subtree=False):
        spec.add_sensor(
            name=name, type=mujoco.mjtSensor.mjSENS_CONTACT,
            objtype=mujoco.mjtObj.mjOBJ_GEOM if mode == "geom" else mujoco.mjtObj.mjOBJ_BODY,
            objname=primary,
            reftype=mujoco.mjtObj.mjOBJ_XBODY if subtree else (
                mujoco.mjtObj.mjOBJ_GEOM if mode == "geom" else mujoco.mjtObj.mjOBJ_BODY
            ),
            refname=secondary, intprm=(1, 0 if subtree else 1, 1),
        )

    for finger in range(1, 6):
        contact(f"tip_contact_{finger}", f"robot/right_finger{finger}_link4_col", "geom")
        for link in (1, 2):
            contact(f"palm_contact_finger{finger}_link{link}", f"robot/right_finger{finger}_link{link}")
        for link in (3, 4):
            contact(f"distal_contact_{finger}_link{link}", f"robot/right_finger{finger}_link{link}")
    contact("palm_contact_palm", "robot/right_palm_link")
    collisions = [g.name for g in spec.geoms if re.fullmatch(r"robot/right_finger.*_col", g.name)]
    if len(collisions) != 16:
        raise ValueError(f"Hand 1 source requires 16 finger collision slots, got {len(collisions)}")
    for index, geom in enumerate(collisions):
        contact(f"finger_collision_{index}", geom, "geom", "robot/right_palm_link", True)
    for index, actuator in enumerate(spec.actuators):
        spec.add_sensor(name=f"actuator_force_{index}", type=mujoco.mjtSensor.mjSENS_ACTUATORFRC,
                        objtype=mujoco.mjtObj.mjOBJ_ACTUATOR, objname=actuator.name)
    for finger in range(1, 6):
        spec.add_sensor(name=f"tip_velocity_{finger}", type=mujoco.mjtSensor.mjSENS_FRAMELINVEL,
                        objtype=mujoco.mjtObj.mjOBJ_SITE, objname=f"robot/right_finger{finger}_tip")
    spec.add_key(name="home", qpos=(*_HOME_CUBE_QPOS, *_HOME_JOINT_POS), ctrl=_HOME_JOINT_POS,
                 mpos=_HOME_PALM_POS, mquat=_HOME_PALM_QUAT)
    root = ET.fromstring(spec.to_xml())
    # Startup DR moves the cube COM and changes each hand link's inertia.
    # These bodies must keep the general inertia path after compilation;
    # MuJoCo's simple-body optimization otherwise assumes an unchanged COM.
    for body in root.iter("body"):
        if body.get("name", "").startswith(("object/", "robot/")):
            body.set("simple", "false")
    for element in root.findall("./asset/*"):
        base = HAND_MESH_DIR if element.get("name", "").startswith("robot/") else CUBE_ROOT
        for name, value in list(element.attrib.items()):
            if name.startswith("file"):
                element.set(name, str(base / value))
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=cache, delete=False) as stream:
        stream.write(ET.tostring(root, encoding="unicode"))
        staged = Path(stream.name)
    try:
        model = mujoco.MjModel.from_xml_path(str(staged))
        if (model.nq, model.nv, model.nu, model.nmocap) != (27, 26, 20, 1):
            raise ValueError("Unexpected Hand 1 model dimensions")
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)
    return path


__all__ = ["materialize_scene"]

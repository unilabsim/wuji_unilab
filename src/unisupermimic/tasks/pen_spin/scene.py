"""Repository-local MuJoCo scene for Wuji Hand 2 pen spin.

The hand composition is shared with the migrated Hand 2 reorientation scene,
but the free object is replaced by the 250 mm tracking pen from
Dex-SuperMimic.  The source hand scene is never modified; this module writes a
separate cached XML keyed by both scene sources and the pen asset.
"""

from __future__ import annotations

import hashlib
import os
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import mujoco

from unisupermimic.tasks.dex_reorient.hand2.scene import (
    materialize_scene as materialize_hand2_scene,
)

ROOT = Path(__file__).resolve().parents[4]
PEN_ROOT = ROOT / "src/unisupermimic/assets/objects/pen250"
PEN_XML = PEN_ROOT / "xmls/pen_tracking.xml"
_PEN_MASS_KG = 0.030
_PEN_WRIST_POSE = (
    -0.028110361960901435,
    0.00300004,
    0.4952974018391883,
    0.5416752204197018,
    0.45451947767204376,
    0.45451947767204376,
    0.5416752204197018,
)

# Pen Spin uses the identified Hand 2 controller and joint parameters, which
# differ from the cube-reorientation calibration in the shared scene. Keep
# these values local so materializing Pen Spin cannot mutate Hand2 Reorient.
_TORQUE_CONSTANT = (
    0.4136, 0.3545, 0.2216, 0.1329,
    0.4136, 0.1673, 0.1673, 0.1673,
    0.4136, 0.1673, 0.1673, 0.1673,
    0.4136, 0.1673, 0.1673, 0.1673,
    0.4461, 0.1115, 0.1115, 0.1115,
)
_MIT_KD = (0.075, 0.07, 0.05, 0.05)
_AMPS_PER_MIT = 1.9114709702965849
_PEN_KP = tuple(value * _AMPS_PER_MIT * 3.0 for value in _TORQUE_CONSTANT)
_PEN_KV = tuple(
    value * _AMPS_PER_MIT * _MIT_KD[index % 4]
    for index, value in enumerate(_TORQUE_CONSTANT)
)
_PEN_FORCE_LIMIT = tuple(value * 1.5 for value in _TORQUE_CONSTANT)
_PEN_ARMATURE = (
    1.457162e-4, 1.366389e-4, 1.273553e-4, 8.768902e-5,
    1.130474e-4, 5.127161e-5, 4.843783e-5, 7.906193e-5,
    2.011048e-4, 7.106655e-5, 6.054746e-5, 5.659272e-5,
    1.393897e-4, 3.544108e-5, 5.233974e-5, 6.047948e-5,
    9.706259e-5, 3.826054e-5, 4.664456e-5, 4.743778e-5,
)
_PEN_DAMPING = (
    0.0026915050, 0.0042812198, 0.0014135711, 0.0005131806,
    0.0010510923, 0.0, 0.0010302809, 0.0000663526,
    0.0021587672, 0.0008842859, 0.0012610488, 0.0001128493,
    0.0014506542, 0.0, 0.0007149635, 0.0,
    0.0, 0.0, 0.0016246495, 0.0,
)
_PEN_FRICTIONLOSS = (
    0.0367355430, 0.0342266392, 0.0187921182, 0.0122545518,
    0.0377084683, 0.0177932607, 0.0297927367, 0.0161822467,
    0.0355644840, 0.0240616629, 0.0127755677, 0.0109169800,
    0.0528257763, 0.0266927477, 0.0148677454, 0.0215947297,
    0.0338251954, 0.0261331247, 0.0142008972, 0.0129471168,
)


def _apply_pen_hand_calibration(root: ET.Element) -> None:
    """Apply source Pen Spin gains, joint damping, and contact dimensions."""
    joints = {
        item.get("name"): item
        for item in root.iter("joint")
        if item.get("name", "").startswith("robot/right_finger")
    }
    actuators = {
        item.get("name"): item
        for item in root.iter("general")
        if item.get("name", "").startswith("robot/right_finger")
    }
    joint_names = tuple(
        f"robot/right_finger{finger}_joint{joint}"
        for finger in range(1, 6)
        for joint in range(1, 5)
    )
    for index, joint_name in enumerate(joint_names):
        joint = joints.get(joint_name)
        actuator = actuators.get(f"{joint_name}_actuator")
        if joint is None or actuator is None:
            raise ValueError(f"pen scene is missing calibrated joint/actuator {joint_name}")
        joint.set("armature", f"{_PEN_ARMATURE[index]:.17g}")
        joint.set("damping", f"{_PEN_DAMPING[index]:.17g}")
        joint.set("frictionloss", f"{_PEN_FRICTIONLOSS[index]:.17g}")
        actuator.set("gainprm", f"{_PEN_KP[index]:.17g}")
        actuator.set("biasprm", f"0 {-_PEN_KP[index]:.17g} {-_PEN_KV[index]:.17g}")
        actuator.set("forcerange", f"{-_PEN_FORCE_LIMIT[index]:.17g} {_PEN_FORCE_LIMIT[index]:.17g}")

    # The source task uses condim 4 on palm/finger/tip contacts. The shared
    # cube task uses condim 3, so update only this standalone XML.
    for geom in root.iter("geom"):
        name = geom.get("name", "")
        if name == "robot/right_palm_collision" or (
            name.startswith("robot/right_finger")
            and ("_link" in name or "_tip_sensor" in name)
            and name.endswith("_col")
        ):
            geom.set("condim", "4")


@lru_cache(maxsize=1)
def materialize_scene() -> Path:
    """Build and validate a standalone Hand 2 + pen scene once per process."""
    if not PEN_XML.is_file():
        raise FileNotFoundError(f"pen MJCF is missing: {PEN_XML}")
    source = materialize_hand2_scene()
    digest = hashlib.sha256(
        source.read_bytes() + PEN_XML.read_bytes()
        + b"".join(path.read_bytes() for path in sorted((PEN_ROOT / "textures").glob("*.png")))
        + Path(__file__).read_bytes()
    ).hexdigest()[:20]
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", ROOT / ".cache")) / "unisupermimic"
    cache_root.mkdir(parents=True, exist_ok=True)
    output = cache_root / f"hand2_pen_spin_{digest}.xml"
    if output.is_file():
        return output

    root = ET.fromstring(source.read_text(encoding="utf-8"))
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError("Hand 2 scene is missing asset/worldbody sections")
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    # Source Pen Spin runs four 5 ms physics steps per 50 Hz control tick.
    # The shared cube scene is 10 ms, so this task must own its option block.
    option.set("timestep", "0.005")
    option.set("integrator", "implicitfast")
    option.set("iterations", "10")
    option.set("ls_iterations", "20")
    option.set("impratio", "2")
    _apply_pen_hand_calibration(root)
    object_body = next((body for body in worldbody.iter("body") if body.get("name") == "object/cube"), None)
    if object_body is None:
        raise ValueError("Hand 2 scene is missing object/cube")
    palm_body = next(
        (body for body in worldbody.iter("body") if body.get("name") == "robot/right_palm_link"),
        None,
    )
    if palm_body is None:
        raise ValueError("Hand 2 scene is missing robot/right_palm_link")
    # On MuJoCo's host reset path mocap is restored from the XML body default.
    # Use the same fixed wrist frame as the source clip command; otherwise the
    # first reset would place the reference pen several millimetres from the
    # actual palm and immediately trigger the loss termination.
    palm_body.set("pos", " ".join(f"{value:.17g}" for value in _PEN_WRIST_POSE[:3]))
    palm_body.set("quat", " ".join(f"{value:.17g}" for value in _PEN_WRIST_POSE[3:]))
    # Source PEN_SPIN_HAND2_ROBOT_INIT_STATE has zero default joints. These
    # are the actor's q_default, even though RSI resets physical joints to a
    # clip frame. Inheriting the cube grasp key changes every actor input.
    home = root.find("./keyframe/key[@name='home']")
    if home is None:
        raise ValueError("pen scene is missing the home keyframe")
    home_qpos = home.get("qpos", "").split()
    if len(home_qpos) != 27:
        raise ValueError("pen scene expects a 27D home qpos")
    home.set("qpos", " ".join(home_qpos[:7] + ["0"] * 20))
    home.set("qvel", " ".join(["0"] * 26))
    home.set("ctrl", " ".join(["0"] * 20))
    home.set("mpos", " ".join(f"{value:.17g}" for value in _PEN_WRIST_POSE[:3]))
    home.set("mquat", " ".join(f"{value:.17g}" for value in _PEN_WRIST_POSE[3:]))

    # Keep the body's freejoint and center site, replacing only object geoms.
    for child in list(object_body):
        if child.tag == "geom":
            object_body.remove(child)
    texture_specs = {
        "object/pen_bottom_texture": {
            "type": "cube", "fileright": "bottom_right.png", "fileleft": "bottom_left.png",
            "fileup": "bottom_up.png", "filedown": "bottom_down.png", "filefront": "white.png",
            "fileback": "bottom_back.png",
        },
        "object/pen_top_texture": {
            "type": "cube", "fileright": "top_right.png", "fileleft": "top_left.png",
            "fileup": "top_up.png", "filedown": "top_down.png", "filefront": "top_front.png",
            "fileback": "white.png",
        },
    }
    for name, attrs in texture_specs.items():
        attrs = {key: str(value) for key, value in attrs.items()}
        attrs.update({key: str(PEN_ROOT / "textures" / value) for key, value in list(attrs.items()) if key.startswith("file")})
        ET.SubElement(asset, "texture", name=name, **attrs)
    ET.SubElement(asset, "material", name="object/pen_white", rgba="1 1 1 1", specular="0")
    ET.SubElement(asset, "material", name="object/pen_bottom_material", texture="object/pen_bottom_texture", texuniform="false", specular="0")
    ET.SubElement(asset, "material", name="object/pen_top_material", texture="object/pen_top_texture", texuniform="false", specular="0")
    # The shared object's childclass carries the cube task's contact tuning.
    # The source tracking pen uses MuJoCo's defaults for these three fields.
    # Keep them explicit so the cube default cannot leak into this task.
    pen_contact = {
        "friction": "1 .005 .0001",
        "solref": ".02 1",
        "solimp": ".9 .95 .001 .5 2",
    }
    ET.SubElement(
        object_body, "geom", name="object/cube", type="cylinder", fromto="-.089 0 0 .089 0 0",
        size=".00745", material="object/pen_white", group="2", contype="1", conaffinity="2",
        density="469.05012453696503", **pen_contact,
    )
    for name, pos, material in (
        ("object/A_bottom", "-.107 0 0", "object/pen_bottom_material"),
        ("object/A_top", ".107 0 0", "object/pen_top_material"),
    ):
        ET.SubElement(
            object_body, "geom", name=name, type="box", pos=pos, size=".009 .009 .018",
            quat=".7071067811865476 0 .7071067811865476 0", material=material, group="2",
            contype="1", conaffinity="2", density="469.05012453696503", **pen_contact,
        )
    # The source asset explicitly scales the tracking pen's compiled inertia
    # to 30 g. The density-only XML defaults to about 25.5 g, so add an
    # explicit inertial element after probing the geometry-derived inertia.
    probe = output.with_suffix(".probe.xml")
    probe.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    try:
        probe_model = mujoco.MjModel.from_xml_path(str(probe))
        object_id = mujoco.mj_name2id(
            probe_model, mujoco.mjtObj.mjOBJ_BODY, "object/cube"
        )
        base_mass = float(probe_model.body_mass[object_id])
        if not base_mass > 0.0:
            raise ValueError("pen geometry produced a non-positive mass")
        scale = _PEN_MASS_KG / base_mass
        inertial = next((child for child in object_body if child.tag == "inertial"), None)
        if inertial is not None:
            object_body.remove(inertial)
        ET.SubElement(
            object_body,
            "inertial",
            mass=f"{_PEN_MASS_KG:.17g}",
            pos=" ".join(f"{value:.17g}" for value in probe_model.body_ipos[object_id]),
            quat=" ".join(f"{value:.17g}" for value in probe_model.body_iquat[object_id]),
            diaginertia=" ".join(
                f"{value * scale:.17g}" for value in probe_model.body_inertia[object_id]
            ),
        )
    finally:
        probe.unlink(missing_ok=True)
    # Hand2 sensors refer to the object body, so their names remain stable.
    payload = ET.tostring(root, encoding="unicode")
    temporary = output.with_suffix(".tmp.xml")
    temporary.write_text(payload, encoding="utf-8")
    try:
        model = mujoco.MjModel.from_xml_path(str(temporary))
        if (model.nq, model.nv, model.nu, model.nmocap) != (27, 26, 20, 1):
            raise ValueError(
                "Unexpected Hand 2 pen dimensions: "
                f"nq={model.nq}, nv={model.nv}, nu={model.nu}, nmocap={model.nmocap}"
            )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


__all__ = ["materialize_scene"]

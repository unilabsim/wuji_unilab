"""Cold-path MuJoCo composition for the Wuji Hand 2 reorientation task.

The source project builds this scene from mjlab entities.  UniLab's public
backend factory accepts one MJCF scene, so the hand and cube are attached once
and cached.  No scene construction is performed during reset or stepping.
"""

from __future__ import annotations

import hashlib
import os
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
HAND_XML = ROOT / "third_party/wuji-description/src/wuji_description/resources/hand2/hand2_2/body/mjcf/right.xml"
HAND_MESH_DIR = HAND_XML.parent.parent / "meshes/right"
CUBE_ROOT = ROOT / "src/unisupermimic/assets/objects/inhand_cube"
JIG_TABLE = ROOT / "src/unisupermimic/assets/robots/wuji_hand2/jig_table.STL"

# The calibrated mount frame is expressed relative to the palm.  The rig
# poses below are expressed in that calibrated frame, not directly in the
# palm frame or the resource MJCF's actual right_mount body.
_CALIBRATED_MOUNT_POS = (-0.0030000041, -0.000250110803, 0.0285)
_CALIBRATED_MOUNT_QUAT = (1.0, 0.0, 0.0, -0.0000081995)
_JIG_POS = (0.0, -0.1062088, 0.0178309)
_JIG_QUAT = (0.541668573, -0.454527399, 0.454527399, 0.541668573)
_TAG_POS = (0.0, 0.037408, 0.034737)
_TAG_QUAT = (0.5, -0.5, -0.5, -0.5)

# Measured Wuji Hand 2 reorientation gains and torque limits.  The order is
# thumb, index, middle, ring, pinky; each finger follows joint1..joint4.
_GAINS = (
    (1.354985, 0.04769), (2.198821, 0.062838), (0.886467, 0.027982), (0.654026, 0.020457),
    (1.345207, 0.047402), (0.88946, 0.027239), (0.779361, 0.024489), (0.656424, 0.020508),
    (1.299313, 0.045699), (0.942641, 0.027846), (0.73725, 0.023408), (0.710314, 0.022366),
    (1.276892, 0.046209), (1.013334, 0.028704), (0.859159, 0.027436), (0.648064, 0.020448),
    (1.102915, 0.043816), (0.910201, 0.026593), (0.861968, 0.027476), (0.656704, 0.020611),
)
_FORCE_LIMITS = (1.02, 0.83, 0.35, 0.21, 1.05, 0.38, 0.31, 0.30, 1.06, 0.39, 0.32, 0.29,
                 1.02, 0.41, 0.28, 0.30, 0.87, 0.36, 0.28, 0.30)
# Dex-Supermimic's calibrated Hand 2 cage pose.  The composed model orders
# the cube free joint before the twenty hand hinges; the palm is mocap and is
# therefore represented by ``mpos``/``mquat`` in the keyframe instead of qpos.
_HOME_CUBE_QPOS = (-0.108, 0.0072, 0.5616, 1.0, 0.0, 0.0, 0.0)
_HOME_JOINT_POS = (
    0.8, -0.3, 0.5, 0.4,
    0.45, -0.15, 0.8, 0.5,
    0.45, 0.0, 0.9, 0.3,
    0.6, 0.15, 1.1, 0.2,
    1.2, 0.2, 0.8, 0.4,
)
_HOME_PALM_POS = (-0.0284999, 0.00300004, 0.500250158)
_HOME_PALM_QUAT = (0.5, 0.5, 0.5, 0.5)

# Collision capsules used by Dex-Supermimic's Hand 2 reorientation recipe.
# The upstream visual meshes remain available for rendering, while their
# collision geoms are disabled and replaced by these compact primitives.  The
# values are expressed in each link's local frame as ``fromto`` endpoints and
# a capsule radius.
_COLLISION_CAPSULES = {
    "right_finger1_link1": (((0.0001600994, -0.0025818024, -0.001025414),
                              (0.0010229472, 0.00022010189, -0.014007877)), 0.008326602),
    "right_finger1_link2": (((1.549657e-06, -3.15346e-06, -0.03143319),
                              (1.556016e-06, -3.148208e-06, -0.004805587)), 0.00874906),
    "right_finger1_link3": (((-5.1675061e-05, -9.0148385e-07, -0.034401508),
                              (-5.1675061e-05, -9.0148385e-07, -0.00053389498)), 0.008768497),
    "right_finger2_link1": (((2.0003946e-20, -0.0012000001, -0.00087331082),
                              (-7.5588199e-19, -0.0012000001, -0.01354449)), 0.009388893),
    "right_finger2_link2": (((1.43177e-05, 1.68479e-07, -0.003500009),
                              (-3.205759e-06, 1.684791e-07, -0.03710302)), 0.007507256),
    "right_finger2_link3": (((-1.805026e-18, -1.1052597e-34, -0.029478312),
                              (2.1212593e-21, 1.2988967e-37, 3.4642794e-05)), 0.00801817),
    "right_finger3_link1": (((2.0003946e-20, -0.0012000001, -0.00087331082),
                              (-7.5588199e-19, -0.0012000001, -0.01354449)), 0.009388893),
    "right_finger3_link2": (((1.43177e-05, 1.68479e-07, -0.003500009),
                              (-3.2057593e-06, 1.684791e-07, -0.03710302)), 0.007507256),
    "right_finger3_link3": (((-1.805026e-18, -1.1052597e-34, -0.029478312),
                              (2.1212593e-21, 1.2988967e-37, 3.4642794e-05)), 0.00801817),
    "right_finger4_link1": (((2.0003946e-20, -0.0012000001, -0.00087331082),
                              (-7.5588199e-19, -0.0012000001, -0.01354449)), 0.009388893),
    "right_finger4_link2": (((1.43177e-05, 1.68479e-07, -0.003500009),
                              (-3.2057593e-06, 1.684791e-07, -0.03710302)), 0.007507256),
    "right_finger4_link3": (((-1.805026e-18, -1.1052597e-34, -0.029478312),
                              (2.1212593e-21, 1.2988967e-37, 3.4642794e-05)), 0.00801817),
    "right_finger5_link1": (((2.0003975e-20, -0.0012000005, -0.00087331082),
                              (-7.5588196e-19, -0.0012000005, -0.01354449)), 0.009407771),
    "right_finger5_link2": (((-2.479326e-08, 9.787299e-07, -0.03333651),
                              (-2.479326e-08, 9.787299e-07, -0.003263492)), 0.007500022),
    "right_finger5_link3": (((-1.6782627e-18, -1.0276395e-34, -0.02740811),
                              (-2.177578e-21, -1.333382e-37, -3.5562547e-05)), 0.008049908),
}
_SCENE_SCHEMA_VERSION = "hand2-palm-rooted-r2"


@lru_cache(maxsize=1)
def materialize_scene() -> Path:
    """Compose and compile the hand, cube, floor, and task sensors."""
    if not HAND_XML.is_file():
        raise FileNotFoundError(f"Wuji Hand 2 MJCF is missing: {HAND_XML}")
    cube_xml = CUBE_ROOT / "xmls/cube.xml"
    if not cube_xml.is_file():
        raise FileNotFoundError(f"cube MJCF is missing: {cube_xml}")
    # Include every cold-path input in the cache key.  Hashing only the two
    # upstream XML files leaves stale scenes behind when this module changes
    # (for example after adding a sensor, calibration jig, or actuator limit).
    # The repository-local fallback also keeps an implicit invocation inside
    # this checkout; callers can still select an explicit cache with
    # ``XDG_CACHE_HOME`` (the smoke tools use ``/tmp``).
    cache_inputs = (
        # MjSpec serialization can introduce XML syntax unsupported by an
        # older runtime sharing this cache (e.g. actuator input="pos").
        mujoco.__version__.encode(),
        HAND_XML.read_bytes(),
        cube_xml.read_bytes(),
        JIG_TABLE.read_bytes() if JIG_TABLE.is_file() else b"<missing-jig>",
        Path(__file__).read_bytes(),
    )
    source_hash = hashlib.sha256(b"".join(cache_inputs)).hexdigest()[:20]
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", ROOT / ".cache")) / "unisupermimic"
    cache_root.mkdir(parents=True, exist_ok=True)
    output = cache_root / f"hand2_reorient_{source_hash}.xml"
    if output.is_file():
        return output

    spec = mujoco.MjSpec()
    spec.option.timestep = 0.01
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.iterations = 10
    spec.option.ls_iterations = 20

    cube = mujoco.MjSpec.from_file(str(cube_xml))
    # The cube XML references ``meshes/...`` relative to its asset root.
    # Point the compiler at the root so it does not duplicate that segment.
    cube.meshdir = str(CUBE_ROOT)
    cube.texturedir = str(CUBE_ROOT)
    cube.body("cube").pos = (-0.108, 0.0072, 0.5616)
    cube.body("cube").quat = (1.0, 0.0, 0.0, 0.0)
    spec.attach(cube, prefix="object/", frame=spec.worldbody.add_frame())

    hand_xml = _canonical_hand_xml()
    hand = mujoco.MjSpec.from_string(hand_xml)
    # The entity root is the palm.  The upstream XML is mount-rooted;
    # _canonical_hand_xml reframes it so UniLab can apply the configured
    # world pose to ``right_palm_link`` through ``root_body_name``.
    palm = hand.body("right_palm_link")
    palm.mocap = True
    # MuJoCo's host reset path does not expose the mjwarp mocap transaction.
    # Keep the model's native mocap default at the calibrated task pose so an
    # Entity reset also restores the palm when that bridge is unavailable.
    palm.pos = _HOME_PALM_POS
    palm.quat = _HOME_PALM_QUAT
    mount = hand.body("right_mount")
    _add_calibration_rig(hand, mount)
    _apply_collision_overlay(hand)
    for index, actuator in enumerate(hand.actuators):
        kp, kv = _GAINS[index]
        limit = _FORCE_LIMITS[index]
        # MuJoCo 3.11 exposes position-actuator gains through the MjSpec
        # type setter; MjsActuator has no writable ``kp``/``kv`` attributes.
        actuator.set_to_position(kp=kp, kv=kv)
        actuator.forcerange = (-limit, limit)
        actuator.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    hand.option.timestep = 0.01
    hand.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
    hand.option.iterations = 5
    hand.option.ls_iterations = 8
    hand.option.tolerance = 1.0e-8
    hand.option.jacobian = mujoco.mjtJacobian.mjJAC_AUTO
    hand.option.disableflags = int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
    for finger in range(1, 6):
        hand.site(f"right_finger{finger}_tip").group = 4
    # Match the task-level contact solver defaults used by Dex-Supermimic.
    # These assignments are cold-path model metadata and do not add per-step
    # Python work.
    for geom in hand.geoms:
        geom.friction = (0.7, 0.005, 0.0001)
        geom.solref = (0.02, 1.5)
        geom.solimp = (0.9, 0.95, 0.001, 0.5, 2.0)
        geom.priority = 1
    for joint_name, limit in zip(
        (f"right_finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5)),
        _FORCE_LIMITS,
        strict=True,
    ):
        joint = hand.joint(joint_name)
        joint.actfrcrange = (-limit, limit)
        joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
    spec.attach(hand, prefix="robot/", frame=spec.worldbody.add_frame())
    _add_home_keyframe(spec)
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(5.0, 5.0, 0.05),
        friction=(0.7, 0.005, 0.0001), rgba=(0.15, 0.18, 0.22, 1.0),
    )
    spec.worldbody.add_light(pos=(0.0, 0.0, 2.0), dir=(0.0, 0.0, -1.0))

    # The contact sensors are named and grouped so Manager terms can consume
    # them through the packed read plan on both mjwarp and MuJoCo backends.
    obj = mujoco.mjtObj
    sensor = mujoco.mjtSensor
    def add_contact(name: str, body: str, ref: str = "object/cube", *, found: bool = True):
        spec.add_sensor(
            name=name, type=sensor.mjSENS_CONTACT, objtype=obj.mjOBJ_BODY,
            objname=body, reftype=obj.mjOBJ_BODY, refname=ref,
            intprm=(1 if found else 2, 1, 1),
        )

    for finger in range(1, 6):
        add_contact(
            f"tip_contact_{finger}_link4", f"robot/right_finger{finger}_link4"
        )
        add_contact(
            f"tip_contact_{finger}_tip", f"robot/right_finger{finger}_tip_sensor"
        )
    # The source ``palm_object_found`` sensor matches the palm and every
    # finger link1/link2 body.  Keep one binary sensor per body and collapse
    # them into the single proximal channel in ``contact_tensor``.  A force
    # sensor would produce signed values, which are not valid WM BCE targets.
    add_contact("palm_contact_palm", "robot/right_palm_link", found=True)
    for finger in range(1, 6):
        for link in (1, 2):
            add_contact(
                f"palm_contact_finger{finger}_link{link}",
                f"robot/right_finger{finger}_link{link}",
                found=True,
            )
    # Source distal contact groups link3, link4 and the separate tip-sensor
    # body into one binary channel.  Keep the individual body sensors here
    # and reduce them to five channels in ``contact_tensor``.
    for finger in range(1, 6):
        for part, body in (
            ("link3", f"robot/right_finger{finger}_link3"),
            ("link4", f"robot/right_finger{finger}_link4"),
            ("tip", f"robot/right_finger{finger}_tip_sensor"),
        ):
            add_contact(f"distal_contact_{finger}_{part}", body)
    # Dex-SuperMimic's ``finger_collision`` contact sensor binds the 25
    # physical finger collision geoms (four links plus the tip sensor per
    # finger) against the palm subtree.  The source contact layout reserves
    # these channels after the seven object-contact bits.
    collision_geoms = [
        geom.name
        for geom in spec.geoms
        if geom.name.startswith("robot/right_finger")
        and geom.name.endswith("_col")
        and not geom.name.endswith("_col_mesh_disabled")
    ]
    if len(collision_geoms) != 25:
        raise ValueError(
            "Hand2 self-collision sensor contract expects 25 finger geoms, "
            f"got {len(collision_geoms)}"
        )
    for index, geom_name in enumerate(collision_geoms):
        spec.add_sensor(
            name=f"finger_collision_{index}",
            type=sensor.mjSENS_CONTACT,
            objtype=obj.mjOBJ_GEOM,
            objname=geom_name,
            reftype=obj.mjOBJ_XBODY,
            refname="robot/right_palm_link",
            intprm=(1, 0, 1),
        )
    for index, actuator in enumerate(spec.actuators):
        spec.add_sensor(
            name=f"actuator_force_{index}", type=sensor.mjSENS_ACTUATORFRC,
            objtype=obj.mjOBJ_ACTUATOR, objname=actuator.name,
        )
    for index in range(1, 6):
        spec.add_sensor(
            name=f"tip_velocity_{index}", type=sensor.mjSENS_FRAMELINVEL,
            objtype=obj.mjOBJ_SITE, objname=f"robot/right_finger{index}_tip",
        )

    root = ET.fromstring(spec.to_xml())
    # MjSpec.attach does not retain child mesh directories in the serialized
    # XML.  Resolve all asset files to repository-local absolute paths.
    for element in root.findall("./asset/*"):
        prefix = "robot/" if element.attrib.get("name", "").startswith("robot/") else "object/"
        base = HAND_MESH_DIR if prefix == "robot/" else CUBE_ROOT
        for key, value in list(element.attrib.items()):
            if key.startswith("file"):
                element.set(key, str(base / value))
    payload = ET.tostring(root, encoding="unicode")
    temporary = output.with_suffix(".tmp.xml")
    temporary.write_text(payload, encoding="utf-8")
    try:
        model = mujoco.MjModel.from_xml_path(str(temporary))
        if (model.nq, model.nv, model.nu, model.nmocap) != (27, 26, 20, 1):
            raise ValueError(
                "Unexpected Hand 2 dimensions: "
                f"nq={model.nq}, nv={model.nv}, nu={model.nu}, nmocap={model.nmocap}"
            )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _add_home_keyframe(spec: mujoco.MjSpec) -> None:
    """Publish the source task's deterministic ``home`` keyframe."""
    qpos = np.asarray((*_HOME_CUBE_QPOS, *_HOME_JOINT_POS), dtype=np.float64)
    ctrl = np.asarray(_HOME_JOINT_POS, dtype=np.float64)
    if qpos.shape != (27,) or ctrl.shape != (20,):
        raise AssertionError(f"Unexpected Hand 2 home shapes: qpos={qpos.shape}, ctrl={ctrl.shape}")
    spec.add_key(
        name="home",
        qpos=qpos,
        ctrl=ctrl,
        mpos=np.asarray(_HOME_PALM_POS, dtype=np.float64),
        mquat=np.asarray(_HOME_PALM_QUAT, dtype=np.float64),
    )


def _add_calibration_rig(hand: mujoco.MjSpec, mount: Any) -> None:
    """Attach the source task's jig table and wrist tag marker.

    Rig constants are expressed in the calibrated mount frame in
    Dex-Supermimic.  Compose that calibration with each pose before converting
    to the actual mount body's local coordinates.
    """
    jig_pos, jig_quat = _pose_in_mount(mount, _JIG_POS, _JIG_QUAT)
    tag_pos, tag_quat = _pose_in_mount(mount, _TAG_POS, _TAG_QUAT)
    if JIG_TABLE.is_file():
        hand.add_mesh(name="jig_table", file=str(JIG_TABLE), content_type="model/stl")
        mount.add_geom(
            name="jig_table_visual", type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="jig_table", pos=jig_pos, quat=jig_quat,
            contype=0, conaffinity=0, group=1, density=0,
            rgba=(0.35, 0.55, 0.9, 0.55),
        )
    tag = mount.add_body(name="right_wrist_tag_frame", pos=tag_pos, quat=tag_quat)
    tag.add_site(
        name="right_wrist_tag", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.0252, 0.0252, 0.0006), rgba=(1.0, 0.9, 0.0, 0.5), group=1,
    )


def _canonical_hand_xml() -> str:
    """Rename the upstream resource into the task's stable right-hand schema."""
    root = ET.parse(HAND_XML).getroot()
    model_names = {"wujihand2_2-right": "wujihand2-right-mjlab"}
    body_names = {
        "r_mount": "right_mount", "r_wrist": "right_palm_link",
        **{
            f"r_{finger}_{part}": f"right_finger{idx}_link{part_idx}"
            for idx, (finger, parts) in enumerate(
                (("thumb", ("proximal", "proximal_abd", "middle", "distal")),
                 ("index_finger", ("proximal", "proximal_abd", "middle", "distal")),
                 ("middle_finger", ("proximal", "proximal_abd", "middle", "distal")),
                 ("ring_finger", ("proximal", "proximal_abd", "middle", "distal")),
                 ("pinky", ("proximal", "proximal_abd", "middle", "distal"))),
                1,
            )
            for part_idx, part in enumerate(parts, 1)
        },
    }
    # The source uses *_proximal, *_middle and *_distal; the task uses the
    # explicit link1..link4 names.  Keep this table separate from string
    # substitution so references in joints, sites, and excludes stay bijective.
    body_names.update({
        "r_thumb_proximal": "right_finger1_link1", "r_thumb_proximal_abd": "right_finger1_link2",
        "r_thumb_middle": "right_finger1_link3", "r_thumb_distal": "right_finger1_link4",
        "r_index_finger_proximal": "right_finger2_link1", "r_index_finger_proximal_abd": "right_finger2_link2",
        "r_index_finger_middle": "right_finger2_link3", "r_index_finger_distal": "right_finger2_link4",
        "r_middle_finger_proximal": "right_finger3_link1", "r_middle_finger_proximal_abd": "right_finger3_link2",
        "r_middle_finger_middle": "right_finger3_link3", "r_middle_finger_distal": "right_finger3_link4",
        "r_ring_finger_proximal": "right_finger4_link1", "r_ring_finger_proximal_abd": "right_finger4_link2",
        "r_ring_finger_middle": "right_finger4_link3", "r_ring_finger_distal": "right_finger4_link4",
        "r_pinky_proximal": "right_finger5_link1", "r_pinky_proximal_abd": "right_finger5_link2",
        "r_pinky_middle": "right_finger5_link3", "r_pinky_distal": "right_finger5_link4",
        "r_thumb_tip_sensor_frame": "right_finger1_tip_sensor",
        "r_index_finger_tip_sensor_frame": "right_finger2_tip_sensor",
        "r_middle_finger_tip_sensor_frame": "right_finger3_tip_sensor",
        "r_ring_finger_tip_sensor_frame": "right_finger4_tip_sensor",
        "r_pinky_tip_sensor_frame": "right_finger5_tip_sensor",
    })
    joint_names = {
        "r_thumb_cmc_flex": "right_finger1_joint1", "r_thumb_cmc_abd": "right_finger1_joint2",
        "r_thumb_mcp": "right_finger1_joint3", "r_thumb_ip": "right_finger1_joint4",
        **{
            f"r_{finger}_{joint}": f"right_finger{idx}_joint{jidx}"
            for idx, finger in enumerate(("index_finger", "middle_finger", "ring_finger", "pinky"), 2)
            for jidx, joint in enumerate(("mcp_flex", "mcp_abd", "pip", "dip"), 1)
        },
    }
    site_names = {
        "r_thumb_tip": "right_finger1_tip", "r_index_finger_tip": "right_finger2_tip",
        "r_middle_finger_tip": "right_finger3_tip", "r_ring_finger_tip": "right_finger4_tip",
        "r_pinky_tip": "right_finger5_tip", "r_wrist_imu_frame": "right_wrist_imu_frame",
    }
    actuator_names = {
        **{f"r_{prefix}{j}": f"right_finger{idx}_joint{j+1}_actuator"
           for idx, prefix in enumerate(("THJ", "FFJ", "MFJ", "RFJ", "LFJ"), 1)
           for j in range(4)},
    }
    mesh_names = {
        "r_mount": "r_mount", "r_wrist": "r_wrist",
        **{old: old for old in (
            "r_thumb_proximal", "r_thumb_proximal_abd", "r_thumb_middle", "r_thumb_distal",
            "r_thumb_tip_sensor_frame", "r_index_finger_proximal", "r_index_finger_proximal_abd",
            "r_index_finger_middle", "r_index_finger_distal", "r_index_finger_tip_sensor_frame",
            "r_middle_finger_proximal", "r_middle_finger_proximal_abd", "r_middle_finger_middle",
            "r_middle_finger_distal", "r_middle_finger_tip_sensor_frame", "r_ring_finger_proximal",
            "r_ring_finger_proximal_abd", "r_ring_finger_middle", "r_ring_finger_distal",
            "r_ring_finger_tip_sensor_frame", "r_pinky_proximal", "r_pinky_proximal_abd",
            "r_pinky_middle", "r_pinky_distal", "r_pinky_tip_sensor_frame")},
    }
    if root.get("model") in model_names:
        root.set("model", model_names[root.get("model")])
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", str(HAND_MESH_DIR))
    # Names overlap across MJCF namespaces (for example ``r_wrist`` is both
    # a body and a mesh).  Applying one flattened map would let the mesh map
    # silently undo the body rename.  Resolve every attribute in its own
    # namespace instead.
    for element in root.iter():
        if element.tag == "body":
            maps = {"name": body_names, "body1": body_names, "body2": body_names}
        elif element.tag == "joint":
            maps = {"name": joint_names}
        elif element.tag == "site":
            maps = {"name": site_names}
        elif element.tag in ("actuator", "position", "motor", "velocity"):
            maps = {"name": actuator_names, "joint": joint_names, "target": joint_names}
        elif element.tag == "mesh":
            maps = {"name": mesh_names}
        else:
            maps = {"joint": joint_names, "mesh": mesh_names,
                    "body1": body_names, "body2": body_names,
                    "target": joint_names}
        for attr, mapping in maps.items():
            value = element.get(attr)
            if value in mapping:
                element.set(attr, mapping[value])
        # Preserve one explicit visual and collision name per body.  The
        # upstream XML intentionally leaves geoms unnamed.
        if element.tag == "body":
            body = element.get("name")
            geoms = [child for child in element if child.tag == "geom"]
            if body and geoms:
                for index, geom in enumerate(geoms):
                    if index == 0:
                        name = "right_palm_visual" if body == "right_palm_link" else f"{body}_visual"
                    elif body == "right_palm_link":
                        name = "right_palm_collision"
                    elif body == "right_mount":
                        name = "right_mount_collision"
                    elif body.endswith("tip_sensor"):
                        name = f"{body}_col"
                    else:
                        name = f"{body}_col"
                    geom.set("name", name)
    sensor = root.find("sensor")
    if sensor is not None:
        root.remove(sensor)
    _palm_root_scene(root)
    return ET.tostring(root, encoding="unicode")


def _palm_root_scene(root: ET.Element) -> None:
    """Reframe the upstream mount-rooted XML around ``right_palm_link``.

    UniLab's ``root_body_name`` initialization writes the world pose of the
    selected body.  Keeping ``right_mount`` as the world body would therefore
    apply the configured pose to the wrong frame and shift every finger.
    """
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Hand 2 MJCF has no worldbody")
    mount = next((item for item in worldbody if item.tag == "body" and item.get("name") == "right_mount"), None)
    if mount is None:
        raise ValueError("Hand 2 MJCF is missing right_mount body")
    palm = next((item for item in mount if item.tag == "body" and item.get("name") == "right_palm_link"), None)
    if palm is None:
        raise ValueError("Hand 2 MJCF is missing right_palm_link child")
    pos = np.fromstring(palm.get("pos", "0 0 0"), sep=" ", dtype=np.float64)
    quat = np.fromstring(palm.get("quat", "1 0 0 0"), sep=" ", dtype=np.float64)
    quat /= np.linalg.norm(quat)
    mount_pos = np.zeros(3, dtype=np.float64)
    mount_quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_negPose(mount_pos, mount_quat, pos, quat)
    mount.set("pos", " ".join(f"{value:.17g}" for value in mount_pos))
    mount.set("quat", " ".join(f"{value:.17g}" for value in mount_quat))
    mount.remove(palm)
    palm.attrib.pop("pos", None)
    palm.attrib.pop("quat", None)
    # ResetWuji writes the selected root pose through MuJoCo's mocap state.
    # Keep the palm as the single movable root, as in the source task.
    palm.set("mocap", "true")
    palm.append(mount)
    worldbody.remove(mount)
    worldbody.append(palm)


def _pose_in_mount(mount: Any, pos: tuple[float, ...], quat: tuple[float, ...]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Convert a calibrated-mount-frame pose to the actual mount body."""
    mount_pos = np.asarray(mount.pos, dtype=np.float64)
    mount_quat = np.asarray(mount.quat, dtype=np.float64)
    mount_quat /= np.linalg.norm(mount_quat)
    inv_pos = np.zeros(3, dtype=np.float64)
    inv_quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_negPose(inv_pos, inv_quat, mount_pos, mount_quat)
    calibrated_pos = np.zeros(3, dtype=np.float64)
    calibrated_quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mulPose(
        calibrated_pos,
        calibrated_quat,
        np.asarray(_CALIBRATED_MOUNT_POS, dtype=np.float64),
        np.asarray(_CALIBRATED_MOUNT_QUAT, dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.asarray(quat, dtype=np.float64),
    )
    out_pos = np.zeros(3, dtype=np.float64)
    out_quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mulPose(out_pos, out_quat, inv_pos, inv_quat, calibrated_pos, calibrated_quat)
    return tuple(float(value) for value in out_pos), tuple(float(value) for value in out_quat)


def _apply_collision_overlay(hand: mujoco.MjSpec) -> None:
    """Replace Hand 2 mesh collisions with the source capsule overlay."""
    for body_name, (fromto, radius) in _COLLISION_CAPSULES.items():
        collision_name = f"{body_name}_col"
        mesh = hand.geom(collision_name)
        mesh.name = f"{collision_name}_mesh_disabled"
        mesh.contype = 0
        mesh.conaffinity = 0
        body = hand.body(body_name)
        enabled = not body_name.endswith("link1")
        body.add_geom(
            name=collision_name,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=(*fromto[0], *fromto[1]),
            size=(radius, 0.0, 0.0),
            contype=int(enabled),
            conaffinity=int(enabled),
            group=3,
            density=0,
        )
    # The upstream excludes are for the mesh representation.  The task uses a
    # single explicit palm/thumb exclusion with the capsule overlay.
    while hand.excludes:
        hand.delete(hand.excludes[0])
    hand.add_exclude(bodyname1="right_finger1_link2", bodyname2="right_palm_link")
    mount = hand.geom("right_mount_collision")
    mount.contype = 0
    mount.conaffinity = 0


__all__ = ["materialize_scene"]

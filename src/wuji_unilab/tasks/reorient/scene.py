"""Cold-path native MJCF composition using repository-local Wuji assets.

Sensor selection is derived from wuji-technology/wuji-mjlab 9410a3a,
src/wuji_mjlab/tasks/reorient/reorient_terms.py (Apache-2.0, Wuji Technology).
The resulting named sensors are consumed through UniLab's public sensor view.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from wuji_unilab.assets import ASSET_ROOT, verify_assets

# MuJoCo's MjSpec and enums are exported dynamically by its native bindings.
mujoco: Any = import_module("mujoco")


@dataclass(frozen=True)
class SceneDescription:
    path: Path
    sensors: dict[str, tuple[str, ...]]


@lru_cache(maxsize=1)
def materialize_scene() -> SceneDescription:
    """Build once per process, never from a reset/step/randomization path."""
    hashes = verify_assets()
    task_xml = Path(__file__).with_name("initial_state.xml")
    key = hashlib.sha256(
        (str(ASSET_ROOT.resolve()) + json.dumps(hashes, sort_keys=True)).encode()
        + task_xml.read_bytes()
        + Path(__file__).read_bytes()
    ).hexdigest()[:24]
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "wuji_unilab" / key
    cache.mkdir(parents=True, exist_ok=True)
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.01
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.iterations = 10
    spec.option.ls_iterations = 20
    cube_root = ASSET_ROOT / "objects/inhand_object"
    cube = mujoco.MjSpec.from_file(str(cube_root / "xmls/cube.xml"))
    cube.meshdir = str(cube_root)
    cube.texturedir = str(cube_root)
    spec.attach(cube, prefix="object/", frame=spec.worldbody.add_frame())
    robot_root = ASSET_ROOT / "robots/wuji_hand"
    hand = mujoco.MjSpec.from_file(str(robot_root / "mjcf/right_mjlab.xml"))
    hand.meshdir = str(robot_root / "meshes/right")
    hand.option.iterations = spec.option.iterations
    hand.option.ls_iterations = spec.option.ls_iterations
    settings = ET.parse(task_xml).getroot()
    pose_node = settings.find("./custom/numeric")
    assert pose_node is not None
    pose = np.fromstring(pose_node.attrib["data"], sep=" ")
    palm = hand.body("right_palm_link")
    palm.mocap = True
    palm.pos = pose[:3]
    palm.quat = pose[3:]
    spec.attach(hand, prefix="robot/", frame=spec.worldbody.add_frame())
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[5, 5, 0.05],
        rgba=[0.15, 0.18, 0.22, 1],
    )
    spec.worldbody.add_light(pos=[0, 0, 2], dir=[0, 0, -1])
    groups: dict[str, tuple[str, ...]] = {}
    objtypes = {
        "geom": mujoco.mjtObj.mjOBJ_GEOM,
        "body": mujoco.mjtObj.mjOBJ_BODY,
        "subtree": mujoco.mjtObj.mjOBJ_XBODY,
    }

    def contact(group, mode, patterns, secondary_mode, secondary, fields, reduce):
        candidates = spec.geoms if mode == "geom" else spec.bodies
        primaries = [
            item.name
            for item in candidates
            if item.name.startswith("robot/")
            and any(re.fullmatch(p, item.name.removeprefix("robot/")) for p in patterns)
        ]
        if not primaries:
            raise ValueError(f"No contact selectors matched for {group}")
        for field in fields:
            names = []
            for index, primary in enumerate(primaries):
                name = f"{group}_{field}_{index}"
                kwargs = dict(
                    name=name,
                    type=mujoco.mjtSensor.mjSENS_CONTACT,
                    objtype=objtypes[mode],
                    objname=primary,
                    intprm=[1 if field == "found" else 2, reduce, 1],
                )
                if secondary is not None:
                    kwargs.update(reftype=objtypes[secondary_mode], refname=secondary)
                spec.add_sensor(**kwargs)
                names.append(name)
            groups[f"{group}_{field}"] = tuple(names)

    contact(
        "tips",
        "geom",
        [r"right_finger[1-5]_link4_col"],
        "geom",
        "object/cube",
        ("found", "force"),
        1,
    )
    contact("palm", "body", [r"right_palm_link"], "body", "object/cube", ("force",), 3)
    contact(
        "proximal",
        "body",
        [r"right_palm_link", r"right_finger.*_link[12]"],
        "body",
        "object/cube",
        ("found",),
        1,
    )
    contact("distal", "body", [r"right_finger.*_link[34]"], "body", "object/cube", ("found",), 1)
    undesired = [
        "right_palm_link",
        "right_finger1_link1",
        "right_finger5_link1",
        "right_finger5_link2",
    ]
    undesired += [f"right_finger{f}_link{j}" for f in (2, 3, 4) for j in (1, 2, 3)]
    contact("undesired", "body", undesired, "body", "object/cube", ("force",), 3)
    contact("tip_any", "body", [r"right_finger[1-5]_link4"], None, None, ("force",), 3)
    contact(
        "self",
        "geom",
        [r"right_finger.*_col"],
        "subtree",
        "robot/right_palm_link",
        ("found", "force"),
        0,
    )
    force_names = []
    for index, actuator in enumerate(spec.actuators):
        name = f"actuator_force_{index}"
        spec.add_sensor(
            name=name,
            type=mujoco.mjtSensor.mjSENS_ACTUATORFRC,
            objtype=mujoco.mjtObj.mjOBJ_ACTUATOR,
            objname=actuator.name,
        )
        force_names.append(name)
    groups["actuator_force"] = tuple(force_names)
    site_names = []
    for f in range(1, 6):
        name = f"tip_velocity_{f}"
        spec.add_sensor(
            name=name,
            type=mujoco.mjtSensor.mjSENS_FRAMELINVEL,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname=f"robot/right_finger{f}_tip",
        )
        site_names.append(name)
    groups["tip_velocity"] = tuple(site_names)
    key_node = settings.find("./keyframe/key")
    assert key_node is not None
    spec.add_key(
        name="home",
        qpos=np.fromstring(key_node.attrib["qpos"], sep=" "),
        ctrl=np.fromstring(key_node.attrib["ctrl"], sep=" "),
        mpos=pose[:3],
        mquat=pose[3:],
    )
    root = ET.fromstring(spec.to_xml())
    # MjSpec.attach/to_xml does not preserve child meshdir. Resolve asset files
    # in the cold materialization output; no external source path is used.
    for element in root.findall("./asset/*"):
        base = (
            robot_root / "meshes/right"
            if element.attrib.get("name", "").startswith("robot/")
            else cube_root
        )
        for attribute, value in list(element.attrib.items()):
            if attribute.startswith("file"):
                element.set(attribute, str(base / value))
    payload = ET.tostring(root, encoding="unicode")
    path = cache / "scene.xml"
    with tempfile.NamedTemporaryFile(mode="w", dir=cache, suffix=".xml", delete=False) as stream:
        stream.write(payload)
        temporary = Path(stream.name)
    try:
        model = mujoco.MjModel.from_xml_path(str(temporary))
        if (model.nq, model.nv, model.nu, model.nmocap) != (27, 26, 20, 1):
            raise ValueError("Unexpected Wuji model dimensions")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return SceneDescription(path, groups)

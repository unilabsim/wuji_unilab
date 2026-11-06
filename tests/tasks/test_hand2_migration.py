"""Dependency-light checks for the migrated Hand 2 task layout."""

import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).parents[2]
HAND_XML = ROOT / "third_party/wuji-description/src/wuji_description/resources/hand2/hand2_2/body/mjcf/right.xml"
CUBE_XML = ROOT / "src/unisupermimic/assets/objects/inhand_cube/xmls/cube.xml"
CFG = ROOT / "src/unisupermimic/conf/supermimic/task/dex_hand2_reorient/base.yaml"
PPO_CFG = ROOT / "src/unisupermimic/conf/supermimic/config_ppo.yaml"


def test_hand2_resources_are_repository_local():
    assert HAND_XML.is_file()
    assert CUBE_XML.is_file()
    assert (CUBE_XML.parent.parent / "meshes/dex_cube.obj").is_file()
    assert len(list(HAND_XML.parent.parent.glob("meshes/right/*.STL"))) >= 30


def test_upstream_hand2_has_the_expected_control_contract():
    root = ET.parse(HAND_XML).getroot()
    joints = [joint for joint in root.findall(".//joint") if joint.get("name")]
    actuators = root.findall("./actuator/*")
    assert len(joints) == 20
    assert len(actuators) == 20
    assert {sensor.get("name") for sensor in root.findall("./sensor/*")} == {
        "r_wrist_imu_acc",
        "r_wrist_imu_gyro",
    }


def test_hand2_config_exposes_unilab_task_contract():
    text = CFG.read_text(encoding="utf-8")
    for required in (
        "Hand2ReorientCommandCfg",
        "WujiActionCfg",
        "sim_dt: 0.01",
        "default_keyframe_name: home",
        "ctrl_dt: 0.05",
        "task_name: DexHand2Reorient",
        "params: {kind: orientation, injection_prob: 0.02}",
        "unisupermimic.tasks.dex_reorient.hand2.events.ResetWuji",
    ):
        assert required in text


def test_hand2_ppo_entrypoint_config_is_separate_from_g1_learner():
    text = PPO_CFG.read_text(encoding="utf-8")
    assert "task: dex_hand2_reorient/mujoco" in text
    assert "algo: ppo" in text
    assert "class_name: rsl_rl.algorithms:PPO" in text
    assert "tf32_matmul: ieee" in text
    assert "task_name: DexHand2Reorient" in text

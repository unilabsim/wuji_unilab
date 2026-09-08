import pickle
from pathlib import Path

import mujoco
import pytest
from unilab.base import registry
from unilab.base.env_factory import registry_env_factory

from wuji_unilab.assets import ASSET_ROOT, verify_assets
from wuji_unilab.config import compose_task, env_overrides
from wuji_unilab.tasks.reorient.scene import materialize_scene


def test_local_assets_compile_and_preserve_physical_model():
    hashes = verify_assets()
    assert len(hashes) == 50
    assert all((ASSET_ROOT / name).is_file() for name in hashes)
    scene = materialize_scene()
    model = mujoco.MjModel.from_xml_path(str(scene.path))
    assert (model.nq, model.nv, model.nu, model.nmocap) == (27, 26, 20, 1)
    assert model.opt.iterations == 10 and model.opt.ls_iterations == 20
    assert model.body("robot/right_palm_link").mocapid[0] == 0
    assert model.joint("object/cube_freejoint").qposadr[0] == 0
    assert "wuji-mjlab/src" not in scene.path.read_text()
    assert len(scene.sensors["tips_found"]) == 5
    assert len(scene.sensors["actuator_force"]) == 20


@pytest.mark.parametrize(
    "task,envs", [("WujiHand_Reorient", 8192), ("WujiHand_Reorient_Light", 4096)]
)
def test_owners_resolve_required_features_and_picklable_factory(task, envs):
    cfg = compose_task(task)
    overrides = env_overrides(cfg)
    assert cfg.algo.num_envs == envs
    assert cfg.training.sim_backend == "mjwarp"
    assert set(overrides["events"]) == {"reset", "randomize", "disturbance"}
    assert set(overrides["commands"]) == {"reorient_command"}
    assert set(overrides["terminations"]) == {"time_out", "cage_drop"}
    assert set(overrides["curriculum"]) == {"success_curriculum", "adaptive_episode"}
    assert set(overrides["rewards"]) == {
        "orientation_alignment",
        "hand_pose",
        "action_rate",
        "torque",
        "tip_slide",
        "cage_escape",
        "finger_collision",
        "hold_escalation",
        "palm_detach",
    }
    assert {
        "action_delta_rms",
        "action_jerk_rms",
        "cage_escape_frequency",
        "fingertip_contact_count",
        "torque_saturation_ratio",
        "joint_vel_rms",
        "cube_height_above_palm",
        "finger_collision_frequency",
        "cube_survival_steps",
        "success_interval",
        "goal_reach_count",
    } <= set(overrides["metrics"])
    assert Path(overrides["scene"]["model_file"]).is_file()
    factory = pickle.loads(pickle.dumps(registry_env_factory(task, "mjwarp")))
    assert factory.keywords == registry_env_factory(task, "mjwarp").keywords
    assert registry.list_registered_envs()[task]["available_backends"] == ["mjwarp"]
    with pytest.raises(ValueError, match="does not support"):
        registry.make(task, sim_backend="mujoco", env_cfg_override=overrides)

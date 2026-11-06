"""Compare native Pen rewards with the original task's measured MDP terms."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from unilab.managers import ManagerTermBase

from unisupermimic.tasks.pen_spin import metrics, rewards
from unisupermimic.tasks.pen_spin.observations import _quat_rotate


def _source_terms(filename):
    path = (
        Path(__file__).parents[2]
        / "references/Dex-Supermimic/src/dex_supermimic/tasks/pen_spin/mdp"
        / filename
    )
    if not path.is_file():
        pytest.skip("Dex-SuperMimic reference checkout is unavailable")
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    namespace = {
        "torch": torch, "np": np, "ManagerTermBase": ManagerTermBase,
        "SceneEntityCfg": lambda name: SimpleNamespace(name=name),
        "_ROBOT": SimpleNamespace(name="robot"), "_OBJECT": SimpleNamespace(name="object"),
    }
    # Postponed annotations keep source API type names simulator-independent.
    source = "from __future__ import annotations\n" + ast.unparse(tree)
    exec(compile(source, str(path), "exec"), namespace)
    return namespace


class _Scene(dict):
    pass


def _fixture():
    torch.manual_seed(193)
    batch = 3
    target_quat = torch.randn(batch, 4)
    target_quat /= target_quat.norm(dim=-1, keepdim=True)
    current_quat = torch.randn(batch, 4)
    current_quat /= current_quat.norm(dim=-1, keepdim=True)
    current_pos = torch.randn(batch, 3) * 0.015
    target_pos = torch.randn(batch, 3) * 0.015
    joints = torch.randn(batch, 20) * 0.1
    links = torch.randn(batch, 21, 3) * 0.02
    hard_limits = torch.stack((-torch.ones(20), torch.ones(20) * 2.0), dim=-1)
    nominal_limits = torch.linspace(0.1, 0.8, 20)
    live_limits = nominal_limits[None, :] * torch.tensor([[0.7], [0.9], [1.05]])
    torque = torch.randn(batch, 20) * nominal_limits[None, :]
    contact = torch.randint(0, 2, (batch, 20)).float()
    sensor_names = tuple(f"actuator_force_{index}" for index in range(20))
    collision_names = tuple(tuple(f"collision_{group}_{finger}" for finger in range(5)) for group in range(4))
    sensors = dict(zip(sensor_names, torque.unbind(-1), strict=True))
    sensors.update(zip(sum(collision_names, ()), contact.unbind(-1), strict=True))
    motion = SimpleNamespace(
        axis_points=torch.tensor(((0.1, 0.0, 0.0), (-0.1, 0.0, 0.0))),
        object_pose=torch.cat((target_pos, target_quat), dim=-1),
        joint_pos=torch.randn(batch, 20) * 0.1,
        link_pos=torch.randn(batch, 21, 3) * 0.02,
    )
    robot = SimpleNamespace(data=SimpleNamespace(
        joint_pos=joints, joint_pos_limits=hard_limits[None].expand(batch, -1, -1),
        actuator_force=torque, joint_vel=torch.randn(batch, 20), joint_acc=torch.randn(batch, 20),
    ))
    scene = _Scene(robot=robot)
    scene.read_plan = SimpleNamespace(
        joint_tensor_view=lambda name: robot.data,
        sensor_tensor_views=lambda name, names: SimpleNamespace(values={key: sensors[key] for key in names}),
    )
    env = SimpleNamespace(
        scene=scene, num_envs=batch, device="cpu", step_dt=0.02,
        command_manager=SimpleNamespace(get_term=lambda name: motion),
        action_manager=SimpleNamespace(
            action=torch.randn(batch, 20), prev_action=torch.randn(batch, 20),
            prev_prev_action=torch.randn(batch, 20),
        ),
        sim=SimpleNamespace(model=SimpleNamespace(
            actuator_forcerange=torch.stack((-live_limits, live_limits), dim=-1)
        )),
        _pen_model_fields={"actuator_forcerange": torch.stack((-live_limits, live_limits), dim=-1)},
        _pen_reward_metadata={
            "joint_limits": hard_limits.numpy(), "joint_dof_ids": tuple(range(6, 26)),
            "nominal_limits": nominal_limits.numpy(), "actuator_ids": np.arange(20),
            "torque_sensors": sensor_names, "collision_sensors": collision_names,
        },
    )
    return env, motion, current_pos, current_quat, links, nominal_limits, contact


def test_native_tracking_rewards_match_source_with_translation_and_hard_limits(monkeypatch):
    source = _source_terms("rewards.py")
    env, motion, position, quaternion, links, _, _ = _fixture()
    monkeypatch.setattr(rewards, "task", lambda env: motion)
    monkeypatch.setattr(rewards, "_runtime_pose", lambda env, motion: (position, quaternion))
    monkeypatch.setattr(rewards, "_runtime_link_pos", lambda env: links)
    axis = torch.tensor([0.1, 0.0, 0.0]).expand(env.num_envs, -1)
    current_axis = _quat_rotate(quaternion, axis)
    target_axis = _quat_rotate(motion.object_pose[:, 3:], axis)
    point_error = torch.stack((
        position + current_axis - motion.object_pose[:, :3] - target_axis,
        position - current_axis - motion.object_pose[:, :3] + target_axis,
    ), dim=1)
    source.update(
        object_pose_wrist_local=lambda *args: torch.cat((position, quaternion), dim=-1),
        object_axis_point_error=lambda *args: point_error.flatten(1),
        joint_position_error=lambda env, cfg, name: env.scene["robot"].data.joint_pos - motion.joint_pos,
        link_pos_wrist_local=lambda *args, **kwargs: links.flatten(1),
    )
    joint_cfg = SimpleNamespace(name="robot", joint_ids=slice(None))
    expected = {
        "object_pos": source["object_position_tracking_reward"](env),
        "object_rot": source["object_axis_point_tracking_reward"](env),
        "joint_pos": source["joint_position_tracking_reward"](env, joint_cfg),
        "link_pos": source["link_position_tracking_reward"](env, joint_cfg),
        "action_rate": source["action_rate_combined"](env),
    }
    for kind, value in expected.items():
        torch.testing.assert_close(rewards.reward(env, kind), value)
    # A pure translation must also reduce object_rot; matching orientation
    # alone cannot earn the source's full endpoint tracking reward.
    quaternion.copy_(motion.object_pose[:, 3:])
    assert (rewards.reward(env, "object_rot") < 1).all()


def test_native_torque_overcurrent_and_contacts_match_measured_source():
    source = _source_terms("rewards.py")
    env, _, _, _, _, nominal, contact = _fixture()
    cfg = SimpleNamespace(params={})
    torque = rewards.normalized_joint_torques_l2(cfg, env)
    expected_torque = source["normalized_joint_torques_l2"].__new__(source["normalized_joint_torques_l2"])
    expected_torque._asset, expected_torque._ids, expected_torque._limits = env.scene["robot"], slice(None), nominal
    torch.testing.assert_close(torque(env), expected_torque(env, None))
    overcurrent = rewards.sustained_overcurrent(cfg, env)
    expected_current = source["sustained_overcurrent"].__new__(source["sustained_overcurrent"])
    expected_current.__dict__.update(expected_torque.__dict__)
    expected_current._global_ids = torch.arange(20)
    expected_current._charge = torch.zeros_like(overcurrent.charge)
    for step in range(100):
        if step == 50:
            overcurrent.reset(torch.tensor([1]))
            expected_current.reset(torch.tensor([1]))
        torch.testing.assert_close(
            overcurrent(env, charge_s=1.0, leak_s=2.0, threshold=0.8),
            expected_current(env, None, charge_s=1.0, leak_s=2.0, threshold=0.8),
        )
        torch.testing.assert_close(overcurrent.charge, expected_current._charge)
    collision = rewards.finger_self_collision_penalty(cfg, env)
    expected_collision = source["finger_self_collision_penalty"].__new__(source["finger_self_collision_penalty"])
    expected_collision._sensor = SimpleNamespace(data=SimpleNamespace(found=contact))
    expected_collision._middle = list(range(10))
    expected_collision._distal, expected_collision._tips = list(range(10, 15)), list(range(15, 20))
    torch.testing.assert_close(collision(env), expected_collision(env))
    saturation = metrics.actuator_saturation_share(cfg, env)
    expected = (env.scene["robot"].data.actuator_force.abs() >= 0.98 * env._pen_model_fields["actuator_forcerange"][:, :, 1]).float().mean(-1)
    torch.testing.assert_close(saturation(env), expected)


def test_live_limits_do_not_silently_ignore_an_incomplete_dr_table():
    env, *_ = _fixture()
    env._pen_model_fields = {}
    with pytest.raises(KeyError, match="actuator_forcerange"):
        metrics.actuator_saturation_share(SimpleNamespace(params={}), env)(env)


def test_native_smoothness_and_tracking_metrics_match_source(monkeypatch):
    source = _source_terms("metrics.py")
    reward_source = _source_terms("rewards.py")
    env, motion, position, quaternion, _, _, _ = _fixture()
    monkeypatch.setattr(metrics, "task", lambda env: motion)
    monkeypatch.setattr(metrics, "_runtime_pose", lambda env, motion: (position, quaternion))
    axis = torch.tensor([1.0, 0.0, 0.0]).expand(env.num_envs, -1)
    angle = torch.acos((
        _quat_rotate(quaternion, axis) * _quat_rotate(motion.object_pose[:, 3:], axis)
    ).sum(-1).clamp(-1, 1))
    source.update(
        object_position_error=lambda *args, **kwargs: position - motion.object_pose[:, :3],
        object_shaft_direction_error=lambda *args, **kwargs: angle[:, None],
    )
    cfg = SimpleNamespace(name="robot", joint_ids=slice(None))
    torch.testing.assert_close(metrics.object_pos_tracking_error_cm(env), source["object_pos_tracking_error_cm"](env))
    torch.testing.assert_close(metrics.object_ori_tracking_error_deg(env), source["object_ori_tracking_error_deg"](env))
    torch.testing.assert_close(metrics.joint_vel_rms(env), source["joint_vel_rms"](env, cfg))
    torch.testing.assert_close(
        metrics.action_rate_l2(env) + metrics.action_acc_l2(env),
        reward_source["action_rate_combined"](env),
    )
    from unisupermimic.tasks.pen_spin import reset_bridge

    monkeypatch.setattr(reset_bridge, "read_joint_acceleration", lambda actual_env, ids: actual_env.scene["robot"].data.joint_acc)
    acceleration = metrics.joint_acc_rms(SimpleNamespace(params={}), env)
    torch.testing.assert_close(acceleration(env), source["joint_acc_rms"](env, cfg))


def test_native_measured_terms_bind_real_mujoco_sensors_and_reset_charge():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from unilab.base import registry
    from unilab.base.config_adapter import EnvConfigAdapter
    from unilab.training import ensure_registries

    root = Path(__file__).parents[2]
    ensure_registries()
    with initialize_config_dir(
        config_dir=str(root / "src/unisupermimic/conf/supermimic"), version_base="1.3"
    ):
        cfg = compose("config_pen_spin", overrides=["algo.num_envs=2"])
    OmegaConf.set_struct(cfg, False)
    # This check isolates measured reward/metric bindings from the independent
    # production startup DR migration.
    cfg.env.events = OmegaConf.create({
        "reset_to_reference_state": {
            "func": "unisupermimic.tasks.pen_spin.events.ResetPenReference", "mode": "reset"
        }
    })
    cfg.env.commands.motion.synthetic_reference = True
    cfg.env.auto_reset = False
    for name, class_name, weight, params in (
        ("torque", "normalized_joint_torques_l2", -1.0, {}),
        ("overcurrent", "sustained_overcurrent", -4.0, {"charge_s": 1.0, "leak_s": 2.0, "threshold": 0.8}),
        ("finger_collision", "finger_self_collision_penalty", -1.0, {}),
    ):
        cfg.reward[name] = {"func": f"unisupermimic.tasks.pen_spin.rewards.{class_name}", "weight": weight, "params": params}
    cfg.env.metrics = OmegaConf.create({
        name: {"func": f"unisupermimic.tasks.pen_spin.metrics.{function}"}
        for name, function in (
            ("object_pos_tracking_error_cm", "object_pos_tracking_error_cm"),
            ("object_ori_tracking_error_deg", "object_ori_tracking_error_deg"),
            ("action_rate_sq", "action_rate_l2"),
            ("action_acc_sq", "action_acc_l2"),
            ("joint_vel_rms_rad_s", "joint_vel_rms"),
            ("joint_acc_rms_rad_s2", "joint_acc_rms"),
            ("actuator_saturation_share", "actuator_saturation_share"),
        )
    })
    override = EnvConfigAdapter(cfg, root_dir=root, algo_name="ppo").build_task_env_cfg_override()
    env = registry.make("DexHand2PenSpin", sim_backend="mujoco", env_cfg_override=override, num_envs=2)
    try:
        env.reset()
        result = env.step(torch.full((2, 20), 0.1))
        assert torch.isfinite(result.reward).all()
        assert torch.isfinite(env.metrics_manager._step_values).all()
        overcurrent = env.reward_manager.get_term_cfg("overcurrent").func
        assert (overcurrent.charge > 0).any()
        survivor = overcurrent.charge[1].clone()
        env.reset(env_ids=torch.tensor([0]))
        torch.testing.assert_close(overcurrent.charge[0], torch.zeros(20))
        torch.testing.assert_close(overcurrent.charge[1], survivor, rtol=0, atol=0)
    finally:
        env.close()

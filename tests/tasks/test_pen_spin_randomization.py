"""Effect checks for production Pen Spin physics and reference reset noise."""

from __future__ import annotations

import math
import re
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from unisupermimic.rl.algos.dex_supermimic.env_bridge import SourceEnvBridge
from unisupermimic.rl.algos.dex_supermimic.sim import PenSimReader
from unisupermimic.tasks.pen_spin.randomization import (
    MODEL_FIELDS,
    default_model_fields,
    randomize_pen_dimensions,
    sample_pseudo_inertia,
)
from unisupermimic.tasks.pen_spin.scene import materialize_scene

ROOT = Path(__file__).parents[2]
CLIPS = ROOT / "src/unisupermimic/tasks/pen_spin/data/rawdata/clips"
RECIPE = ROOT / "src/unisupermimic/tasks/pen_spin/data/recipe/01_single_axis.json"


@pytest.fixture
def production_env():
    """Use the source training profile, real clips and actual CPU model pool."""
    from hydra import compose, initialize_config_dir
    from unilab.base import registry
    from unilab.base.config_adapter import EnvConfigAdapter
    from unilab.training import ensure_registries

    if not (CLIPS / "01_single_axis/index.json").is_file():
        pytest.skip("upstream pen clip catalog is not installed")
    ensure_registries()
    with initialize_config_dir(
        config_dir=str(ROOT / "src/unisupermimic/conf/supermimic"), version_base="1.3"
    ):
        cfg = compose(
            "config_pen_spin_supermimic",
            overrides=[
                "task=dex_hand2_pen_spin/mujoco", "algo.num_envs=4",
                "training.sim_backend=mujoco", "training.collector_tensor_device=cpu",
                f"env.commands.motion.recipe_path={RECIPE}",
                f"env.commands.motion.clips_root={CLIPS}",
                # Exercise moving reference velocities, beyond the separate
                # first-frame hold behavior covered by sampling tests.
                "env.commands.motion.first_frame_hold_prob=0.0",
            ],
        )
    override = EnvConfigAdapter(cfg, root_dir=ROOT, algo_name="ppo").build_task_env_cfg_override()
    env = registry.make("DexHand2PenSpin", sim_backend="mujoco", env_cfg_override=override, num_envs=4)
    try:
        yield env
    finally:
        env.close()


def _pool_field(env, name):
    expected = env._pen_model_fields[name]
    values = np.array(env.backend._pool.expand(name), copy=True).reshape(tuple(expected.shape))
    return torch.as_tensor(values, dtype=expected.dtype)


def test_resume_counter_keeps_curriculum_on_the_restored_timeline(production_env):
    from unisupermimic.tasks.pen_spin.curriculums import linear_step_curriculum

    env = SourceEnvBridge(production_env)
    env.reset()
    env.common_step_counter = 240000
    assert production_env.step_counter == 240000
    env.step(torch.zeros(env.num_envs, 20))
    assert env.common_step_counter == 240001
    assert linear_step_curriculum(production_env, None, 240000, 40000)["value"] == 1.0


def test_instability_checks_all_source_physics_channels_without_filtering(production_env):
    from unisupermimic.tasks.pen_spin.reset_bridge import physics_nan_mask

    env = production_env
    env.reset()
    backend = env.backend
    assert not physics_nan_mask(env).any()
    buffers = (
        backend._qpos_view, backend._qvel_view,
        backend._pool.bind("qacc", backend._np_dtype), backend._warm_view,
        backend._sensor_data,
    )
    for value in buffers:
        old = float(value[0, 0])
        try:
            value[0, 0] = float("nan")
            assert physics_nan_mask(env).tolist() == [True, False, False, False]
            assert np.isnan(value[0, 0])
        finally:
            value[0, 0] = old


def test_startup_draws_reach_actual_per_world_physics_and_survive_partial_reset(production_env):
    env = production_env
    model = env._pen_physics_model
    fields = env._pen_model_fields
    # Every published table must reach the executor, including the fields
    # unavailable in Entity: marker positions and actuator effort limits.
    for name in MODEL_FIELDS:
        torch.testing.assert_close(_pool_field(env, name), fields[name], rtol=0, atol=0)
    actuator = model.actuator("robot/right_finger2_joint2_actuator").id
    palm = model.body("robot/right_palm_link").id
    contact = model.geom("robot/right_finger2_link2_col").id
    shaft = model.geom("object/cube").id
    top = model.geom("object/A_top").id
    for name, selection in (
        ("actuator_gainprm", (actuator, 0)),
        ("actuator_forcerange", (actuator, 1)),
        ("body_inertia", (palm, 0)),
        ("geom_friction", (contact, 0)),
        ("geom_size", (shaft, 0)),
        ("geom_pos", (top, 0)),
    ):
        value = _pool_field(env, name)[(slice(None), *selection)]
        assert value.unique().numel() == env.num_envs, name
    before = {name: _pool_field(env, name) for name in MODEL_FIELDS}
    bias_before = env.scene["robot"].data.encoder_bias_tensor.clone()
    env.reset()
    env.step(torch.zeros(env.num_envs, 20))
    survivor_q = env.backend.get_state_views(("qpos",), device="cpu")["qpos"][1:].clone()
    env.reset(env_ids=torch.tensor([0]))
    for name, value in before.items():
        torch.testing.assert_close(_pool_field(env, name), value, rtol=0, atol=0)
        torch.testing.assert_close(fields[name], value, rtol=0, atol=0)
    torch.testing.assert_close(env.scene["robot"].data.encoder_bias_tensor, bias_before, rtol=0, atol=0)
    torch.testing.assert_close(
        env.backend.get_state_views(("qpos",), device="cpu")["qpos"][1:], survivor_q, rtol=0, atol=0
    )


def test_reader_context_uses_executor_gains_and_six_actual_dr_ratios(production_env):
    native = production_env
    env = SourceEnvBridge(native)
    env.reset()
    reader = PenSimReader(env)
    model, defaults = native._pen_physics_model, native._pen_default_model_fields
    current = {name: _pool_field(native, name) for name in MODEL_FIELDS}
    actuator_ids = [model.actuator(f"{name}_actuator").id for name in reader.joint_names]
    dofs = [model.jnt_dofadr[model.joint(name).id] for name in reader.joint_names]
    contact = [index for index in range(model.ngeom) if re.fullmatch(
        r"robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)", model.geom(index).name
    )]
    obj = model.body("object/cube").id
    geoms = [model.geom(name).id for name in ("object/cube", "object/A_bottom", "object/A_top")]
    # These expectations use the independently read, expanded executor tables,
    # rather than the producer's publication cache.
    kp = current["actuator_gainprm"][:, actuator_ids, 0] / defaults["actuator_gainprm"][:, actuator_ids, 0]
    kd = current["actuator_biasprm"][:, actuator_ids, 2] / defaults["actuator_biasprm"][:, actuator_ids, 2]
    ratios = torch.stack((
        (current["geom_friction"][:, contact, 0] / defaults["geom_friction"][:, contact, 0]).mean(-1),
        current["body_mass"][:, obj] / defaults["body_mass"][:, obj],
        kp.mean(-1), kd.mean(-1),
        (current["dof_damping"][:, dofs] / defaults["dof_damping"][:, dofs].clamp_min(1e-8)).mean(-1),
        (current["geom_size"][:, geoms] / defaults["geom_size"][:, geoms].clamp_min(1e-8)).flatten(1).mean(-1),
    ), dim=-1)
    actual = reader.context()
    torch.testing.assert_close(actual, torch.cat((ratios, kp, kd), dim=-1))
    assert (actual[:, :4].std(0) > 0).all()
    torch.testing.assert_close(reader.torque_limits(), current["actuator_forcerange"][:, actuator_ids, 1])
    torch.testing.assert_close(reader.memory()["encoder_bias"], native.scene["robot"].data.encoder_bias_tensor)


def _rotation(quaternions):
    matrices = []
    for quaternion in np.asarray(quaternions):
        value = np.empty(9)
        mujoco.mju_quat2Mat(value, np.array(quaternion, dtype=np.float64))
        matrices.append(value.reshape(3, 3))
    return np.stack(matrices)


def test_curriculum_reset_slide_and_roll_preserve_shaft_line_and_reference_velocities(production_env):
    env = production_env
    env.common_step_counter = 240_000
    env.reset()
    motion = env.command_manager.get_term("motion")
    assert env.curriculum_manager._curriculum_state["disturbance_success_rate"]["value"] == 1.0
    model = env._pen_physics_model
    joint = model.joint("object/cube_freejoint").id
    qadr, vadr = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
    state = env.backend.get_state_views(("qpos", "qvel"), device="cpu")
    root_q, root_v = state["qpos"][:, qadr:qadr + 7].numpy(), state["qvel"][:, vadr:vadr + 6].numpy()
    # Physics uses sampled frame k; source reset then advances the command
    # to k+1 before publishing the actor's reference observation.
    frame = motion.trajectory_starts + (motion.time_steps - 1).clamp_min(0).minimum(motion.trajectory_lengths - 1)
    wrist = motion._frames["wrist_pose_w"][frame].numpy()
    local = motion._frames["object_pose"][frame].numpy()
    linear = motion._frames["object_linvel_w"][frame].numpy()
    angular = motion._frames["object_angvel_w"][frame].numpy()
    wrist_rot, local_rot, actual_rot = _rotation(wrist[:, 3:]), _rotation(local[:, 3:]), _rotation(root_q[:, 3:])
    reference_rot = wrist_rot @ local_rot
    reference_pos = wrist[:, :3] + np.einsum("nij,nj->ni", wrist_rot, local[:, :3])
    shaft_axis = reference_rot[:, :, 0]
    displacement = root_q[:, :3] - reference_pos
    slide = np.einsum("ni,ni->n", displacement, shaft_axis)
    np.testing.assert_allclose(displacement - slide[:, None] * shaft_axis, 0.0, atol=2e-7)
    np.testing.assert_allclose(actual_rot[:, :, 0], shaft_axis, atol=2e-6)
    assert np.all(np.abs(slide) <= 0.020001)
    assert np.any(np.abs(slide) > 1e-5)
    relative_rot = reference_rot.transpose(0, 2, 1) @ actual_rot
    roll = np.arctan2(relative_rot[:, 2, 1], relative_rot[:, 1, 1])
    assert np.all(np.abs(roll) <= math.radians(20.0) + 1e-5)
    assert np.any(np.abs(roll) > 1e-4)
    # MuJoCo's free-joint angular qvel is object-local, whereas source reset
    # velocities are world-frame. A noisy roll must not rotate that velocity.
    np.testing.assert_allclose(root_v[:, :3], linear, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(
        np.einsum("nij,nj->ni", actual_rot, root_v[:, 3:]), angular, rtol=1e-5, atol=1e-6
    )
    assert np.linalg.norm(linear) + np.linalg.norm(angular) > 0.01
    robot_dofs = [model.jnt_dofadr[model.joint(name).id] for name in env.scene["robot"].joint_names]
    torch.testing.assert_close(state["qvel"][:, robot_dofs].float(), motion._frames["joint_vel"][frame])


def _inertia_in_body(inertia, quaternion):
    rotation = _rotation(np.asarray(quaternion).reshape(-1, 4))
    values = np.asarray(inertia).reshape(-1, 3)
    return rotation @ (np.eye(3)[None] * values[:, None, :]) @ rotation.transpose(0, 2, 1)


def test_pseudo_inertia_preserves_physical_consistency_and_known_density_scaling():
    model = mujoco.MjModel.from_xml_path(str(materialize_scene()))
    defaults = default_model_fields(model, 8, "cpu")
    bodies = [model.body("robot/right_palm_link").id, model.body("robot/right_finger3_link2").id]
    mass, com, inertia, quat = sample_pseudo_inertia(defaults, bodies, (-0.458, 0.203), (-0.2, 0.2))
    assert torch.isfinite(torch.cat((mass[..., None], com, inertia, quat), dim=-1)).all()
    assert (mass > 0).all() and (inertia > 0).all()
    torch.testing.assert_close(quat.norm(dim=-1), torch.ones_like(mass))
    assert (inertia.sum(-1, keepdim=True) - 2.0 * inertia > 0).all()
    # For density scaling alone, the physical inertia tensor and mass both
    # double; principal-axis eigenvalue ordering is intentionally irrelevant.
    doubled = sample_pseudo_inertia(defaults, bodies, (math.log(2) / 2,) * 2)
    torch.testing.assert_close(doubled[0], defaults["body_mass"][:, bodies] * 2)
    torch.testing.assert_close(doubled[1], defaults["body_ipos"][:, bodies], atol=1e-7, rtol=1e-5)
    expected = _inertia_in_body(defaults["body_inertia"][:, bodies], defaults["body_iquat"][:, bodies]) * 2
    np.testing.assert_allclose(_inertia_in_body(doubled[2], doubled[3]), expected, rtol=2e-5, atol=1e-9)


@pytest.mark.parametrize("diameter,length,mass", [(0.014, 0.205, 0.012), (0.016, 0.265, 0.048)])
def test_pen_inertia_matches_independent_mujoco_uniform_volume_model(diameter, length, mass):
    model = mujoco.MjModel.from_xml_path(str(materialize_scene()))
    fields = default_model_fields(model, 1, "cpu")
    body_id = model.body("object/cube").id
    fields["body_mass"][:, body_id] = mass
    retained_com = fields["body_ipos"][:, body_id].clone() + torch.tensor([0.01, 0.0002, -0.0001])
    fields["body_ipos"][:, body_id] = retained_com
    randomize_pen_dimensions(fields, model, (diameter, diameter), (length, length))
    # Compile a separate homogeneous-density solid from its three primitive
    # geometries. MuJoCo integrates its volume/inertia independently of the
    # migrated analytical formula, including the box parallel-axis terms.
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="physical_pen")
    body.add_freejoint()
    shaft_length, end_center = length - 0.072, length / 2.0 - 0.018
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, fromto=(-shaft_length / 2, 0, 0, shaft_length / 2, 0, 0), size=(diameter / 2, 0, 0), density=1)
    for sign in (-1, 1):
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=(sign * end_center, 0, 0), size=(0.009, 0.009, 0.018), quat=(math.sqrt(0.5), 0, math.sqrt(0.5), 0), density=1)
    independent = spec.compile()
    physical_body = independent.body("physical_pen").id
    expected = _inertia_in_body(independent.body_inertia[[physical_body]], independent.body_iquat[[physical_body]])
    expected *= mass / independent.body_mass[physical_body]
    actual = _inertia_in_body(fields["body_inertia"][:, body_id], fields["body_iquat"][:, body_id])
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-10)
    torch.testing.assert_close(fields["body_mass"][:, body_id], torch.tensor([mass]))
    torch.testing.assert_close(fields["body_ipos"][:, body_id], retained_com)
    shaft, bottom, top = [model.geom(name).id for name in ("object/cube", "object/A_bottom", "object/A_top")]
    np.testing.assert_allclose(fields["geom_size"][0, shaft, :2], (diameter / 2, shaft_length / 2))
    np.testing.assert_allclose(fields["geom_pos"][0, [bottom, top], 0], (-end_center, end_center))
    np.testing.assert_allclose(fields["geom_size"][0, [bottom, top]], np.array([[.009, .009, .018]] * 2))

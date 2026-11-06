"""Fixed-state checks against Dex-Supermimic's world-axis Hand2 reset.

The oracle uses MuJoCo/NumPy math independently of the Torch reset path.
Source: Dex-Supermimic 2becb738, event_impl/reset.py and hand2_constants.py.
"""

from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from unilab.base import registry
from unilab.base.config_adapter import EnvConfigAdapter
from unilab.training import ensure_registries

from unisupermimic.tasks.dex_reorient.hand2 import events


def _source_pose(angle):
    position = np.array((-0.0284999, 0.00300004, 0.500250158))
    orientation = np.array((0.5, 0.5, 0.5, 0.5))
    pivot = np.array((-0.0030000041, -0.000250110803, 0.0285))
    rotated_pivot = np.empty(3)
    mujoco.mju_rotVecQuat(rotated_pivot, pivot, orientation)
    center = position + rotated_pivot
    cosine, sine = np.cos(angle), np.sin(angle)
    world_y = np.array(((cosine, 0, sine), (0, 1, 0), (-sine, 0, cosine)))
    delta = np.array((np.cos(angle / 2), 0, np.sin(angle / 2), 0))
    result_quat = np.empty(4)
    mujoco.mju_mulQuat(result_quat, delta, orientation)
    return center + world_y @ (position - center), result_quat


def _contacts(pairs, distances):
    result = {}
    for pair, distance in zip(pairs, distances, strict=True):
        if distance <= 0:
            key = tuple(sorted(map(int, pair)))
            result[key] = min(result.get(key, 0.0), float(distance))
    return result


@pytest.mark.gpu
def test_mjwarp_reset_matches_source_pose_contacts_and_partial_rows(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA MJWarp device")
    ensure_registries()
    root = Path(__file__).parents[2]
    with initialize_config_dir(
        config_dir=str(root / "src/unisupermimic/conf/supermimic"), version_base="1.3"
    ):
        cfg = compose(
            "config_hand2",
            overrides=["task=dex_hand2_reorient/mjwarp", "training.sim_backend=mjwarp"],
        )
    # Disable DR only for this controlled comparison; production config stays intact.
    cfg.env.seed = 42
    cfg.env.events.randomize = None
    cfg.env.events.disturbance = None
    cfg.env.events.reset.params.joint_noise = [-0.2, -0.2]
    cfg.env.events.reset.params.cube_noise = 0.0
    cfg.env.observations.policy.enable_corruption = False
    monkeypatch.setattr(
        events, "random_quaternions",
        lambda rng, count, device=None: torch.tensor(
            (1.0, 0.0, 0.0, 0.0), device=device
        ).repeat(count, 1),
    )
    override = EnvConfigAdapter(
        cfg, root_dir=root, algo_name="supermimic_hand2"
    ).build_task_env_cfg_override()
    env = registry.make(
        "DexHand2Reorient", sim_backend="mjwarp", env_cfg_override=override, num_envs=2
    )
    try:
        env.reset()
        backend = env.backend
        model = env.get_playback_model(0)
        if isinstance(model, (str, Path)):
            model = mujoco.MjModel.from_xml_path(str(model))
        palm_id = model.body("robot/right_palm_link").id
        term = env.event_manager.get_term_cfg("reset")
        for angle in (-0.4, -0.15, 0.0, 0.1):
            term.params["wrist_pitch"] = (angle, angle)
            env.reset()
            expected_pos, expected_quat = _source_pose(angle)
            data = backend._device_data
            device_pose = np.concatenate(
                (data.mocap_pos.numpy()[:, 0], data.mocap_quat.numpy()[:, 0]), axis=-1
            )
            for row in range(2):
                np.testing.assert_allclose(device_pose[row, :3], expected_pos, atol=1e-6)
                np.testing.assert_allclose(device_pose[row, 3:], expected_quat, atol=1e-6)
            np.testing.assert_allclose(term.func.pose.read(), device_pose, atol=1e-6)
            snapshot = env.get_physics_state_snapshot()
            np.testing.assert_allclose(snapshot[:, -7:], device_pose, atol=1e-6)
            pos, quat = env.command_manager.get_term("reorient").palm_pose()
            np.testing.assert_allclose(pos.cpu(), data.xpos.numpy()[:, palm_id], atol=1e-6)
            np.testing.assert_allclose(quat.cpu(), data.xquat.numpy()[:, palm_id], atol=1e-6)

            # Same joint/cube state, original source root formula, native MuJoCo forward.
            native = mujoco.MjData(model)
            native.qpos[:] = data.qpos.numpy()[0]
            native.qvel[:] = data.qvel.numpy()[0]
            np.testing.assert_allclose(native.qpos[:3], (-0.108, 0.0072, 0.5616), atol=1e-6)
            native.mocap_pos[0] = expected_pos
            native.mocap_quat[0] = expected_quat
            mujoco.mj_forward(model, native)
            np.testing.assert_allclose(data.geom_xpos.numpy()[0], native.geom_xpos, atol=1e-6)
            count = int(data.nacon.numpy()[0])
            rows = data.contact.worldid.numpy()[:count] == 0
            actual_contacts = _contacts(
                data.contact.geom.numpy()[:count][rows], data.contact.dist.numpy()[:count][rows]
            )
            expected_contacts = _contacts(
                [contact.geom for contact in native.contact],
                [contact.dist for contact in native.contact],
            )
            assert actual_contacts.keys() == expected_contacts.keys()
            for pair in expected_contacts:
                assert actual_contacts[pair] == pytest.approx(expected_contacts[pair], abs=1e-6)

        env.step(torch.zeros((2, 20), device=env.device))
        untouched = device_pose[0].copy()
        untouched_qpos = data.qpos.numpy()[0].copy()
        untouched_qvel = data.qvel.numpy()[0].copy()
        untouched_steps = env.episode_length_buf[0].clone()
        term.params["wrist_pitch"] = (-0.3, -0.3)
        env.reset(env_ids=torch.tensor([1], device=env.device))
        data = backend._device_data
        np.testing.assert_array_equal(data.mocap_pos.numpy()[0, 0], untouched[:3])
        np.testing.assert_array_equal(data.mocap_quat.numpy()[0, 0], untouched[3:])
        np.testing.assert_array_equal(data.qpos.numpy()[0], untouched_qpos)
        np.testing.assert_array_equal(data.qvel.numpy()[0], untouched_qvel)
        torch.testing.assert_close(env.episode_length_buf[0], untouched_steps)
        expected_pos, expected_quat = _source_pose(-0.3)
        np.testing.assert_allclose(data.mocap_pos.numpy()[1, 0], expected_pos, atol=1e-6)
        np.testing.assert_allclose(data.mocap_quat.numpy()[1, 0], expected_quat, atol=1e-6)
        before_empty = data.mocap_pos.numpy().copy()
        env.reset(env_ids=torch.empty(0, dtype=torch.long, device=env.device))
        np.testing.assert_array_equal(data.mocap_pos.numpy(), before_empty)
    finally:
        env.close()


"""Exercise the source learner boundary on a real catalog and MuJoCo state."""

from pathlib import Path

import pytest
import torch

from unisupermimic.rl.algos.dex_supermimic.env_bridge import SourceEnvBridge
from unisupermimic.rl.algos.dex_supermimic.pen import PenObservations, flatten_pen_memory
from unisupermimic.rl.algos.dex_supermimic.rotations import mm, mv, quat_to_mat
from unisupermimic.rl.algos.dex_supermimic.sim import PenSimReader
from unisupermimic.rl.algos.dex_supermimic.state import unpack

ROOT = Path(__file__).parents[3]
RECIPE = ROOT / "src/unisupermimic/tasks/pen_spin/data/recipe/01_single_axis.json"
CLIPS = ROOT / "src/unisupermimic/tasks/pen_spin/data/rawdata/clips"


def test_real_pen_reader_sensor_capture_and_partial_reset():
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
            "config_pen_spin",
            overrides=[
                "algo.num_envs=2",
                "env.commands.motion.synthetic_reference=false",
                f"env.commands.motion.recipe_path={RECIPE}",
                "env.observations.policy.enable_corruption=true",
            ],
        )
        cfg.env.observations.policy.terms.object_pose.func = (
            "unisupermimic.tasks.pen_spin.observations.PenObjectPoseCameraModel"
        )
    override = EnvConfigAdapter(cfg, root_dir=ROOT, algo_name="ppo").build_task_env_cfg_override()
    native_env = registry.make(
        "DexHand2PenSpin", sim_backend="mujoco", env_cfg_override=override, num_envs=2
    )
    reader = None
    try:
        env = SourceEnvBridge(native_env)
        env.reset()
        reader = PenSimReader(env)
        assert reader.camera is not None
        reader.attach_capture()
        observation = PenObservations(reader)
        # The source pen tracking reward uses hard limits, not the source
        # articulation's separate 0.9 soft-limit table used by cube control.
        joints = [reader.model.joint(name).id for name in reader.joint_names]
        limits = torch.tensor(reader.model.jnt_range[joints].copy(), dtype=torch.float32)
        torch.testing.assert_close(reader.mdp.lower, limits[:, 0])
        torch.testing.assert_close(reader.mdp.upper, limits[:, 1])
        for _ in range(3):
            policy, reward, terminated, truncated, _ = env.step(torch.zeros((2, 20)))
            assert policy["policy"].shape == (2, 155)
            torch.testing.assert_close(policy["policy"][:, 129:135], reader.memory()["camera_obs"])
            assert reader.state().shape == (2, 58)
            assert reader.contact().shape == (2, 15)
            assert reader.torque().shape == (2, 20)
            assert torch.isfinite(reward).all()
            assert terminated.dtype == truncated.dtype == torch.bool
            captured = reader.reward_snapshot
            torch.testing.assert_close(reward, sum(captured["terms"].values()))

            # Validate against the backend's final generalized state. The
            # packed body-pose sensors can precede the final integration on
            # CPU; using them here would change WM labels and actor inputs.
            values = native_env.backend.get_state_views(("qpos", "qvel"), device="cpu")
            backend_q, backend_v = values["qpos"], values["qvel"]
            q, qd, pos, rot, vel, omega = unpack(reader.state())
            torch.testing.assert_close(q, backend_q[:, reader._joint_q], rtol=0, atol=0)
            torch.testing.assert_close(qd, backend_v[:, reader._joint_v], rtol=0, atol=0)
            object_q = backend_q[:, reader._object_q:reader._object_q + 7]
            object_v = backend_v[:, reader._object_v:reader._object_v + 6]
            torch.testing.assert_close(mv(reader.palm_rot, pos) + reader.palm_pos, object_q[:, :3])
            object_rot = quat_to_mat(object_q[:, 3:])
            torch.testing.assert_close(mm(reader.palm_rot, rot), object_rot)
            torch.testing.assert_close(mv(reader.palm_rot, vel), object_v[:, :3])
            # MuJoCo free-joint angular qvel is object-local. Convert it to
            # world angular velocity before transforming into the palm frame.
            torch.testing.assert_close(mv(reader.palm_rot, omega), mv(object_rot, object_v[:, 3:]))
        assert flatten_pen_memory(reader.memory()).shape == (2, 230)
        auxiliary = observation.auxiliary()
        context = observation.context(reader.state(), reader.memory(), auxiliary, reader.xi)
        assert context.shape == (2, 368)
        assert observation.critic(context).shape == (2, 301)
        assert observation.disturbance(auxiliary).shape == (2, 174)

        # Resetting one episode must preserve the other row's physical state,
        # trajectory frame and every controller/camera-memory field exactly.
        memory_before = reader.memory()
        state_before = reader.state().clone()
        frames_before = reader.motion.time_steps.clone()
        env.reset(env_ids=torch.tensor([0]))
        memory_after = reader.memory()
        torch.testing.assert_close(reader.state()[1], state_before[1], rtol=0, atol=0)
        assert reader.motion.time_steps[1] == frames_before[1]
        for name, value in memory_before.items():
            torch.testing.assert_close(memory_after[name][1], value[1], rtol=0, atol=0)
    finally:
        if reader is not None:
            reader.detach_capture()
        native_env.close()

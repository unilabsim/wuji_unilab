import numpy as np
import pytest
import torch
import warp as wp
from unilab.base import registry

from wuji_unilab.config import compose_task, env_overrides
from wuji_unilab.rl.runtime import WujiWrapper
from wuji_unilab.tasks.reorient.commands import task

pytestmark = pytest.mark.gpu


@pytest.fixture
def env():
    wp.init()
    wp.set_device("cuda:0")
    cfg = compose_task(overrides=["algo.num_envs=4"])
    value = registry.make(
        cfg.training.task_name,
        sim_backend="mjwarp",
        env_cfg_override=env_overrides(cfg),
        num_envs=4,
    )
    value.reset()
    try:
        yield value
    finally:
        value.close()


def test_full_randomization_step_and_partial_reset_isolation(env):
    state = task(env)
    assert set(state.metrics) == {
        "goal_reach_count",
        "ori_error",
        "hold_counter",
        "goal_timer",
        "in_success_window",
        "window_timer",
    }
    assert env.obs_groups_spec == {"obs": 207, "critic": 413}
    assert bool(torch.any(torch.abs(state.dr_ratios - 1) > 0.01))
    for _ in range(12):
        result = env.step(torch.zeros((4, 20), dtype=torch.float32, device=env.device))
        assert all(torch.isfinite(value).all() for value in result.obs.values())
        assert torch.isfinite(result.reward).all()
        # Fixed DR writes must retain MuJoCo position-actuator semantics. A
        # positive-damping sign regression previously ejected every cube here.
        assert not result.terminated.any()
    assert (state.goal_timer > 0).all()
    before_joint = env.scene.read_plan.joint_tensor_view("robot").joint_pos.clone()
    palm_pos, palm_quat = state.palm_pose()
    tag_pos, tag_quat = state.tag_pose()
    assert not torch.allclose(tag_pos, torch.zeros_like(tag_pos))
    assert not torch.allclose(tag_quat, torch.zeros_like(tag_quat))
    torch.testing.assert_close(tag_pos, state._tag_pos)
    torch.testing.assert_close(tag_quat, state._tag_quat)
    torch.testing.assert_close(palm_pos, state._palm_pos)
    torch.testing.assert_close(palm_quat, state._palm_quat)
    state.robot.bind_mocap_pose_write("robot/right_palm_link", term_name="test_mocap_read")
    before_pose = state.robot.read_mocap_pose().copy()
    ratios = state.dr_ratios.clone()
    env.reset(torch.tensor([1], dtype=torch.int64, device=env.device))
    torch.testing.assert_close(
        env.scene.read_plan.joint_tensor_view("robot").joint_pos[[0, 2, 3]], before_joint[[0, 2, 3]]
    )
    # reset() refreshes derived body state for the whole batch. The position
    # cache may therefore change from the pre-reset, post-step derived frame;
    # mocap, joint state and fixed DR values below are the non-stale isolation
    # boundaries of this task.
    np.testing.assert_allclose(state.robot.read_mocap_pose()[[0, 2, 3]], before_pose[[0, 2, 3]])
    torch.testing.assert_close(state.dr_ratios, ratios, rtol=0.0, atol=1e-7)
    assert not np.allclose(state.robot.read_mocap_pose()[1], before_pose[1])


def test_curriculum_resume_survives_next_step_and_rejects_bad_state(env):
    wrapper = WujiWrapper(env, device="cpu")
    task(env).difficulty, task(env).adaptive = 0.4, 0.7
    for _ in range(3):
        env.step(torch.zeros((4, 20), dtype=torch.float32, device=env.device))
    value = wrapper.export_training_state()
    saved_step = env.step_counter
    task(env).difficulty = 0.0
    wrapper.import_training_state(value)
    assert task(env).difficulty == 0.4
    env.step(torch.zeros((4, 20), dtype=torch.float32, device=env.device))
    assert env.step_counter == saved_step + 1
    with pytest.raises(ValueError, match="curriculum"):
        wrapper.import_training_state({**value, "adaptive": float("nan")})

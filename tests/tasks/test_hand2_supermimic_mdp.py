"""Shape and differentiability checks for the Hand 2 SuperMimic adapter."""

from __future__ import annotations

import mujoco
import pytest
import torch

from unisupermimic.tasks.dex_reorient.hand2.kinematics import Hand2Kinematics
from unisupermimic.tasks.dex_reorient.hand2.mdp import Hand2SuperMimicMDP
from unisupermimic.tasks.dex_reorient.hand2.memory import (
    MEMORY_DIMS,
    flatten_memory,
    unflatten_memory,
)
from unisupermimic.tasks.dex_reorient.hand2.robot import JOINT_NAMES, TIP_SITES
from unisupermimic.tasks.dex_reorient.hand2.scene import materialize_scene


def _inputs(batch: int = 3):
    state = torch.zeros(batch, 58)
    state[:, 43] = 1.0
    state[:, 47] = 1.0
    state[:, 51] = 1.0
    memory = {name: torch.zeros(batch, width) for name, width in MEMORY_DIMS.items()}
    memory["goal"][:, 0] = 1.0
    auxiliary = torch.zeros(batch, 21)
    auxiliary[:, 3] = auxiliary[:, 7] = auxiliary[:, 11] = 1.0
    return state, memory, auxiliary


def test_hand2_supermimic_shapes_and_memory_roundtrip() -> None:
    mdp = Hand2SuperMimicMDP(torch.zeros(20), torch.full((20,), -1.0), torch.ones(20))
    state, memory, auxiliary = _inputs()
    assert mdp.obs(state, memory, auxiliary).shape == (3, 69)
    assert mdp.critic_features(state, memory, auxiliary, torch.ones(3, 46)).shape == (3, 211)
    assert mdp.world_model_inputs(state, torch.zeros(3, 20), torch.ones(3, 46), torch.zeros(3, 9)).shape == (3, 133)
    assert mdp.world_model_targets(state, state).shape == (3, 52)
    flat = flatten_memory(memory)
    assert flat.shape == (3, 100)
    restored = unflatten_memory(flat)
    for key in memory:
        torch.testing.assert_close(restored[key], memory[key])


def test_hand2_supermimic_gradients_through_state_and_controller() -> None:
    mdp = Hand2SuperMimicMDP(torch.zeros(20), torch.full((20,), -1.0), torch.ones(20))
    state, memory, auxiliary = _inputs(2)
    state.requires_grad_(True)
    action = torch.zeros(2, 20, requires_grad=True)
    reward = mdp.reward(state, memory, action)
    reward.sum().backward()
    assert torch.isfinite(state.grad).all()
    assert torch.isfinite(action.grad).all()


def test_hand2_rotation_reward_has_finite_aligned_gradient() -> None:
    mdp = Hand2SuperMimicMDP(torch.zeros(20), torch.full((20,), -1.0), torch.ones(20))
    state, memory, _ = _inputs(2)
    # The source uses atan2 here because acos(trace(R)) has an unbounded
    # derivative at alignment.
    state.requires_grad_(True)
    reward = mdp.reward(state, memory, torch.zeros(2, 20))
    reward.sum().backward()
    assert torch.isfinite(state.grad).all()


def test_hand2_fk_matches_mujoco_body_positions_and_is_differentiable() -> None:
    """FK used by imagined cage/rewards must use the exact MJCF frames."""
    model = mujoco.MjModel.from_xml_path(str(materialize_scene()))
    data = mujoco.MjData(model)
    body_names = (
        "robot/right_palm_link",
        *(f"robot/right_finger{finger}_link{link}"
          for finger in range(1, 6) for link in range(1, 5)),
    )
    kinematics = Hand2Kinematics(model, body_names, JOINT_NAMES, TIP_SITES, "cpu")
    q = torch.randn(2, 20) * 0.2
    qd = torch.randn(2, 20)
    positions, rotations, velocities, angular = kinematics.forward(q, qd)
    assert positions.shape == (2, 21, 3)
    assert rotations.shape == (2, 21, 3, 3)
    assert velocities.shape == angular.shape == (2, 21, 3)
    q_grad = q[:1].clone().requires_grad_(True)
    tips, tip_velocity = kinematics.tips(kinematics.forward(q_grad, qd[:1]))
    (tips.square().sum() + tip_velocity.square().sum()).backward()
    assert torch.isfinite(q_grad.grad).all()

    for row in range(q.shape[0]):
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        for index, name in enumerate(JOINT_NAMES):
            joint_id = model.joint(name).id
            data.qpos[model.jnt_qposadr[joint_id]] = float(q[row, index])
            data.qvel[model.jnt_dofadr[joint_id]] = float(qd[row, index])
        mujoco.mj_forward(model, data)
        root_id = model.body("robot/right_palm_link").id
        root_pos = data.xpos[root_id].copy()
        root_rot = data.xmat[root_id].reshape(3, 3).copy()
        expected = torch.as_tensor(
            (root_rot.T @ (data.xpos[[model.body(name).id for name in body_names]] - root_pos).T).T,
            dtype=torch.float32,
        )
        torch.testing.assert_close(positions[row], expected, atol=2e-5, rtol=2e-5)


def test_hand2_command_window_flag_stays_bounded_after_success_window() -> None:
    """The timer is separate from the binary source ``window`` flag."""
    mdp = Hand2SuperMimicMDP(
        torch.zeros(20), torch.full((20,), -1.0), torch.ones(20), hold_steps=13, switch_steps=50
    )
    state, memory, auxiliary = _inputs(2)
    memory["window"].fill_(1.0)
    memory["window_timer"].fill_(11.0)
    memory["reward_window"].fill_(10.0)
    memory["reward_hold"].fill_(10.0)
    memory["reward_snapshot"].fill_(10.0)
    goal = memory["goal"].clone()
    for _ in range(32):
        memory = mdp.command_step(state, memory, goal, soft=True)
        assert torch.isfinite(flatten_memory(memory)).all()
        assert bool((memory["window"] >= 0).all() and (memory["window"] <= 1).all())


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("state", lambda state, memory, auxiliary, context: state.__setitem__((0, 0), float("nan"))),
        ("memory", lambda state, memory, auxiliary, context: memory["age"].__setitem__((0, 0), float("inf"))),
        ("auxiliary", lambda state, memory, auxiliary, context: auxiliary.__setitem__((0, 3), float("nan"))),
        ("context", lambda state, memory, auxiliary, context: context.__setitem__((0, 0), float("inf"))),
    ],
)
def test_hand2_critic_features_report_nonfinite_source(field, mutate) -> None:
    """A bad imagined block must identify its source before RunningNorm."""
    mdp = Hand2SuperMimicMDP(torch.zeros(20), torch.full((20,), -1.0), torch.ones(20))
    state, memory, auxiliary = _inputs()
    context = torch.ones(3, 46)
    mutate(state, memory, auxiliary, context)
    with pytest.raises(FloatingPointError, match=field):
        mdp.critic_features(state, memory, auxiliary, context)


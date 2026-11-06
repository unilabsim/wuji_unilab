"""Feature contracts and aligned-state gradients for Pen SuperMimic."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from unisupermimic.rl.algos.dex_supermimic.pen import (
    PEN_MEMORY_DIMS,
    PenMDP,
    PenObservations,
    flatten_pen_memory,
    unflatten_pen_memory,
)
from unisupermimic.rl.algos.dex_supermimic.state import pack
from unisupermimic.tasks.pen_spin.commands import LINK_NAMES


class _Kinematics:
    names = LINK_NAMES

    def forward(self, q, qd):
        # An actual body-state dependent function keeps the link reward and
        # critic path in the gradient check without invoking any simulator.
        positions = torch.cat((q[..., :1], q[..., :1], q[..., :1]), -1)
        positions = positions.unsqueeze(-2).expand(-1, len(LINK_NAMES), -1)
        return positions, None, None, None


def _fixture(batch=2):
    frames = {
        "joint_pos": torch.zeros(3, 20),
        "object_pose": torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]).expand(3, -1),
        "link_pos": torch.zeros(3, 21, 3),
    }
    motion = SimpleNamespace(
        cfg=SimpleNamespace(object_position_tanh_scale_m=0.15),
        _frames=frames,
        _trajectory_ends=torch.tensor([3]),
    )
    env = SimpleNamespace(
        device="cpu",
        num_envs=batch,
        cfg=SimpleNamespace(observations={"policy": SimpleNamespace(enable_corruption=False)}),
        command_manager=SimpleNamespace(get_term=lambda name: motion),
    )
    reader = SimpleNamespace(
        env=env,
        device="cpu",
        kin=_Kinematics(),
        default_joint_pos=torch.zeros(20),
        lower=-torch.ones(20),
        upper=torch.ones(20),
        action_cfg=SimpleNamespace(
            alpha=0.5, mask_prob=0.0, mask_num_dofs=3,
            mask_duration_min=1, mask_duration_max=5, mask_curriculum_term=None,
        ),
        actuator_defaults={
            "kp": torch.ones(20),
            "kd": -torch.ones(20),
            "force_limit": torch.ones(20),
            "gear": torch.ones(20),
        },
        xi=torch.ones(batch, 46),
        palm_pos=torch.zeros(batch, 3),
        palm_rot=torch.eye(3).expand(batch, -1, -1),
        external_wrench=lambda: torch.zeros(batch, 174),
    )
    reader.mdp = mdp = PenMDP(reader)
    memory = {name: torch.zeros(batch, width) for name, width in PEN_MEMORY_DIMS.items()}
    memory["end"].fill_(3)
    memory["limits"].fill_(1)
    memory["camera_bias_rot"] = torch.eye(3).flatten().expand(batch, -1).clone()
    state = pack(
        torch.zeros(batch, 20),
        torch.zeros(batch, 20),
        torch.zeros(batch, 3),
        torch.eye(3).expand(batch, -1, -1),
        torch.zeros(batch, 3),
        torch.zeros(batch, 3),
    )
    return reader, mdp, state, memory


def test_pen_source_feature_widths_and_memory_roundtrip():
    reader, mdp, state, memory = _fixture()
    observation = PenObservations(reader)
    auxiliary = observation.auxiliary()
    context = observation.context(state, memory, auxiliary, reader.xi)
    assert mdp.observation(state, memory).shape == (2, 155)
    assert observation.critic(context).shape == (2, 301)
    assert auxiliary.shape == (2, 195)
    assert observation.disturbance(auxiliary).shape == (2, 174)
    assert observation.context_dim == 368
    assert context.shape == (2, 368)
    assert torch.cat((state, context), -1).shape == (2, 426)
    dynamics = observation.dynamics(state, reader.xi, observation.disturbance(auxiliary))
    assert torch.cat((dynamics, torch.zeros(2, 20)), -1).shape == (2, 298)
    flat = flatten_pen_memory(memory)
    assert flat.shape == (2, 230)
    restored = unflatten_pen_memory(flat)
    for key, value in memory.items():
        torch.testing.assert_close(restored[key], value)


def test_pen_aligned_reward_and_gradient_are_finite():
    reader, mdp, state, memory = _fixture()
    state.requires_grad_(True)
    action = torch.zeros(2, 20, requires_grad=True)
    residual, target = mdp.control_target(action, memory, torch.zeros_like(action))
    torque = mdp.pd_torque(state, target, reader.xi)
    reward, charge, terms = mdp.rewards(
        state, memory, action, residual, torch.zeros(2, 15), torque
    )
    torch.testing.assert_close(reward, torch.full((2,), 0.24))
    assert set(terms) == {
        "object_pos", "object_rot", "link_pos", "joint_pos", "action_rate", "torque", "overcurrent", "finger_collision"
    }
    assert charge.shape == (2, 20)
    (reward.sum() + mdp.termination(state, memory).sum()).backward()
    assert torch.isfinite(state.grad).all()
    assert torch.isfinite(action.grad).all()


def test_pen_reference_stops_at_trajectory_boundary_and_holds_first_frame():
    _, mdp, _, memory = _fixture()
    mdp.motion._frames["joint_pos"] = torch.arange(3).float()[:, None].expand(-1, 20)
    memory["time"][:, 0] = torch.tensor([-5.0, 100.0])
    q, _, _ = mdp.reference(memory)
    torch.testing.assert_close(q[:, 0], torch.tensor([0.0, 2.0]))


def _source_pen_classes():
    """Load the original pure adapter without importing its mjlab reader.

    The reference repository is kept outside the installed package.  AST
    isolation only removes the simulator reader class/import, preserving the
    actual source MDP and observation methods being compared.
    """
    path = Path(__file__).parents[3] / "references/Dex-Supermimic/src/dex_supermimic/rl/dex_supermimic/pen.py"
    if not path.is_file():
        pytest.skip("Dex-SuperMimic reference checkout is unavailable")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        node for node in tree.body
        if not (isinstance(node, ast.ClassDef) and node.name == "PenSimReader")
        and not (isinstance(node, ast.ImportFrom) and node.module == "sim")
    ]

    class _PureImports(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.module == "dex_supermimic.tasks.pen_spin.mdp.curriculum_utils":
                return None
            return node

    # The original mask method imports this pure helper inside its body. Load
    # that actual source helper too, while removing the package-level import
    # that would otherwise initialize the mjlab task registry.
    tree = _PureImports().visit(tree)
    namespace = {"__package__": "unisupermimic.rl.algos.dex_supermimic"}
    helper = path.parents[2] / "tasks/pen_spin/mdp/curriculum_utils.py"
    exec(compile(helper.read_text(encoding="utf-8"), str(helper), "exec"), namespace)
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["PenMDP"], namespace["PenObservations"]


@pytest.mark.parametrize("mask_prob", [0.0, 1.0])
def test_pen_pure_adapter_matches_original_source_formulas(mask_prob):
    source_mdp_class, source_observations_class = _source_pen_classes()
    reader, mdp, state, memory = _fixture()
    original = source_mdp_class.__new__(source_mdp_class)
    original.__dict__.update(mdp.__dict__)
    torch.manual_seed(17)
    state = state.clone()
    state[:, :43] += torch.randn_like(state[:, :43]) * 0.01
    memory["time"].fill_(1)
    for key in ("c1", "a1", "a2"):
        memory[key] = torch.randn_like(memory[key]) * 0.05
    memory["age"].fill_(3)
    memory["frozen_target"] = torch.randn_like(memory["frozen_target"]) * 0.05
    reader.action_cfg.mask_prob = mask_prob
    action = torch.randn(2, 20)
    contact = torch.rand(2, 15)
    actual_control_memory = dict(memory)
    expected_control_memory = dict(memory)
    torch.manual_seed(321)
    residual, target = mdp.control_target(action, actual_control_memory, torch.zeros_like(action))
    torch.manual_seed(321)
    expected_residual, expected_target = original.control_target(action, expected_control_memory, torch.zeros_like(action))
    torch.testing.assert_close(residual, expected_residual)
    torch.testing.assert_close(target, expected_target)
    for key, value in actual_control_memory.items():
        torch.testing.assert_close(value, expected_control_memory[key])
    torque = mdp.pd_torque(state, target, reader.xi)
    expected_torque = original.pd_torque(state, target, reader.xi)
    torch.testing.assert_close(torque, expected_torque)
    reward, charge, terms = mdp.rewards(state, memory, action, residual, contact, torque)
    expected_reward, expected_charge, expected_terms = original.rewards(
        state, memory, action, expected_residual, contact, expected_torque
    )
    torch.testing.assert_close(reward, expected_reward)
    torch.testing.assert_close(charge, expected_charge)
    for name, value in terms.items():
        torch.testing.assert_close(value, expected_terms[name])
    torch.testing.assert_close(mdp.observation(state, memory), original.observation(state, memory))
    torch.testing.assert_close(mdp.termination(state, memory), original.termination(state, memory))

    ours = PenObservations(reader)
    original_reader = SimpleNamespace(**{**vars(reader), "mdp": original})
    source = source_observations_class(original_reader)
    auxiliary = ours.auxiliary()
    context = ours.context(state, memory, auxiliary, reader.xi)
    expected_context = source.context(state, memory, auxiliary, reader.xi)
    torch.testing.assert_close(context, expected_context)
    torch.testing.assert_close(ours.critic(context), source.critic(expected_context))

    # The stochastic camera path is part of the source actor input contract.
    # Identical RNG state must produce identical pixels-as-axis-points and
    # persistent camera memory, including held-frame logic.
    mdp.camera_enabled = original.camera_enabled = True
    actual_memory = {key: value.clone() for key, value in memory.items()}
    expected_memory = {key: value.clone() for key, value in memory.items()}
    torch.manual_seed(123)
    actual = mdp.policy_observation(state, actual_memory, auxiliary)
    torch.manual_seed(123)
    expected = original.policy_observation(state, expected_memory, auxiliary)
    torch.testing.assert_close(actual, expected)
    for key, value in actual_memory.items():
        torch.testing.assert_close(value, expected_memory[key])

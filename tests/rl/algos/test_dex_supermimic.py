"""Source contracts and gradients for the independent Dex SuperMimic core."""

from __future__ import annotations

import csv
import time
from types import SimpleNamespace

import pytest
import torch

from unisupermimic.rl.algos.dex_supermimic.buffer import ReplayBuffer
from unisupermimic.rl.algos.dex_supermimic.common import RunningNorm
from unisupermimic.rl.algos.dex_supermimic.config import DexSuperMimicCfg
from unisupermimic.rl.algos.dex_supermimic.env_bridge import SourceEnvBridge
from unisupermimic.rl.algos.dex_supermimic.metrics import MetricLogger
from unisupermimic.rl.algos.dex_supermimic.networks import Actor, Critic, FailureHead, WorldModel
from unisupermimic.rl.algos.dex_supermimic.runner import DexSuperMimicRunner
from unisupermimic.rl.algos.dex_supermimic.state import difference, pack, unpack


def _state(batch=2, *, dtype=torch.float32):
    return pack(
        torch.zeros(batch, 20, dtype=dtype),
        torch.zeros(batch, 20, dtype=dtype),
        torch.zeros(batch, 3, dtype=dtype),
        torch.eye(3, dtype=dtype).expand(batch, 3, 3),
        torch.zeros(batch, 3, dtype=dtype),
        torch.zeros(batch, 3, dtype=dtype),
    )


def test_pen_source_network_contract_and_delta_integration():
    """Check the documented 298 -> 67 head against an independent SO(3) oracle."""
    x = _state()
    model = WorldModel(278, 15, hidden=(), amp=True)
    assert model.net[-1].in_features == 298
    assert model.net[-1].out_features == 67
    assert Actor(155)(torch.zeros(2, 155)).shape == (2, 20)
    assert Critic(301)(torch.zeros(2, 301)).shape == (2,)
    assert FailureHead(368)(torch.zeros(2, 426)).shape == (2,)

    # A known normalized output exercises scale/mean and all six state blocks.
    raw = torch.linspace(-0.03, 0.04, 52)
    with torch.no_grad():
        model.net[-1].bias[:52].copy_(raw)
        model.delta_norm.mean.fill_(0.01)
        model.delta_norm.var.fill_(0.25)
    predicted, contact = model(x, torch.zeros(2, 20), torch.zeros(2, 278))
    delta = raw * (0.25 + 1e-8) ** 0.5 + 0.01
    dq, dqd, dp, dr, dv, dw = delta.split((20, 20, 3, 3, 3, 3))
    skew = torch.tensor([[0.0, -dr[2], dr[1]], [dr[2], 0.0, -dr[0]], [-dr[1], dr[0], 0.0]])
    expected_rotation = torch.linalg.matrix_exp(skew).expand(2, 3, 3)
    q, qd, p, rotation, v, w = unpack(predicted)
    for actual, expected in ((q, dq), (qd, dqd), (p, dp), (v, dv), (w, dw)):
        torch.testing.assert_close(actual, expected.expand_as(actual))
    torch.testing.assert_close(rotation, expected_rotation, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(contact, torch.full((2, 15), -3.0))


@pytest.mark.parametrize("angle", (0.0, 1e-10, 1e-6))
def test_world_model_identity_rotation_has_finite_open_loop_gradient(angle):
    """Zero-initialized dynamics must remain differentiable near identity."""
    model = WorldModel(278, 15, hidden=(), amp=False).double()
    x = _state(dtype=torch.float64)
    with torch.no_grad():
        model.net[-1].bias[43:46].fill_(angle)
    for _ in range(4):
        dynamics = torch.cat((x, torch.zeros(2, 220, dtype=x.dtype)), -1)
        x, _ = model(x, torch.zeros(2, 20, dtype=x.dtype), dynamics)
    # A nonzero rotation target requires a gradient even at an identity guess.
    target = _state(dtype=torch.float64)
    target[:, 43:52] = torch.linalg.matrix_exp(
        torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -0.1], [0.0, 0.1, 0.0]], dtype=target.dtype)
    ).flatten().expand(2, -1)
    loss = difference(x, target).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.net[-1].bias.grad[43:46].abs().sum() > 0


def test_policy_update_receives_gradient_through_predicted_state():
    """Exercise recurrent actor -> target -> WM -> state -> actor data flow."""
    torch.manual_seed(7)
    actor = Actor(155)
    model = WorldModel(278, 15, hidden=(), amp=False)
    # A fitted local response: q_delta depends on q and executed target;
    # rotation_delta also depends on action so identity autograd is covered.
    with torch.no_grad():
        for joint in range(20):
            model.net[-1].weight[joint, joint] = 0.01
            model.net[-1].weight[joint, 278 + joint] = 0.03
        model.net[-1].weight[43:46, 278:281] = torch.eye(3) * 0.01
    model.requires_grad_(False)
    actor.norm.update(torch.randn(8, 155))
    x = _state()
    residual = torch.zeros(2, 20)
    previous_action = torch.zeros(2, 20)
    objective = torch.zeros(2)
    for _ in range(4):
        q, qd, position, rotation, *_ = unpack(x)
        # Same source Pen field widths: motion 89, q 20, qd 20, points 6, a1 20.
        axis = rotation[..., :, 0] * 0.10
        points = torch.cat((position + axis, position - axis), -1)
        observation = torch.cat((torch.zeros(2, 89), q, qd * 0.25, 0.15 * torch.tanh(points / 0.15), previous_action), -1)
        action = actor(observation)
        residual = 0.5 * (0.5 * action).clamp(-0.6, 0.6) + 0.5 * residual
        dynamics = torch.cat((x, torch.zeros(2, 220)), -1)
        x, _ = model(x, residual, dynamics)
        q_next = unpack(x)[0]
        objective = objective + torch.exp(-((q_next - 0.2) / 0.1).square()).mean(-1)
        previous_action = action
    loss = -objective.mean()
    optimizer = torch.optim.AdamW(actor.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.0)
    before = actor.net[-1].weight.detach().clone()
    optimizer.zero_grad()
    loss.backward()
    gradients = [p.grad for p in actor.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0
    optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.equal(actor.net[-1].weight, before)


def test_numerical_errors_fail_before_normalization_and_replay():
    with pytest.raises(FloatingPointError, match="RunningNorm.update input"):
        RunningNorm(2).update(torch.tensor([[float("nan"), 0.0]]))
    replay = ReplayBuffer(8, 2, "cpu", holdout_fraction=0)
    with pytest.raises(FloatingPointError, match="transition.x"):
        replay.add({"x": torch.tensor([[float("inf")], [0.0]])})


def test_supermimic_config_and_csv_log(tmp_path):
    cfg = DexSuperMimicCfg(task="hand2-pen-spin", logger="none", device="cpu")
    assert cfg.max_iterations == 8000
    assert cfg.wm_hidden == (2048, 2048, 2048)
    assert cfg.algorithm == "world_model"
    with pytest.raises(ValueError, match="SuperMimic"):
        DexSuperMimicCfg(task="hand2-pen-spin", algorithm="ppo")
    logger = MetricLogger("none", log_dir=tmp_path, project="test", entity="test")
    logger.log_scalars({"wm/wm_loss": 0.5, "policy/actor_loss": -1.0}, 500)
    logger.close()
    with (tmp_path / "train_log.csv").open(newline="") as stream:
        (row,) = tuple(csv.DictReader(stream))
    assert row["iteration"] == "500"
    assert float(row["wm/wm_loss"]) == 0.5
    assert float(row["policy/actor_loss"]) == -1.0


def test_csv_resume_adds_episode_columns_without_losing_existing_data(tmp_path):
    path = tmp_path / "train_log.csv"
    path.write_text("iteration,wm/wm_loss,custom/retained\n10,0.5,7\n")
    logger = MetricLogger("none", log_dir=tmp_path, project="test", entity="test")
    logger.log_scalars({"task/episode_object_ori_tracking_error_deg": 12.5}, 20)
    logger.close()
    with path.open(newline="") as stream:
        old, new = tuple(csv.DictReader(stream))
    assert old["iteration"] == "10"
    assert old["wm/wm_loss"] == "0.5"
    assert old["custom/retained"] == "7"
    assert old["task/episode_object_ori_tracking_error_deg"] == ""
    assert new["iteration"] == "20"
    assert float(new["task/episode_object_ori_tracking_error_deg"]) == 12.5


def test_checkpoint_preserves_task_modules_iteration_and_cpu_rng(tmp_path):
    """Resume reproduces the next random draw and rejects another clip catalog."""
    runner = DexSuperMimicRunner.__new__(DexSuperMimicRunner)
    runner.cfg = DexSuperMimicCfg(task="hand2-pen-spin", logger="none", device="cpu")
    runner.device = torch.device("cpu")
    runner.is_pen = True
    runner.context_dim = 368
    runner.contact_dim = 15
    runner.mdp = SimpleNamespace(reference_signature="fixed-reference")
    runner.reader = SimpleNamespace(
        context_names=[("friction_ratio", 1)],
        robot=SimpleNamespace(joint_names=[f"joint_{index}" for index in range(20)]),
    )
    native = SimpleNamespace(
        common_step_counter=123,
        max_common_steps=2064,
        _adaptive_episode_curriculum=0.25,
        _reorient_success_curriculum=0.5,
        cfg=SimpleNamespace(auto_reset=True),
        set_autoreset=lambda enabled: None,
    )
    native.import_training_state = lambda state: vars(native).update(
        common_step_counter=state["step_counter"], step_counter=state["step_counter"]
    )
    runner.env = SourceEnvBridge(native)
    runner.actor = torch.nn.Linear(2, 2)
    runner.wm = torch.nn.Linear(2, 2)
    runner.fail_head = torch.nn.Linear(2, 1)
    runner.critic = torch.nn.Linear(2, 1)
    runner.torque_norm = RunningNorm(20)
    runner.optimizers = {
        name: torch.optim.Adam(module.parameters(), lr=runner.cfg.lr)
        for name, module in runner.modules().items()
    }
    runner.iteration = 499
    runner.sim_steps = 10000
    runner.wall_offset = 0.0
    runner.started_at = time.perf_counter()
    path = tmp_path / "nested" / "model_500.pt"
    original_weight = runner.actor.weight.detach().clone()
    torch.manual_seed(42)
    runner.save(path, next_iteration=500)
    expected_random = torch.rand(8)
    assert path.is_file()
    assert not path.with_suffix(".tmp").exists()
    with torch.no_grad():
        runner.actor.weight.fill_(100)
    runner.env.common_step_counter = 0
    native._adaptive_episode_curriculum = 0.0
    native._reorient_success_curriculum = 0.0
    torch.manual_seed(1)
    runner.load(path)
    torch.testing.assert_close(torch.rand(8), expected_random)
    torch.testing.assert_close(runner.actor.weight, original_weight)
    assert runner.iteration == 500
    assert runner.sim_steps == 10000
    assert runner.env.common_step_counter == 123
    assert native.common_step_counter == 123
    assert native.step_counter == 123
    assert native._adaptive_episode_curriculum == 0.25
    assert native._reorient_success_curriculum == 0.5
    runner.mdp.reference_signature = "changed-reference"
    with pytest.raises(ValueError, match="reference data mismatch"):
        runner.load(path)


def test_bridge_training_progress_is_shared_with_native_environment():
    """Real disturbance ramps and imagination must use one training timeline."""
    native = SimpleNamespace(
        common_step_counter=10,
        max_common_steps=100,
        _adaptive_episode_curriculum=0.1,
        _reorient_success_curriculum=0.2,
        cfg=SimpleNamespace(auto_reset=True),
        set_autoreset=lambda enabled: None,
    )
    native.import_training_state = lambda state: vars(native).update(
        common_step_counter=state["step_counter"], step_counter=state["step_counter"]
    )
    bridge = SourceEnvBridge(native)
    bridge.max_common_steps = 20064
    bridge.common_step_counter = 501
    assert native.max_common_steps == 20064
    assert native.common_step_counter == 501
    assert native.step_counter == 501
    bridge._adaptive_episode_curriculum = 0.3
    bridge._reorient_success_curriculum = 0.4
    assert native._adaptive_episode_curriculum == 0.3
    assert native._reorient_success_curriculum == 0.4
    native.common_step_counter += 1
    native._adaptive_episode_curriculum = 0.35
    assert bridge.common_step_counter == 502
    assert bridge._adaptive_episode_curriculum == 0.35
    # Reader/camera caches belong to the facade, not the native env.
    bridge.training_reader = object()
    bridge._policy = torch.zeros(1, 155)
    assert not hasattr(native, "training_reader")
    assert not hasattr(native, "_policy")

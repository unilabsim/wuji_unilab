"""Compare the real Pen actor velocity slice with the source noise pipeline."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from unilab.base import registry
from unilab.base.config_adapter import EnvConfigAdapter
from unilab.training import ensure_registries

ROOT = Path(__file__).parents[2]
SOURCE = ROOT / "references/Dex-Supermimic/src/dex_supermimic/tasks/pen_spin/pen_spin_env_cfg.py"
VELOCITY = slice(109, 129)  # motion(89), joint_pos(20), joint_vel(20)


def _source_velocity_noise_and_scale():
    if not SOURCE.is_file():
        pytest.skip("Dex-SuperMimic reference checkout is unavailable")
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    factory = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "make_pen_spin_env_cfg")
    terms = next(node.value for node in factory.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "policy_terms"
                         for target in node.targets))
    term = next(value for key, value in zip(terms.keys, terms.values, strict=True)
                if ast.literal_eval(key) == "joint_vel")
    keywords = {keyword.arg: keyword.value for keyword in term.keywords}
    noise = {keyword.arg: ast.literal_eval(keyword.value)
             for keyword in keywords["noise"].keywords}
    return noise["n_min"], noise["n_max"], ast.literal_eval(keywords["scale"])


@pytest.fixture
def pen_env():
    ensure_registries()
    with initialize_config_dir(
        config_dir=str(ROOT / "src/unisupermimic/conf/supermimic"), version_base="1.3"
    ):
        cfg = compose("config_pen_spin_supermimic", overrides=[
            "task=dex_hand2_pen_spin/mujoco", "algo.num_envs=2",
        ])
    # Isolate velocity noise in the actual 155D manager output. A synthetic
    # reference keeps this test independent of the downloaded clip catalog.
    cfg.env.commands.motion.recipe_path = ""
    cfg.env.commands.motion.clips_root = None
    cfg.env.commands.motion.synthetic_reference = True
    cfg.env.observations.policy.terms.joint_pos.params.noise_bound = 0.0
    cfg.env.observations.policy.terms.object_pose.func = (
        "unisupermimic.tasks.pen_spin.observations.PenObjectPoseFeature"
    )
    override = EnvConfigAdapter(cfg, root_dir=ROOT, algo_name="supermimic").build_task_env_cfg_override()
    env = registry.make(
        "DexHand2PenSpin", sim_backend="mujoco", num_envs=2, env_cfg_override=override
    )
    try:
        env.reset()
        yield env
    finally:
        env.close()


def test_real_pen_actor_velocity_matches_source_noise_before_scale(pen_env):
    low, high, scale = _source_velocity_noise_and_scale()
    assert (low, high, scale) == (-0.05, 0.05, 0.25)
    term = pen_env.observation_manager.get_term_cfg("policy", "joint_vel")
    assert term.params["noise_bound"] == high
    raw = pen_env.scene.read_plan.joint_tensor_view("robot").joint_vel.clone()
    # mjlab 0fb8a681 ObservationManager.compute_group applies noise, clip,
    # then scale. Use the source task's actual bounds with the same RNG draw.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(734)
        expected = (raw + torch.empty_like(raw).uniform_(low, high)) * scale
        torch.manual_seed(734)
        groups = pen_env.observation_manager.compute(update_history=True)
    assert groups["policy"].shape == (2, 155)
    actual = groups["policy"][:, VELOCITY]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    error = (actual - raw * scale).abs()
    assert error.max() <= high * scale + 1e-7
    assert error.max() > 0.01  # The sampled actor input must contain noise.
    torch.testing.assert_close(groups["critic"][:, VELOCITY], raw * scale, rtol=0, atol=0)


def test_real_pen_play_velocity_disables_noise_without_consuming_rng(pen_env):
    pen_env.cfg.observations["policy"].enable_corruption = False
    # Keep the training noise bound present: the play corruption gate must
    # disable it, rather than relying only on a zero-valued override.
    assert pen_env.observation_manager.get_term_cfg("policy", "joint_vel").params["noise_bound"] == 0.05
    raw = pen_env.scene.read_plan.joint_tensor_view("robot").joint_vel.clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        before = torch.get_rng_state().clone()
        groups = pen_env.observation_manager.compute(update_history=True)
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    for group in ("policy", "critic"):
        torch.testing.assert_close(groups[group][:, VELOCITY], raw * 0.25, rtol=0, atol=0)

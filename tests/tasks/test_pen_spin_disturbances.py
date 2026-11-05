"""Exercise nonzero source disturbances, phase changes, and partial resets."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import mujoco
import pytest
import torch

from unisupermimic.tasks.pen_spin.commands import (
    LINK_NAMES,
    _quat_conjugate,
    _quat_multiply,
    _quat_rotate,
)
from unisupermimic.tasks.pen_spin.curriculums import get_curriculum_value
from unisupermimic.tasks.pen_spin.disturbances import (
    OBJECT_BODY_NAMES,
    PenDisturbances,
    _sample_uniform,
    apply_link_random_force,
    apply_object_disturbance_wrench,
)
from unisupermimic.tasks.pen_spin.scene import materialize_scene


class _Asset:
    def __init__(self, names, model, quats, fields):
        self.body_names = tuple(names)
        self.num_bodies = len(names)
        ids = torch.tensor([model.body(name).id for name in names])
        self.indexing = SimpleNamespace(body_ids=ids)
        link_quat = quats[:, ids]
        self.data = SimpleNamespace(
            body_link_quat_w=link_quat,
            body_com_quat_w=_quat_multiply(link_quat, fields["body_iquat"][:, ids]),
        )
        self.last_wrench = None

    def find_bodies(self, names, preserve_order=False):
        assert preserve_order
        return [self.body_names.index(name) for name in names], list(names)

    def bind_body_wrench(self, ids, *, torque, term_name):
        assert torque and term_name == "pen_disturbances"
        return torch.tensor(ids), self.indexing.body_ids[ids].numpy()

    def write_external_wrench_to_sim(self, force, torque, **kwargs):
        self.last_wrench = (force.clone(), torque.clone())


def _fixture():
    model_file = materialize_scene()
    model = mujoco.MjModel.from_xml_path(str(model_file))
    batch = 3
    torch.manual_seed(9321)
    quats = torch.randn(batch, model.nbody, 4)
    quats /= quats.norm(dim=-1, keepdim=True)
    principal = torch.randn_like(quats)
    principal /= principal.norm(dim=-1, keepdim=True)
    factors = torch.tensor([1.0, 1.3, 0.7])
    fields = {
        "body_mass": torch.tensor(model.body_mass.copy(), dtype=torch.float32)[None] * factors[:, None],
        "body_inertia": torch.tensor(model.body_inertia.copy(), dtype=torch.float32)[None] * factors[:, None, None],
        "body_iquat": principal,
    }
    scene = {
        "robot": _Asset(LINK_NAMES, model, quats, fields),
        "object": _Asset(OBJECT_BODY_NAMES, model, quats, fields),
    }

    def read_body(entity, names):
        ids = [model.body(name).id for name in names]
        return SimpleNamespace(quat_w=quats[:, ids])

    class _Scene(dict):
        read_plan = SimpleNamespace(body_tensor_view=read_body)

    plans = []
    env = SimpleNamespace(
        num_envs=batch, device="cpu", step_dt=0.02,
        scene=_Scene(scene), episode_length_buf=torch.zeros(batch, dtype=torch.long),
        backend=SimpleNamespace(
            backend_type="mujoco", model=model,
            get_scene_model_file=lambda: model_file,
            apply_interval_randomization=plans.append,
        ),
        curriculum_manager=SimpleNamespace(_curriculum_state={"disturbance_success_rate": {"value": 0.65}}),
        _pen_model_fields=fields,
        sim=SimpleNamespace(model=SimpleNamespace(**fields)),
    )
    wrapper = PenDisturbances(SimpleNamespace(params={}), env)
    return env, wrapper, plans


def _source_classes():
    path = Path(__file__).parents[2] / "references/Dex-Supermimic/src/dex_supermimic/tasks/pen_spin/mdp/events.py"
    if not path.is_file():
        pytest.skip("Dex-SuperMimic reference checkout is unavailable")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    classes = []
    for name in ("apply_object_disturbance_wrench", "apply_link_random_force"):
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
        cls.body = [
            node for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name in {
                "__call__", "reset", "_random_unit_dirs", "_selected_body_field",
            }
        ]
        classes.append(cls)
    module = ast.Module(
        body=[ast.parse("from __future__ import annotations").body[0], *classes],
        type_ignores=[],
    )
    namespace = {
        "torch": torch, "math": math, "apply_body_impulse": object,
        "get_curriculum_value": get_curriculum_value,
        "quat_mul": _quat_multiply, "quat_apply": _quat_rotate,
        "quat_apply_inverse": lambda q, v: _quat_rotate(_quat_conjugate(q), v),
        "sample_uniform": _sample_uniform,
        "_DEFAULT_OBJECT_CFG": SimpleNamespace(name="object", body_ids=torch.tensor([0])),
    }
    exec(compile(module, str(path), "exec"), namespace)
    return tuple(namespace[name] for name in ("apply_object_disturbance_wrench", "apply_link_random_force"))


def _clone_process(actual, source_type):
    expected = source_type.__new__(source_type)
    expected.__dict__.update({
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in vars(actual).items()
    })
    return expected


def _assert_process_equal(actual, expected):
    for name, value in vars(actual).items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, getattr(expected, name), rtol=1e-6, atol=1e-9)


def test_nonzero_wrench_processes_match_source_through_phase_changes_and_reset():
    env, wrapper, _ = _fixture()
    original_object, original_fingers = _source_classes()
    expected_object = _clone_process(wrapper.object, original_object)
    expected_fingers = _clone_process(wrapper.fingertips, original_fingers)
    object_cfg = SimpleNamespace(name="object", body_ids=wrapper.object._body_ids)
    finger_cfg = SimpleNamespace(name="robot", body_ids=wrapper.fingertips._body_ids)
    observed_object = observed_tips = observed_silence = False
    for tick in range(220):
        env.episode_length_buf.fill_(tick)
        for actual, expected, params, cfg in (
            (wrapper.object, expected_object, wrapper._object_params, object_cfg),
            (wrapper.fingertips, expected_fingers, wrapper._fingertip_params, finger_cfg),
        ):
            rng = torch.random.get_rng_state()
            forces, torques = actual(env, None, **params)
            after_rng = torch.random.get_rng_state()
            torch.random.set_rng_state(rng)
            expected(env, None, asset_cfg=cfg, **params)
            torch.testing.assert_close(torch.random.get_rng_state(), after_rng)
            source_force, source_torque = env.scene[cfg.name].last_wrench
            torch.testing.assert_close(forces, source_force, rtol=1e-6, atol=1e-9)
            torch.testing.assert_close(torques, source_torque, rtol=1e-6, atol=1e-9)
            _assert_process_equal(actual, expected)
        if tick < 50:
            assert not wrapper.object._phase_active.any()
        else:
            observed_object |= bool(wrapper.object._force.any())
            observed_silence |= bool((~wrapper.object._phase_active).any())
        observed_tips |= bool(wrapper.fingertips._force.any())
        if tick == 80:
            reset_ids = torch.tensor([1])
            for actual, expected in ((wrapper.object, expected_object), (wrapper.fingertips, expected_fingers)):
                rng = torch.random.get_rng_state()
                actual.reset(reset_ids)
                after_rng = torch.random.get_rng_state()
                torch.random.set_rng_state(rng)
                expected.reset(reset_ids)
                torch.testing.assert_close(torch.random.get_rng_state(), after_rng)
                _assert_process_equal(actual, expected)
    assert observed_object and observed_tips and observed_silence


def test_object_and_fingertips_share_one_plan_and_partial_reset_preserves_other_rows():
    env, wrapper, plans = _fixture()
    env.episode_length_buf.fill_(100)
    wrapper(env)
    assert len(plans) == 1
    assert len(plans[0].ops) == 2
    body_ids = plans[0].ops[0].body_ids
    assert body_ids.numel() == 6
    selected = env._pen_external_wrench[:, body_ids]
    assert selected[:, 0, :3].abs().sum() > 0
    assert selected[:, 1:, :3].abs().sum() > 0
    assert selected[..., 3:].abs().sum() > 0
    torch.testing.assert_close(plans[0].ops[0].payload, selected[..., :3])
    torch.testing.assert_close(plans[0].ops[1].payload, selected[..., 3:])
    before = env._pen_external_wrench.clone()
    wrapper.reset(torch.tensor([1]))
    assert len(plans) == 2
    assert not env._pen_external_wrench[1].any()
    torch.testing.assert_close(env._pen_external_wrench[[0, 2]], before[[0, 2]], rtol=0, atol=0)
    assert not wrapper.object._phase_active[1]
    assert not wrapper.fingertips._active[1].any()
    assert (wrapper.fingertips._interval_time_left[1] > 0).all()


def test_curriculum_scales_both_physical_force_producers():
    env, wrapper, _ = _fixture()
    env.episode_length_buf.fill_(100)
    env.curriculum_manager._curriculum_state["disturbance_success_rate"] = {"value": 0.0}
    rng = torch.random.get_rng_state()
    zero_object = apply_object_disturbance_wrench(SimpleNamespace(params={}), env)
    zero_tip = apply_link_random_force(SimpleNamespace(params=wrapper._fingertip_params), env)
    zero_force, zero_torque = zero_object(env, None, **wrapper._object_params)
    zero_tip_force, zero_tip_torque = zero_tip(env, None, **wrapper._fingertip_params)
    assert not zero_force.any() and not zero_torque.any()
    assert not zero_tip_force.any() and not zero_tip_torque.any()
    torch.random.set_rng_state(rng)
    env.curriculum_manager._curriculum_state["disturbance_success_rate"] = {"value": 1.0}
    full_object = apply_object_disturbance_wrench(SimpleNamespace(params={}), env)
    full_tip = apply_link_random_force(SimpleNamespace(params=wrapper._fingertip_params), env)
    full_force, full_torque = full_object(env, None, **wrapper._object_params)
    full_tip_force, full_tip_torque = full_tip(env, None, **wrapper._fingertip_params)
    assert full_force.any() and full_torque.any()
    assert full_tip_force.any() and full_tip_torque.any()


def test_real_native_env_applies_both_nonzero_wrenches_after_startup_randomization():
    from hydra import compose, initialize_config_dir
    from unilab.base import registry
    from unilab.base.config_adapter import EnvConfigAdapter
    from unilab.training import ensure_registries

    root = Path(__file__).parents[2]
    ensure_registries()
    with initialize_config_dir(
        config_dir=str(root / "src/unisupermimic/conf/supermimic"), version_base="1.3"
    ):
        cfg = compose("config_pen_spin", overrides=[
            "algo.num_envs=2", "env.auto_reset=false",
            "env.commands.motion.first_frame_hold_prob=0.0",
            "env.commands.motion.trajectory_sampling_mode=uniform",
        ])
    override = EnvConfigAdapter(cfg, root_dir=root, algo_name="ppo").build_task_env_cfg_override()
    env = registry.make("DexHand2PenSpin", sim_backend="mujoco", env_cfg_override=override, num_envs=2)
    try:
        env.reset()
        env.common_step_counter = 240_000
        env.curriculum_manager.compute(torch.arange(2))
        assert get_curriculum_value(env, "disturbance_success_rate") == 1.0
        env.set_episode_length_buf(torch.full((2,), 60, dtype=torch.long))
        term = env.event_manager.get_term_cfg("source_disturbances").func
        term.fingertips._interval_time_left.zero_()
        result = env.step(torch.zeros(2, 20))
        assert torch.isfinite(result.reward).all()
        selected = env._pen_external_wrench[:, term._body_columns]
        assert selected[:, 0, :3].abs().sum() > 0
        assert selected[:, 1:, :3].abs().sum() > 0
        assert selected[..., 3:].abs().sum() > 0
        assert env._pen_external_wrench.shape[1] == env._pen_physics_model.nbody
        # Consume the staged force/torque plan through a second physical step.
        result = env.step(torch.zeros(2, 20))
        assert torch.isfinite(result.reward).all()
        physical = env.backend.get_state_views(("qpos", "qvel"), device="cpu")
        assert all(torch.isfinite(value).all() for value in physical.values())
        before = env._pen_external_wrench.clone()
        env.reset(env_ids=torch.tensor([0]))
        assert not env._pen_external_wrench[0].any()
        torch.testing.assert_close(env._pen_external_wrench[1], before[1], rtol=0, atol=0)
    finally:
        env.close()

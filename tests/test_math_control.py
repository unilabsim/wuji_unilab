from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from wuji_unilab.tasks.reorient.actions import WujiAction, WujiActionCfg
from wuji_unilab.tasks.reorient.math import (
    conjugate,
    multiply,
    random_quaternions,
    rotate,
    rotation6d,
)


def test_rotations_agree_with_independent_scipy_convention():
    rng = np.random.default_rng(7)
    q = random_quaternions(rng, 30)
    vectors = rng.normal(size=(30, 3))
    rotations = Rotation.from_quat(q[:, [1, 2, 3, 0]])
    np.testing.assert_allclose(rotate(q, vectors), rotations.apply(vectors), atol=1e-6)
    np.testing.assert_allclose(
        rotation6d(q), rotations.as_matrix().reshape(30, 9)[:, 3:], atol=3e-7
    )
    np.testing.assert_allclose(multiply(q, conjugate(q)), np.tile([1, 0, 0, 0], (30, 1)), atol=2e-7)
    np.testing.assert_allclose(
        rotate(q[:, None], vectors[:, None]), rotate(q, vectors)[:, None], atol=1e-7
    )


def test_warmup_clipping_smoothing_and_selected_reset():
    robot = SimpleNamespace(
        data=SimpleNamespace(
            default_joint_pos=np.zeros((2, 20), np.float32),
            soft_joint_pos_limits=np.tile([-1, 1], (20, 1)),
        )
    )
    env = SimpleNamespace(
        scene={"robot": robot}, num_envs=2, episode_length_buf=np.array([7, 8]), step_dt=0.05
    )
    action = WujiAction(WujiActionCfg(entity_name="robot"), env)
    action.process_actions(np.full((2, 20), 5, np.float32))
    np.testing.assert_allclose(action.target[0], 0)
    np.testing.assert_allclose(action.target[1], 0.25)
    assert np.all(action.raw == 5)
    action.process_actions(np.full((2, 20), 5, np.float32))
    np.testing.assert_allclose(action.target[1], 0.375)
    action.reset(np.array([0]))
    np.testing.assert_allclose(action.target[1], 0.375)

"""CPU contract tests for the Wuji curriculum checkpoint runner."""

import pytest
import torch
from tensordict import TensorDict

from wuji_unilab.rl.runtime import _STATE_KEY, WujiOnPolicyRunner


class FakeEnv:
    """Minimal VecEnv + training-state provider for the runner round-trip."""

    def __init__(self):
        self.cfg = {}
        self.num_envs = 2
        self.num_actions = 3
        self.device = "cpu"
        self.max_episode_length = 10
        self.difficulty = 0.5

    def get_observations(self):
        return TensorDict({"actor": torch.zeros(2, 4), "critic": torch.zeros(2, 5)}, batch_size=2)

    def export_training_state(self):
        return {"schema": "fake", "version": 1, "value": self.difficulty}

    def import_training_state(self, value):
        self.difficulty = value["value"]


def _train_cfg():
    return {
        "seed": 1,
        "num_steps_per_env": 4,
        "save_interval": 5,
        "experiment_name": "fake",
        "run_name": "",
        "logger": "none",
        "obs_groups": {"actor": ["actor"], "critic": ["critic"]},
        "actor": {"class_name": "rsl_rl.models.MLPModel", "hidden_dims": [8], "activation": "elu"},
        "critic": {"class_name": "rsl_rl.models.MLPModel", "hidden_dims": [8], "activation": "elu"},
        "algorithm": {
            "class_name": "rsl_rl.algorithms:PPO",
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
        },
    }


def test_curriculum_state_round_trip(tmp_path):
    env = FakeEnv()
    runner = WujiOnPolicyRunner(env, _train_cfg(), log_dir=str(tmp_path), device="cpu")
    checkpoint = tmp_path / "model_0.pt"
    runner.save(str(checkpoint))

    saved = torch.load(checkpoint, weights_only=False)
    assert saved["infos"][_STATE_KEY] == {
        "version": 1,
        "state": {"schema": "fake", "version": 1, "value": 0.5},
    }

    env.difficulty = 0.0
    resumed = WujiOnPolicyRunner(env, _train_cfg(), log_dir=str(tmp_path), device="cpu")
    resumed.load(str(checkpoint), map_location="cpu")
    assert env.difficulty == 0.5


def test_policy_only_load_skips_curriculum_state(tmp_path):
    env = FakeEnv()
    runner = WujiOnPolicyRunner(env, _train_cfg(), log_dir=str(tmp_path), device="cpu")
    checkpoint = tmp_path / "model_0.pt"
    runner.save(str(checkpoint))

    env.difficulty = 0.1
    runner.load(
        str(checkpoint), load_cfg={"actor": True}, restore_training_state=False, map_location="cpu"
    )
    assert env.difficulty == 0.1
    with pytest.raises(ValueError, match="actor-only load_cfg"):
        runner.load(str(checkpoint), restore_training_state=False)


def test_missing_envelope_aborts_resume(tmp_path):
    env = FakeEnv()
    runner = WujiOnPolicyRunner(env, _train_cfg(), log_dir=str(tmp_path), device="cpu")
    checkpoint = tmp_path / "model_0.pt"
    runner.save(str(checkpoint))
    saved = torch.load(checkpoint, weights_only=False)
    del saved["infos"][_STATE_KEY]
    torch.save(saved, checkpoint)

    with pytest.raises(ValueError, match="missing required training state"):
        runner.load(str(checkpoint))


def test_runner_requires_training_state_provider(tmp_path):
    with pytest.raises(TypeError, match="export_training_state"):
        WujiOnPolicyRunner(object(), _train_cfg(), log_dir=str(tmp_path), device="cpu")


def test_round_trip_composes_with_unilab_resume_patch(tmp_path):
    from unilab.training.experiment import patch_rsl_rl_resume_state

    patch_rsl_rl_resume_state()
    env = FakeEnv()
    runner = WujiOnPolicyRunner(env, _train_cfg(), log_dir=str(tmp_path), device="cpu")
    checkpoint = tmp_path / "model_0.pt"
    runner.save(str(checkpoint))

    saved = torch.load(checkpoint, weights_only=False)
    assert saved["infos"][_STATE_KEY]["state"]["value"] == 0.5
    assert "unilab_logger_state" in saved

    env.difficulty = 0.0
    runner.load(str(checkpoint), map_location="cpu")
    assert env.difficulty == 0.5

"""Explicit Wuji curriculum checkpoint provider and PPO runtime selection."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper
from uni_rl.algos.rsl_rl_runtime import RslRlPPORuntime
from uni_rl.algos.rsl_rl_training_state import TrainingStateOnPolicyRunner

from wuji_unilab.tasks.reorient.commands import task


class WujiWrapper(RslRlVecEnvWrapper):
    def export_training_state(self) -> Mapping[str, Any]:
        state = task(self.env)
        return {
            "schema": "wuji_reorient",
            "version": 1,
            "progress": self.env.export_training_state(),
            "difficulty": state.difficulty,
            "adaptive": state.adaptive,
            "obs_groups": dict(self.env.obs_groups_spec),
        }

    def import_training_state(self, value: Mapping[str, Any]) -> None:
        if value.get("schema") != "wuji_reorient" or value.get("version") != 1:
            raise ValueError("Unsupported Wuji training state schema")
        if value.get("obs_groups") != dict(self.env.obs_groups_spec):
            raise ValueError("Wuji checkpoint observation groups do not match the environment")
        difficulty, adaptive = value.get("difficulty"), value.get("adaptive")
        if (
            not isinstance(difficulty, (int, float))
            or not isinstance(adaptive, (int, float))
            or isinstance(difficulty, bool)
            or isinstance(adaptive, bool)
            or not np.isfinite([difficulty, adaptive]).all()
            or not 0 <= difficulty <= 1
            or not 0.05 <= adaptive <= 1
        ):
            raise ValueError("Invalid Wuji curriculum state")
        self.env.import_training_state(value["progress"])
        state = task(self.env)
        state.difficulty, state.adaptive = float(difficulty), float(adaptive)


def resolve_runtime(rl_cfg: dict) -> RslRlPPORuntime:
    del rl_cfg
    return RslRlPPORuntime(wrapper_cls=WujiWrapper, runner_cls=TrainingStateOnPolicyRunner)

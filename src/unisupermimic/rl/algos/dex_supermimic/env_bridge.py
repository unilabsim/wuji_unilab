"""Adapt UniLab's TorchEnvState API to the source learner's tuple boundary."""

from __future__ import annotations

import torch


class SourceEnvBridge:
    def __init__(self, env):
        self.native_env = env
        self.training_reader = None
        self._policy = None
        env.set_autoreset(False)
        env.cfg.auto_reset = False

    def __getattr__(self, name):
        return getattr(self.native_env, name)

    @property
    def common_step_counter(self):
        return self.native_env.common_step_counter

    @common_step_counter.setter
    def common_step_counter(self, value):
        # UniLab derives common_step_counter from its authoritative TorchEnv
        # counter on every step. Restore both through its public checkpoint
        # boundary so a resumed curriculum cannot restart at zero.
        self.native_env.import_training_state({"version": 1, "step_counter": int(value)})

    @property
    def max_common_steps(self):
        return self.native_env.max_common_steps

    @max_common_steps.setter
    def max_common_steps(self, value):
        self.native_env.max_common_steps = value

    @property
    def _adaptive_episode_curriculum(self):
        return self.native_env._adaptive_episode_curriculum

    @_adaptive_episode_curriculum.setter
    def _adaptive_episode_curriculum(self, value):
        self.native_env._adaptive_episode_curriculum = value

    @property
    def _reorient_success_curriculum(self):
        return self.native_env._reorient_success_curriculum

    @_reorient_success_curriculum.setter
    def _reorient_success_curriculum(self, value):
        self.native_env._reorient_success_curriculum = value

    def reset(self, env_ids=None, **kwargs):
        observations, info = self.native_env.reset(env_ids=env_ids, **kwargs)
        reader = self.training_reader
        if reader is not None and hasattr(reader, "policy_observation"):
            reader.refresh_frame()
            ids = (torch.arange(self.num_envs, device=self.device) if env_ids is None
                   else env_ids.to(device=self.device, dtype=torch.long))
            reader.reset_rows(ids)
            policy = reader.policy_observation(ids)
            if self._policy is None:
                self._policy = torch.zeros((self.num_envs, policy.shape[-1]), device=self.device)
            self._policy[ids] = policy
            return {**observations, "policy": self._policy.clone()}, info
        policy = observations.get("policy", observations.get("obs"))
        return {**observations, "policy": policy}, info

    def step(self, actions):
        reader = self.training_reader
        if reader is not None and hasattr(reader, "before_step"):
            reader.before_step()
        state = self.native_env.step(actions)
        observations = dict(state.obs)
        if reader is not None and hasattr(reader, "policy_observation"):
            self._policy = reader.policy_observation()
            observations["policy"] = self._policy.clone()
        else:
            observations["policy"] = observations.get("policy", observations.get("obs"))
        return observations, state.reward, state.terminated, state.truncated, state.info

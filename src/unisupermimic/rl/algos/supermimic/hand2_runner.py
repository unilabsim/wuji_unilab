"""Hand 2 SuperMimic world-model runner.

The G1 learner in :mod:`learner` has a motion-library state contract and a
goal encoder that do not apply to cube reorientation.  This runner keeps
the Hand 2 contract explicit: 58D palm-relative state, 46D simulator context,
100D controller memory, 207D actor history and a 52+32D dynamics/contact
head.  It is device agnostic and supports CPU smoke runs as well as the
CUDA/mjwarp training path.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from unisupermimic.rl.algos.supermimic.geometry import matvec, quat_to_mat
from unisupermimic.rl.algos.supermimic.networks import EmpiricalNorm, RunningNorm, mlp, run_mlp
from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_TRANSITION_KEY
from unisupermimic.tasks.dex_reorient.hand2.mdp import (
    CONTACT_DIM,
    OBS_DIM,
    STATE_DIM,
    Hand2SuperMimicMDP,
)
from unisupermimic.tasks.dex_reorient.hand2.memory import flatten_memory, unflatten_memory

from .hand2_checkpoint import Hand2CSVLogger, load_hand2_checkpoint, save_hand2_checkpoint
from .hand2_failure import Hand2FailureHead
from .hand2_replay import Hand2ReplayBuffer


@dataclass
class Hand2SuperMimicConfig:
    iterations: int = 1000
    num_envs: int = 2
    warmup_steps: int = 64
    wm_window: int = 8
    wm_batches: int = 10
    wm_batch: int = 256
    utd: float = 0.4
    contact_weight: float = 0.1
    horizon: int = 16
    pi_batch: int = 256
    wm_hidden: tuple[int, ...] = (1024, 1024, 1024)
    # Keep the source loss decomposition explicit even though the migration
    # recipe uses its default value of one.
    final_state_weight: float = 1.0
    policy_hidden: tuple[int, ...] = (512, 256, 128)
    # Match the source critic: the imagined privileged frame is high
    # dimensional and its larger trunk is part of the original recipe.
    critic_hidden: tuple[int, ...] = (512, 512, 256, 128)
    lr: float = 2.0e-4
    lr_decay_iters: int = 250_000
    lr_final_ratio: float = 0.03
    gamma: float = 0.99
    action_noise: float = 0.1
    wm_only_iters: int = 100
    critic_epochs: int = 3
    critic_minibatches: int = 4
    lam: float = 0.95
    buffer_steps: int = 1000
    holdout_fraction: float = 0.0
    wm_failure_frac: float = 0.25
    wm_failure_horizon: int = 16
    failure_head: bool = True
    failure_hidden: tuple[int, ...] = (256, 128)
    failure_warmup_iters: int = 300
    failure_clip: float = 0.1
    disturbance_prob_scale: float = 1.0
    disturbance_min_speed: float = 0.05
    disturbance_max_speed: float = 0.15
    disturbance_start_s: float = 3.0
    ckpt_every: int = 1000
    resume: str | None = None
    seed: int = 0
    compile: bool = False
    amp: bool = True
    log_every: int = 10


class Hand2WorldModel(nn.Module):
    """Predict the 52D physical residual and 32 contact logits."""

    def __init__(self, hidden=(1024, 1024, 1024), amp: bool = True):
        super().__init__()
        self.amp = bool(amp)
        self.in_norm = RunningNorm(STATE_DIM + 46 + 9 + 20)
        # Source ``WorldModel.delta_norm`` uses eps=1e-4; the input/failure
        # normalizers intentionally keep the regular 1e-2 epsilon.
        self.out_norm = RunningNorm(52, eps=1.0e-4)
        self.net = mlp(STATE_DIM + 46 + 9 + 20, hidden, 52 + CONTACT_DIM, act=nn.SiLU, layer_norm=True)
        output = self.net[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        output.bias.data[52:] = -3.0
        # Match Dex-SuperMimic's per-coordinate delta floor.  Using a single
        # 1e-4 clamp lets nearly-static velocity/rotation coordinates create
        # enormous normalized errors and is a common source of late actor
        # divergence on long runs.
        self.register_buffer(
            "error_floor",
            torch.tensor(
                [0.003] * 20
                + [0.05] * 20
                + [0.001] * 3
                + [0.005] * 3
                + [0.02] * 3
                + [0.05] * 3,
                dtype=torch.float32,
            ),
        )

    def inputs(self, state, target, context, disturbance):
        return torch.cat((state, context, disturbance, target), -1)

    def forward(self, state, target, context, disturbance):
        features = self.in_norm(self.inputs(state, target, context, disturbance))
        if self.amp and state.is_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                value = self.net(features).float()
        else:
            value = run_mlp(self.net, features)
        delta, contact = value.split((52, CONTACT_DIM), -1)
        delta = delta * self.out_norm.std() + self.out_norm.mean
        return delta, contact


class Hand2Actor(nn.Module):
    def __init__(self, hidden=(512, 256, 128)):
        super().__init__()
        # Dex-SuperMimic's actor uses the same rsl-rl empirical normalizer as
        # the source ``Actor``.  The WM/failure/critic retain RunningNorm;
        # using that different update rule here changes the policy scale as
        # soon as imagination starts.
        self.norm = EmpiricalNorm(OBS_DIM * 3)
        self.net = mlp(OBS_DIM * 3, hidden, 20)
        output = self.net[-1]
        assert isinstance(output, nn.Linear)
        nn.init.orthogonal_(output.weight, gain=0.01)
        nn.init.zeros_(output.bias)

    @staticmethod
    def flatten_history(history: torch.Tensor) -> torch.Tensor:
        """Pack chronological ``[B,3,69]`` frames in source term order."""
        if history.shape[-2:] != (3, OBS_DIM):
            raise ValueError(
                f"Hand2 history must have shape (..., 3, {OBS_DIM}), got {tuple(history.shape)}"
            )
        parts = history.split((20, 20, 3, 6, 20), dim=-1)
        return torch.cat([part.flatten(-2) for part in parts], dim=-1)

    def forward(self, history):
        # Real UniLab observations are already term-major 207D vectors;
        # imagined rollouts keep chronological frames so they can append one
        # frame at a time.  Accept both layouts and normalize only after the
        # source ordering has been restored.
        if history.shape[-2:] == (3, OBS_DIM):
            history = self.flatten_history(history)
        if history.shape[-1] != OBS_DIM * 3:
            raise ValueError(
                f"Hand2 actor input must have width {OBS_DIM * 3}, got {history.shape[-1]}"
            )
        return torch.tanh(run_mlp(self.net, self.norm(history)))


class Hand2Critic(nn.Module):
    def __init__(self, hidden=(256, 128), amp: bool = True):
        super().__init__()
        self.amp = bool(amp)
        self.norm = RunningNorm(211)
        self.net = mlp(211, hidden, 1)

    def forward(self, features):
        normalized = self.norm(features)
        if self.amp and features.is_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                value = self.net(normalized).float()
        else:
            value = run_mlp(self.net, normalized)
        return value.squeeze(-1)


class Hand2SuperMimicRunner:
    """Collect transitions and fit/roll out the Hand 2 SuperMimic model."""

    def __init__(self, env, cfg: Hand2SuperMimicConfig, log_dir: str | Path, device=None):
        self.env = env
        self.cfg = cfg
        # World-model training keeps terminal transitions in replay and
        # performs selected-row resets after publishing each transition.
        # Disable UniLab's PPO-style scattered autoreset so ``s1`` and
        # ``done`` cannot be replaced by a fresh episode before collection.
        set_autoreset = getattr(env, "set_autoreset", None)
        if callable(set_autoreset):
            set_autoreset(False)
        env_cfg = getattr(env, "cfg", None)
        if env_cfg is not None and hasattr(env_cfg, "auto_reset"):
            env_cfg.auto_reset = False
        # The source runner exposes the planned training horizon to the
        # interval disturbance and adaptive episode curriculum.  UniLab's
        # generic env factory does not know the learner's UTD schedule, so
        # install the same carrier before the first reset/collection step.
        self.env.max_common_steps = cfg.warmup_steps + int(cfg.iterations / max(cfg.utd, 1e-6))
        self.env_device = torch.device(env.device)
        self.device = self.env_device if device is None else torch.device(device)
        env_mdp = env.supermimic_mdp
        if not isinstance(env_mdp, Hand2SuperMimicMDP):
            raise TypeError("Hand2 environment did not expose Hand2SuperMimicMDP")
        # MuJoCo is a host-bridge environment: physics and manager tensors can
        # remain on CPU while the learner runs on CUDA. Always materialize a
        # learner-side MDP copy. Some host-bridge backends
        # report ``env.device`` as CUDA while their robot/default tensors used
        # to build the command MDP remain on CPU.  Comparing only the backend
        # device therefore is not sufficient: ``mdp.obs`` combines those
        # constants (joint limits, tag pose, etc.) with replay tensors on the
        # learner device and would fail with a CPU/CUDA mismatch.  ``to`` is a
        # cheap identity when the adapter is already on the right device.
        self.mdp: Hand2SuperMimicMDP = env_mdp.to(self.device)
        if self.mdp.context_dim != 46 or self.mdp.state_dim != STATE_DIM:
            raise ValueError("Unexpected Hand2 SuperMimic state/context dimensions")
        torch.manual_seed(cfg.seed)
        self.wm = Hand2WorldModel(cfg.wm_hidden, amp=cfg.amp).to(self.device)
        self.actor = Hand2Actor(cfg.policy_hidden).to(self.device)
        self.critic = Hand2Critic(cfg.critic_hidden, amp=cfg.amp).to(self.device)
        self.failure = Hand2FailureHead(cfg.failure_hidden).to(self.device)
        self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=cfg.lr)
        # Dex-SuperMimic uses AdamW with the lower second-moment decay for the
        # imagination policy.  Adam's default beta2=0.999 makes the migrated
        # actor substantially more aggressive on long 8k runs and was one of
        # the differences present before the late NaN.
        self.opt_actor = torch.optim.AdamW(
            self.actor.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0
        )
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.opt_failure = torch.optim.Adam(self.failure.parameters(), lr=cfg.lr)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "args.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
        self.replay = Hand2ReplayBuffer(
            cfg.buffer_steps,
            env.num_envs,
            self.device,
            holdout_fraction=cfg.holdout_fraction,
        )
        self.csv_logger = Hand2CSVLogger(self.log_dir / "train_log.csv")
        self.obs_history: torch.Tensor | None = None
        self.obs: torch.Tensor | None = None
        self.iteration = 0
        self.sim_steps = 0
        self._collect_step_acc = 0.0
        self.critic_loss = torch.zeros((), device=self.device)
        self.resumed = False
        if cfg.resume:
            state = load_hand2_checkpoint(
                cfg.resume,
                modules={"wm": self.wm, "actor": self.actor, "critic": self.critic, "failure": self.failure},
                optimizers={"wm": self.opt_wm, "actor": self.opt_actor, "critic": self.opt_critic, "failure": self.opt_failure},
                # Checkpoints are serialized on CPU.  Loading the complete
                # replay payload directly onto CUDA temporarily duplicates
                # its ~3 GiB allocation when it is copied into the live
                # replay buffer; keep the serialized state on host memory and
                # let module/replay loaders transfer only their destination
                # tensors.
                map_location="cpu",
            )
            replay_state = state.get("replay")
            if isinstance(replay_state, dict):
                self.replay.load_state_dict(replay_state)
            self.iteration = int(state.get("iteration", 0))
            self.sim_steps = int(state.get("sim_steps", 0))
            self._collect_step_acc = float(state.get("extra", {}).get("collect_step_acc", 0.0))
            self.obs = state.get("obs")
            self.obs_history = state.get("obs_history")
            if self.obs is not None:
                self.obs = self.obs.to(self.device)
            if self.obs_history is not None:
                self.obs_history = self.obs_history.to(self.device)
            self.resumed = self.replay.time_size > 0 and self.obs is not None and self.obs_history is not None

    @staticmethod
    def _auxiliary(env) -> torch.Tensor:
        command = env.command_manager.get_term("reorient")
        _pos, quat = command.palm_pose()
        from unisupermimic.rl.algos.supermimic.geometry import quat_to_mat

        rot = quat_to_mat(quat).flatten(-2)
        force = getattr(command, "force_direction", rot.new_zeros((rot.shape[0], 3)))
        perturb = getattr(command, "perturbation", rot.new_zeros((rot.shape[0], 6)))
        return torch.cat((_pos, rot, force, perturb), -1)

    def _store(self, transition: dict[str, Any], auxiliary: torch.Tensor) -> None:
        data = {
            "s": transition["s"].detach().to(self.device),
            "s1": transition["s1"].detach().to(self.device),
            # Keep interval-event jumps separate from the post-physics target;
            # imagined windows apply this jump after each learned transition.
            # The replay buffer accepts a zero fallback for older lightweight
            # transitions that did not expose the optional field.
            "push": transition.get(
                "push", transition["s"].new_zeros(transition["s"].shape)
            ).detach().to(self.device),
            "a": transition["a"].detach().to(self.device),
            "a_prev": transition["a_prev"].detach().to(self.device),
            "xi": transition["xi"].detach().to(self.device),
            "m": flatten_memory(transition["m"]).detach().to(self.device),
            "m1": flatten_memory(transition["m1"]).detach().to(self.device),
            # Lightweight/backends created before the encoder-bias extension
            # may omit these optional fields.  The calibrated default is
            # zero bias, so recover the executed target from c1 while keeping
            # the replay schema explicit.
            "u": transition["m1"].get("u", transition["m1"]["c1"]).detach().to(self.device),
            "bias": transition["m1"].get(
                "bias", transition["m1"]["c1"].new_zeros(transition["m1"]["c1"].shape)
            ).detach().to(self.device),
            "aux": transition.get("auxiliary", auxiliary).detach().to(self.device),
            "done": transition["done"].detach().to(self.device),
            "contact": transition.get(
                "contact", transition["s"].new_zeros((transition["s"].shape[0], CONTACT_DIM))
            ).detach().to(self.device),
            "torque": transition.get(
                "torque", transition["s"].new_zeros((transition["s"].shape[0], 20))
            ).detach().to(self.device),
            # The environment policy group is term-major (all three joint
            # frames, then all target-error frames, ...), exactly as the
            # source ObservationManager emits and as the actor normalizer
            # expects.  Keep the packed row instead of flattening chronological
            # frames in the wrong order.
            "obs": self.obs.detach().to(self.device),
            "terminated": transition["term"].detach().to(self.device),
            "timeout": transition["timeout"].detach().to(self.device),
            "ep_step": transition["ep_step"].detach().to(self.device, dtype=torch.long),
        }
        self.replay.add(data)

    @staticmethod
    def _finite_optimizer_step(
        module: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: torch.Tensor,
        *,
        max_grad_norm: float = 1.0,
        max_grad_value: float | None = None,
    ) -> bool:
        """Apply an update only when its loss and gradients are finite."""
        optimizer.zero_grad(set_to_none=True)
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError("Hand2 optimizer received a non-finite loss")
        loss.backward()
        for parameter in module.parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("Hand2 optimizer produced a non-finite gradient")
        norm = nn.utils.clip_grad_norm_(module.parameters(), max_grad_norm)
        if not bool(torch.isfinite(norm).all()):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Hand2 optimizer gradient norm is non-finite")
        if max_grad_value is not None:
            nn.utils.clip_grad_value_(module.parameters(), max_grad_value)
        optimizer.step()
        return True

    @staticmethod
    def _single_frame_observation(value: torch.Tensor) -> torch.Tensor:
        """Convert UniLab's term-major 3-frame output to one 69D frame.

        UniLab's observation manager concatenates each term's history as one
        block (``joint[3] | target_error[3] | position[3] | ...``), whereas
        the learner stores history as chronological complete frames.  A plain
        ``reshape(..., 3, 69)`` mixes the terms and silently changes every
        actor input after the first reset.
        """
        if value.shape[-1] == OBS_DIM:
            return value
        if value.shape[-1] == OBS_DIM * 3:
            parts = value.split((20 * 3, 20 * 3, 3 * 3, 6 * 3, 20 * 3), dim=-1)
            return torch.cat(
                [part.reshape(*part.shape[:-1], 3, width)[:, -1] for part, width in zip(
                    parts, (20, 20, 3, 6, 20), strict=True
                )],
                dim=-1,
            )
        raise ValueError(f"Unexpected Hand2 policy observation width {value.shape[-1]}")

    def _batch(self, count: int) -> dict[str, torch.Tensor]:
        if self.replay.time_size == 0:
            raise RuntimeError("Hand2 replay is empty")
        start, _history = self.replay.sample_starts(count, 1)
        return {name: value for name, value in start.items() if name in self.replay.data}

    def _imagined_disturbance(
        self, state: torch.Tensor, memory: dict[str, torch.Tensor], auxiliary: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the interval velocity disturbance used by the source task.

        The real event term writes a world-frame velocity delta and caches its
        direction/magnitude in the 9D auxiliary tail.  Imagined rollouts use
        the same event rate and start-time gate, while the sampled push is
        applied to the cube linear velocity component of the predicted state.
        """
        batch = state.shape[0]
        event = torch.rand(batch, 1, device=self.device) < (
            self.mdp.dt / 1.2 * self.cfg.disturbance_prob_scale
        )
        active = event & (memory["age"] * self.mdp.dt >= self.cfg.disturbance_start_s)
        direction = F.normalize(torch.randn(batch, 3, device=self.device), dim=-1)
        # Match the source's collection/event schedule: speed ramps from the
        # minimum to the configured maximum over the first 80% of training,
        # then is scaled by the adaptive episode curriculum.  Imagining with
        # the full disturbance range from iteration zero changes the training
        # distribution and can destabilize the actor before the WM is fit.
        total = float(getattr(self.env, "max_common_steps", self.cfg.iterations / max(self.cfg.utd, 1e-6)))
        step = float(getattr(self.env, "common_step_counter", 0))
        warmup = 0.05 * max(total, 1.0)
        rampup = 0.80 * max(total, 1.0)
        if step <= warmup:
            progress = 0.0
        elif step >= rampup:
            progress = 1.0
        else:
            progress = (step - warmup) / max(rampup - warmup, 1.0)
        command = self.env.command_manager.get_term("reorient")
        adaptive = float(getattr(command, "adaptive", 0.05))
        ramp = progress * adaptive
        max_speed = self.cfg.disturbance_min_speed + (
            self.cfg.disturbance_max_speed - self.cfg.disturbance_min_speed
        ) * ramp
        speed = self.cfg.disturbance_min_speed + torch.rand(batch, 1, device=self.device) * (
            max_speed - self.cfg.disturbance_min_speed
        )
        delta = direction * speed * active
        push = state.new_zeros(state.shape)
        push[..., 52:55] = delta
        palm_rotation = auxiliary[..., 3:12].reshape(batch, 3, 3)
        world_direction = matvec(palm_rotation, direction * active)
        world_velocity = matvec(palm_rotation, delta)
        perturbation = torch.cat((world_velocity, delta.new_zeros((batch, 3))), dim=-1)
        # The simulator samples the interval event independently from the
        # minimum-episode gate.  An event that fires during warmup therefore
        # applies no physical push, but it still clears/replaces the cached
        # disturbance metadata with zeros (the source runner deliberately
        # passes ``event`` to ``advance_disturbance`` while ``active`` only
        # gates the velocity delta).  Gating this update by ``active`` leaves
        # stale disturbance context in the imagined WM input and changes the
        # source transition distribution.
        cached_direction = torch.where(event, world_direction, auxiliary[..., 12:15])
        cached_perturbation = torch.where(event, perturbation, auxiliary[..., 15:21])
        next_aux = torch.cat((auxiliary[..., :12], cached_direction, cached_perturbation), dim=-1)
        return push, next_aux

    @staticmethod
    def _advance_replay_disturbance(
        auxiliary: torch.Tensor, push: torch.Tensor
    ) -> torch.Tensor:
        """Carry an observed interval push into the next replay step."""
        delta = push[..., 52:55]
        active = torch.linalg.vector_norm(delta, dim=-1, keepdim=True) > 0
        # Match Dex-SuperMimic's replay path exactly: the captured state jump
        # is carried directly into the cached disturbance slots.  The source
        # reader stores this transition slice in the same tensor coordinate
        # convention as ``push``; applying an additional palm rotation here
        # would change the replay context whenever the wrist is tilted.
        world_velocity = delta
        world_direction = F.normalize(world_velocity, dim=-1)
        cached_direction = torch.where(active, world_direction, auxiliary[..., 12:15])
        cached_perturbation = torch.where(
            active,
            torch.cat((world_velocity, push[..., 55:58].new_zeros((*push.shape[:-1], 3))), dim=-1),
            auxiliary[..., 15:21],
        )
        return torch.cat((auxiliary[..., :12], cached_direction, cached_perturbation), dim=-1)

    @torch.no_grad()
    def collect(self, steps: int) -> dict[str, torch.Tensor]:
        # These diagnostics mirror the source runner's rollout metrics.  They
        # are accumulated from the same transition boundary that is written to
        # replay, so they expose action/target/contact/rotation mismatches
        # without changing the learner's data or optimization path.
        diagnostics: dict[str, list[torch.Tensor]] = {
            "policy_action_abs": [],
            "executed_action_abs": [],
            "action_saturation": [],
            "target_delta_rms": [],
            "rollout_reward": [],
            "rotation_error": [],
            "cube_angvel_rms": [],
            "tip_contact_rate": [],
            "proximal_contact_rate": [],
            "distal_contact_rate": [],
            "self_collision_count": [],
            "goal_reaches": [],
            "termination_rate": [],
            "actor_input_rms": [],
            "actor_normalized_input_rms": [],
        }

        def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            value = value.to(self.device)
            mask = mask.to(device=value.device, dtype=torch.bool).reshape(-1)
            value = value.reshape(mask.shape[0], -1)
            weight = mask.to(value.dtype)[:, None]
            return (value * weight).sum() / weight.sum().clamp_min(1.0) / value.shape[-1]

        def finite_rows(value: torch.Tensor) -> torch.Tensor:
            value = value.to(self.device)
            return torch.isfinite(value).reshape(value.shape[0], -1).all(-1)

        if self.obs is None:
            reset_obs = self.env.reset()[0]
            packed = reset_obs.get("policy", reset_obs.get("obs")).to(self.device)
            if packed.shape[-1] != OBS_DIM * 3:
                raise ValueError(
                    f"Hand2 policy group must have width {OBS_DIM * 3}, got {packed.shape[-1]}"
                )
            self.obs = packed
            frame = self._single_frame_observation(packed)
            self.obs_history = frame[:, None].expand(-1, 3, -1).clone()
        for _ in range(steps):
            assert self.obs_history is not None
            if not bool(torch.isfinite(self.obs_history).all()):
                raise FloatingPointError("Hand2 policy observation history contains NaN or Inf")
            self.actor.norm.update(self.obs)
            policy_action = self.actor(self.obs)
            if not bool(torch.isfinite(policy_action).all()):
                raise FloatingPointError("Hand2 actor produced NaN or Inf action")
            action = (policy_action + self.cfg.action_noise * torch.randn_like(policy_action)).clamp(-1, 1)
            state = self.env.step(action.to(self.env_device))
            info = state.info[SUPERMIMIC_TRANSITION_KEY]
            transition = {name: value for name, value in info.items()}
            auxiliary = transition.get("auxiliary")
            if auxiliary is None:
                auxiliary = self._auxiliary(self.env)
            valid = (
                finite_rows(policy_action)
                & finite_rows(action)
                & finite_rows(transition["s"])
                & finite_rows(transition["s1"])
                & finite_rows(transition["xi"])
                & finite_rows(transition["contact"])
                & finite_rows(transition["torque"])
                & finite_rows(flatten_memory(transition["m"]))
            )
            self._store(transition, auxiliary)
            # Keep output/input statistics tied to every real transition, as
            # in the native SuperMimic learner.  Updating these statistics in
            # ``world_model_update`` after the forward pass makes the first
            # zero-initialized prediction use stale zero-mean/unit-variance
            # statistics, then scores it against the newly tiny batch
            # variance.  With the short ``wm_batch=2`` smoke recipe this can
            # produce an artificially enormous loss (especially when both
            # sampled rows are equal).  The learner-side copies below also
            # ensure the bridge tensors are on the same device as the model.
            state0 = transition["s"].detach().to(self.device)
            state1 = transition["s1"].detach().to(self.device)
            context0 = transition["xi"].detach().to(self.device)
            # The action target for this post-step state is the controller
            # snapshot in ``m1``.  ``m`` is deliberately the pre-action
            # boundary used for actor history and task state.
            target0 = transition["m1"]["u"].detach().to(self.device)
            auxiliary0 = auxiliary.detach().to(self.device)
            disturbance0 = auxiliary0[..., 12:21]
            train_slice = slice(0, self.replay.train_end)
            self.wm.in_norm.update(
                self.wm.inputs(state0, target0, context0, disturbance0)[train_slice]
            )
            self.wm.out_norm.update(self.mdp.world_model_targets(state0, state1)[train_slice])
            if self.cfg.failure_head:
                # FailureHead follows the source contract and is normalized
                # on the complete post-step task context (simulator context
                # plus privileged frame), rather than the 46-D WM context.
                post_memory = unflatten_memory(transition["m1"].detach().to(self.device)) if isinstance(transition["m1"], torch.Tensor) else {
                    key: value.detach().to(self.device) for key, value in transition["m1"].items()
                }
                failure_context = self.mdp.task_features(
                    state1, post_memory, auxiliary0, context0
                )
                self.failure.norm.update(
                    self.failure.features(state1, failure_context)[train_slice]
                )
            observed = self.env.observation_manager.compute(update_history=False)
            next_value = observed.get("policy", observed.get("obs"))
            if next_value is None:
                raise KeyError("Hand2 observation manager did not return policy/obs")
            next_packed = next_value.to(self.device)
            done = transition["done"].to(self.device)
            # Match Dex-SuperMimic's explicit partial reset.  Keep the
            # terminal transition already stored above, then replace only the
            # next action observation for rows whose episode ended.
            if bool(done.any()):
                reset_ids = done.nonzero(as_tuple=False).flatten()
                reset_obs, _ = self.env.reset(env_ids=reset_ids)
                reset_packed = reset_obs.get("policy", reset_obs.get("obs"))
                if reset_packed is None:
                    raise KeyError("Hand2 reset did not return policy/obs observations")
                reset_packed = reset_packed.to(self.device)
                next_packed = torch.where(done[:, None], reset_packed, next_packed)
            if not bool(torch.isfinite(next_packed).all()):
                raise FloatingPointError("Hand2 environment produced NaN or Inf observation")

            # Diagnostics are deliberately computed after the transition has
            # been captured and before any selected-row reset.  The source
            # runner reports the same post-action rotation/contact quantities.
            diagnostics["policy_action_abs"].append(masked_mean(policy_action.abs(), valid))
            diagnostics["executed_action_abs"].append(masked_mean(action.abs(), valid))
            diagnostics["action_saturation"].append(masked_mean((action.abs() >= 0.999).float(), valid))
            target_delta = transition["m1"]["u"] - transition["m"]["c1"]
            diagnostics["target_delta_rms"].append(
                torch.sqrt(masked_mean(target_delta.square(), valid).clamp_min(0.0))
            )
            diagnostics["rollout_reward"].append(masked_mean(transition["r"], valid))
            orientation_error = transition.get("orientation_error")
            if orientation_error is not None:
                orientation_error = orientation_error.to(self.device)
                orientation_mask = valid & torch.isfinite(orientation_error)
                diagnostics["rotation_error"].append(masked_mean(orientation_error, orientation_mask))
            else:
                diagnostics["rotation_error"].append(action.new_zeros(()))
            angular_velocity = transition["s1"][..., 55:58]
            diagnostics["cube_angvel_rms"].append(
                torch.sqrt(masked_mean(angular_velocity.square(), valid).clamp_min(0.0))
            )
            contact = transition["contact"]
            diagnostics["tip_contact_rate"].append(masked_mean(contact[..., :5], valid))
            diagnostics["proximal_contact_rate"].append(masked_mean(contact[..., 5:6], valid))
            diagnostics["distal_contact_rate"].append(masked_mean(contact[..., 6:7], valid))
            diagnostics["self_collision_count"].append(masked_mean(contact[..., 7:].sum(-1), valid))
            diagnostics["goal_reaches"].append(masked_mean(transition.get("success", action.new_zeros(action.shape[0])).float(), valid))
            diagnostics["termination_rate"].append(masked_mean(transition["term"].float(), valid))
            diagnostics["actor_input_rms"].append(torch.sqrt(self.obs.square().mean().clamp_min(0.0)))
            diagnostics["actor_normalized_input_rms"].append(
                torch.sqrt(self.actor.norm(self.obs).square().mean().clamp_min(0.0))
            )
            next_obs = self._single_frame_observation(next_packed)
            shifted = torch.cat((self.obs_history[:, 1:], next_obs[:, None]), 1)
            self.obs_history = torch.where(done[:, None, None], next_obs[:, None].expand_as(shifted), shifted)
            self.obs = next_packed
            self.sim_steps += self.env.num_envs

        return {
            name: torch.stack(values).mean()
            for name, values in diagnostics.items()
            if values
        }

    def world_model_update(self, *, holdout: bool = False) -> torch.Tensor:
        window = min(self.cfg.wm_window, self.replay.time_size)
        batch = self.replay.sample_windows(
            self.cfg.wm_batch,
            window,
            holdout=holdout,
            failure_frac=0.0 if holdout else (self.cfg.wm_failure_frac if self.cfg.failure_head else 0.0),
            failure_horizon=self.cfg.wm_failure_horizon,
        )
        state = batch["s"][:, 0]
        memory = unflatten_memory(batch["m"][:, 0])
        aux = batch["aux"][:, 0]
        losses = []
        state_losses = []
        contact_losses = []
        failure_losses = []
        std = torch.maximum(self.wm.out_norm.std(), self.wm.error_floor)
        for step in range(window):
            # ``m`` is the snapshot at the action boundary, before the
            # action for this transition has been processed.  The target
            # actually applied during that transition is captured in ``m1``
            # after the manager has advanced its EMA controller.  Using
            # ``m.c1`` here trains the WM against the previous action target
            # (one control step stale), unlike the source learner's replay
            # ``u``/``c`` fields.
            target = batch["u"][:, step]
            target_c = target + batch["bias"][:, step]
            context = batch["xi"][:, step]
            raw_delta, contact = self.wm(state, target, context, aux[..., 12:21])
            next_state = self.mdp.integrate_state(state, raw_delta)
            true_error = self.mdp.state_error(next_state, batch["s1"][:, step])
            state_loss = (true_error / std).square().mean(-1)
            contact_loss = F.binary_cross_entropy_with_logits(
                contact, batch["contact"][:, step], reduction="none"
            ).sum(-1)
            if self.cfg.failure_head:
                label = batch["terminated"][:, step].float()
                # The source computes predicted failure after advancing the
                # command memory.  True replay features use the recorded
                # pre-action memory for the independently observed state.
                failure_loss = state_loss.new_zeros(state_loss.shape)
            else:
                failure_loss = state_loss.new_zeros(state_loss.shape)
            state_losses.append(state_loss)
            contact_losses.append(contact_loss)
            failure_losses.append(failure_loss)
            losses.append(
                self.cfg.final_state_weight * state_loss
                + self.cfg.contact_weight * contact_loss
            )
            # Keep the imagined task/controller state in lockstep with the
            # source learner while fitting a multi-step window.  This state
            # is used by the failure head and by the next imagined target;
            # it is deliberately separate from the recorded m1 snapshot.
            _, cage = self.mdp._cage(next_state, memory["cage"], soft=True)
            memory = self.mdp.advance_memory(
                next_state,
                memory,
                batch["a"][:, step],
                target_c,
                cage=cage,
                next_goal=memory["goal"].reshape(*next_state.shape[:-1], 3, 3),
                soft=True,
            )
            if self.cfg.failure_head:
                predicted_context = self.mdp.task_features(
                    next_state.detach(), memory, aux, context
                )
                true_context = self.mdp.task_features(
                    batch["s1"][:, step],
                    unflatten_memory(batch["m"][:, step]),
                    aux,
                    context,
                )
                predicted_failure = self.failure.predict(
                    next_state.detach(), predicted_context.detach()
                )
                true_failure = self.failure.predict(
                    batch["s1"][:, step], true_context.detach()
                )
                failure_loss = 0.5 * (
                    F.binary_cross_entropy_with_logits(predicted_failure, label, reduction="none")
                    + F.binary_cross_entropy_with_logits(true_failure, label, reduction="none")
                )
                failure_losses[-1] = failure_loss
            # Keep external interval disturbances outside the learned
            # physics transition.  The following window step starts from the
            # post-event state, just as source ``x = x1 + w["push"]``.
            push = batch["push"][:, step]
            state = next_state + push
            aux = self._advance_replay_disturbance(aux, push)
        mask = batch["mask"]
        denom = mask.sum().clamp_min(1.0)
        # Match Dex-SuperMimic's objective scaling: the window losses are
        # summed over valid time steps and normalized by the sampled window
        # count.  Per-step diagnostics below use the valid-step denominator.
        loss = (torch.stack(losses, 1) * mask).sum() / mask.shape[0]
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError("Hand2 world-model loss is non-finite")
        if not holdout:
            self._finite_optimizer_step(self.wm, self.opt_wm, loss)
        if not holdout:
            self.last_wm_state_loss = ((torch.stack(state_losses, 1) * mask).sum() / denom).detach()
            self.last_wm_contact_loss = ((torch.stack(contact_losses, 1) * mask).sum() / denom).detach()
            self.last_failure_loss = (
                (torch.stack(failure_losses, 1) * mask * batch["importance"]).sum() / (2.0 * denom)
            ).detach()
        if self.cfg.failure_head and not holdout:
            # Two equally weighted BCE terms are fitted for the recorded and
            # imagined features.  Keep the source's factor of one half so the
            # failure head has the same gradient scale as Dex-SuperMimic.
            failure_loss = (torch.stack(failure_losses, 1) * mask * batch["importance"]).sum() / (2.0 * denom)
            self._finite_optimizer_step(self.failure, self.opt_failure, failure_loss)
        return loss.detach()

    def policy_update(self) -> torch.Tensor:
        # The source actor samples every finite replay row, including the
        # held-out environments used only to evaluate WM generalization.
        batch, history_batch = self.replay.sample_starts(
            self.cfg.pi_batch, 3, all_envs=True
        )
        state = batch["s"]
        memory = unflatten_memory(batch["m"])
        encoder_bias = batch["bias"]
        aux = batch["aux"]
        context = batch["xi"]
        # Each replay row stores the complete 207D actor history at the
        # pre-action boundary.  ``sample_starts`` also returns older rows for
        # generic callers, but the packed row is the authoritative history
        # for Hand2 and avoids stacking three already-packed histories.
        del history_batch
        packed = batch["obs"]
        parts = packed.split((20 * 3, 20 * 3, 3 * 3, 6 * 3, 20 * 3), dim=-1)
        history = torch.cat(
            [part.reshape(*part.shape[:-1], 3, width) for part, width in zip(
                parts, (20, 20, 3, 6, 20), strict=True
            )],
            dim=-1,
        )
        objective = state.new_zeros(state.shape[0])
        survival = state.new_ones(state.shape[0])
        critic_inputs = []
        rewards = []
        dones = []
        for parameter in self.wm.parameters():
            parameter.requires_grad_(False)
        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        if self.cfg.failure_head:
            for parameter in self.failure.parameters():
                parameter.requires_grad_(False)
        for step in range(self.cfg.horizon):
            # Intermediate critic features are used only as detached lambda
            # return inputs after the actor update.  Keeping their tensors
            # attached to each imagined WM/action graph multiplies the peak
            # memory by the horizon without contributing to the actor loss;
            # retain the final feature below attached for the bootstrap
            # gradient, then detach it after that update.
            critic_inputs.append(self.mdp.critic_features(state, memory, aux, context).detach())
            action = self.actor(history)
            if not bool(torch.isfinite(action).all()):
                raise FloatingPointError("Hand2 imagined actor produced NaN or Inf action")
            action = action.clamp(-1.0, 1.0)
            target, actuator_target = self.mdp.control_target(
                action,
                memory["c1"],
                age=memory["age"],
                bias=encoder_bias,
            )
            raw_delta, contact = self.wm(state, actuator_target, context, aux[..., 12:21])
            next_state = self.mdp.integrate_state(state, raw_delta)
            # A non-finite imagined state cannot yield a useful actor or
            # critic gradient.  The finite optimizer guard below skips the
            # update, while collection remains protected by action sanitizing.
            if not bool(torch.isfinite(next_state).all()):
                raise FloatingPointError("Hand2 world-model imagination produced NaN or Inf state")
            reward = self.mdp.reward(
                next_state,
                memory,
                action,
                target,
                contact.sigmoid(),
                self.mdp.pd_torque(state, actuator_target, context),
                auxiliary=aux,
            )
            done = self.mdp.soft_termination(next_state, memory).squeeze(-1)
            _, cage = self.mdp._cage(next_state, memory["cage"], soft=True)
            next_goal = quat_to_mat(
                F.normalize(torch.randn(state.shape[0], 4, device=self.device), dim=-1)
            )
            memory = self.mdp.advance_memory(
                next_state,
                memory,
                action,
                target,
                cage=cage,
                next_goal=next_goal,
                soft=True,
            )
            if self.cfg.failure_head and self.iteration >= self.cfg.wm_only_iters + self.cfg.failure_warmup_iters:
                # As in the source runner, the failure head sees the task
                # context after command/controller memory has advanced for
                # this imagined transition.
                failure_context = self.mdp.task_features(
                    next_state, memory, aux, context
                )
                failure_prob = self.failure.predict(
                    next_state, failure_context
                ).sigmoid().clamp(max=self.cfg.failure_clip)
                done = 1.0 - (1.0 - done) * (1.0 - failure_prob)
            # Apply every termination source before discounting this reward.
            # The source runner updates ``d`` with the learned failure head
            # and only then advances the return/survival recursion.  Updating
            # survival above would make the failure head a logged auxiliary
            # prediction with no effect on the actor objective.
            objective = objective + (self.cfg.gamma**step) * survival * reward
            survival = survival * (1.0 - done)
            # The actor objective above already owns the autograd graph for
            # this reward/termination pair.  Critic lambda returns are
            # targets and must not keep a second reference to that graph
            # alive across the actor update; the source runner stores these
            # tensors detached for the same reason.
            rewards.append(reward.detach())
            dones.append(done.detach())
            push, next_aux = self._imagined_disturbance(next_state, memory, aux)
            state = next_state + push
            aux = next_aux
            history = torch.cat((history[:, 1:], self.mdp.add_noise(self.mdp.obs(state, memory, aux))[:, None]), 1)
        last_critic_input = self.mdp.critic_features(state, memory, aux, context)
        bootstrap = self.critic(last_critic_input)
        objective = objective + (self.cfg.gamma**self.cfg.horizon) * survival * bootstrap
        loss = -objective.mean()
        actor_ok = self._finite_optimizer_step(
            self.actor, self.opt_actor, loss, max_grad_value=0.1
        )
        if not actor_ok:
            raise RuntimeError("Hand2 actor update did not run")
        for parameter in self.wm.parameters():
            parameter.requires_grad_(True)
        for parameter in self.critic.parameters():
            parameter.requires_grad_(True)
        if self.cfg.failure_head:
            for parameter in self.failure.parameters():
                parameter.requires_grad_(True)
        # Dex-SuperMimic updates critic normalization after the actor loss has
        # been evaluated and optimized.  Updating it before bootstrap changes
        # the actor objective exactly when wm_only_iters ends and can create a
        # discontinuous value scale at iteration 500.
        self.critic.norm.update(
            torch.cat([item.detach() for item in critic_inputs] + [last_critic_input.detach()], 0)
        )
        with torch.no_grad():
            # Re-evaluate the complete horizon plus bootstrap frame after the
            # running-statistics update.  The source computes lambda-returns
            # from this post-update value vector; using the pre-update
            # ``bootstrap`` scalar here creates a stale final target whenever
            # critic normalization changes (especially at policy warmup end).
            value_inputs = torch.stack(
                [*critic_inputs, last_critic_input], 0
            ).detach()
            values = self.critic(value_inputs)
            returns = []
            ret = values[-1]
            for index in range(self.cfg.horizon - 1, -1, -1):
                ret = rewards[index].detach() + self.cfg.gamma * (1.0 - dones[index]) * (
                    (1.0 - self.cfg.lam) * values[index + 1] + self.cfg.lam * ret
                )
                returns.append(ret)
            returns = torch.stack(returns[::-1], 0)
        critic_x = torch.stack(critic_inputs, 0).detach().reshape(-1, 211)
        critic_y = returns.detach().reshape(-1)
        value_loss = torch.zeros((), device=self.device)
        critic_finite = True
        # Match the source runner's fixed-size critic minibatches.  It uses
        # floor division and deliberately drops a remainder, rather than
        # ``Tensor.chunk`` which silently creates uneven batches and changes
        # the optimizer weighting when the imagined count is not divisible.
        minibatches = max(1, min(self.cfg.critic_minibatches, critic_x.shape[0]))
        minibatch = max(1, critic_x.shape[0] // minibatches)
        for parameter in self.critic.parameters():
            parameter.requires_grad_(True)
        for _ in range(self.cfg.critic_epochs):
            permutation = torch.randperm(critic_x.shape[0], device=self.device)
            for index in range(minibatches):
                indices = permutation[index * minibatch : (index + 1) * minibatch]
                value_loss = F.mse_loss(self.critic(critic_x[indices]), critic_y[indices])
                if not self._finite_optimizer_step(self.critic, self.opt_critic, value_loss):
                    critic_finite = False
        self.critic_loss = value_loss.detach() if critic_finite and bool(torch.isfinite(value_loss)) else torch.zeros((), device=self.device)
        return loss.detach()

    def save(self, path: str | Path, next_iteration: int | None = None) -> Path:
        """Save the Hand2 learner, normalizers, replay and optimizer state."""
        return save_hand2_checkpoint(
            path,
            cfg=self.cfg,
            modules={"wm": self.wm, "actor": self.actor, "critic": self.critic, "failure": self.failure},
            optimizers={"wm": self.opt_wm, "actor": self.opt_actor, "critic": self.opt_critic, "failure": self.opt_failure},
            iteration=self.iteration if next_iteration is None else next_iteration,
            sim_steps=self.sim_steps,
            replay=self.replay,
            obs=self.obs,
            obs_history=self.obs_history,
            extra={"collect_step_acc": self._collect_step_acc},
        )

    def train(self) -> dict[str, float | int | bool]:
        start = time.perf_counter()
        if not self.resumed:
            self.collect(max(self.cfg.warmup_steps, self.cfg.wm_window))
        losses = []
        for iteration in range(self.iteration, self.cfg.iterations):
            self.iteration = iteration
            # Keep the fractional UTD schedule instead of truncating 1/utd on
            # every iteration (for utd=0.4 the source alternates 2 and 3
            # collection steps, averaging 2.5).  Truncating to two silently
            # changes the replay distribution and total simulated steps.
            self._collect_step_acc += 1.0 / max(self.cfg.utd, 1e-6)
            collect_steps = int(self._collect_step_acc)
            self._collect_step_acc -= collect_steps
            collect_stats = self.collect(max(1, collect_steps))
            progress = min(iteration / max(self.cfg.lr_decay_iters, 1), 1.0)
            lr = self.cfg.lr * (
                self.cfg.lr_final_ratio
                + (1.0 - self.cfg.lr_final_ratio)
                * 0.5
                * (1.0 + math.cos(math.pi * progress))
            )
            for optimizer in (
                self.opt_wm,
                self.opt_actor,
                self.opt_critic,
                self.opt_failure,
            ):
                for group in optimizer.param_groups:
                    group["lr"] = lr
            wm_loss = torch.zeros((), device=self.device)
            for _ in range(self.cfg.wm_batches):
                wm_loss = self.world_model_update()
            losses.append(wm_loss)
            holdout_loss = torch.zeros((), device=self.device)
            have_holdout = self.replay.train_end < self.replay.num_envs and iteration % self.cfg.log_every == 0
            if have_holdout:
                holdout_loss = self.world_model_update(holdout=True)
            pi_loss = torch.zeros((), device=self.device)
            if iteration >= self.cfg.wm_only_iters:
                pi_loss = self.policy_update()
                # The imagination graph is much larger than the real-step
                # collector graph (pi_batch x horizon x WM width).  Once the
                # actor/critic update returns, its temporaries are dead, but
                # PyTorch's CUDA caching allocator otherwise keeps their
                # blocks reserved.  mjwarp allocates its CUDA graph buffers
                # through a separate allocator on the next env.step; release
                # unused PyTorch blocks at this phase boundary so the backend
                # does not report an apparent OOM despite no live learner
                # graph remaining.  This does not alter tensors or training
                # semantics and is a no-op on CPU/MPS.
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            self.csv_logger.log(
                {
                    "iteration": iteration,
                    "wm_loss": wm_loss,
                    "wm_state_loss": getattr(self, "last_wm_state_loss", wm_loss),
                    "wm_contact_loss": getattr(self, "last_wm_contact_loss", wm_loss * 0),
                    "failure_loss": getattr(self, "last_failure_loss", wm_loss * 0),
                    "holdout_wm_loss": holdout_loss if have_holdout else "",
                    "pi_loss": pi_loss,
                    "critic_loss": self.critic_loss,
                    "sim_steps": self.sim_steps,
                    **collect_stats,
                }
            )
            if iteration % self.cfg.log_every == 0:
                holdout_text = f" holdout_wm_loss={float(holdout_loss):.6g}" if have_holdout else ""
                print(f"iteration={iteration} wm_loss={float(wm_loss):.6g} pi_loss={float(pi_loss):.6g}{holdout_text}", flush=True)
            if self.cfg.ckpt_every > 0 and (iteration + 1) % self.cfg.ckpt_every == 0:
                self.save(self.log_dir / f"model_{iteration + 1}.pt", next_iteration=iteration + 1)
                self.save(self.log_dir / "last.pt", next_iteration=iteration + 1)
        self.save(self.log_dir / "last.pt", next_iteration=self.cfg.iterations)
        self.save(self.log_dir / "model_final.pt", next_iteration=self.cfg.iterations)
        self.csv_logger.close()
        return {
            "completed": True,
            "iteration": self.cfg.iterations,
            "sim_steps": self.sim_steps,
            "wall_time": time.perf_counter() - start,
            "wm_loss": float(torch.stack(losses).mean()) if losses else 0.0,
            "critic_loss": float(self.critic_loss),
            "wm_state_loss": float(getattr(self, "last_wm_state_loss", torch.zeros((), device=self.device))),
            "wm_contact_loss": float(getattr(self, "last_wm_contact_loss", torch.zeros((), device=self.device))),
            "failure_loss": float(getattr(self, "last_failure_loss", torch.zeros((), device=self.device))),
        }


__all__ = ["Hand2SuperMimicConfig", "Hand2SuperMimicRunner", "Hand2WorldModel"]

"""Collect -> multi-step dynamics/drop fit -> differentiable imagination -> lambda-return critic.

The environment retains the original reward/termination timing and domain randomization.
Physical failure is learned separately; timeouts end replay sequences.
Invalid numerical values stop with diagnostics before entering replay.
"""

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .buffer import ReplayBuffer
from .common import RunningNorm, require_finite
from .mdp import angle_error
from .metrics import MetricLogger
from .networks import Actor, Critic, FailureHead, GaussianPPOActor, WorldModel
from .observations import BASE_TERMS, CURRENT_TERMS, DR_PARAM_NAMES
from .rotations import mv, quat_to_mat
from .state import difference, ema_target, flatten_memory, unflatten_memory, unpack
from .task import checkpoint_task

WANDB_TASK_LABELS = {
    "hand1-reorient": "hand1 reori",
    "hand2-reorient": "hand2 reori",
    "hand2-pen-spin": "pen spin",
}


LOG_SECTION_RENAMES = {
    "wm_loss": "wm/wm_loss",
    "reward": "wm/reward",
    "reward_parity": "wm/reward_parity",
    "failure_loss": "wm/failure_loss",
    "state_loss": "wm/state_loss",
    "survival": "wm/survival",
    "contact_digit_loss": "wm/contact_digit_loss",
    "actor_loss": "policy/actor_loss",
    "critic_loss": "policy/critic_loss",
    "entropy": "policy/entropy",
    "clip_fraction": "policy/clip_fraction",
    "kl": "policy/kl",
    "explained_variance": "policy/explained_variance",
    "drop_rate": "task/drop_rate",
    "rot_error": "task/rot_error",
    "finger_1_action": "task/finger_1_action",
    "finger_2_action": "task/finger_2_action",
    "finger_3_action": "task/finger_3_action",
    "finger_4_action": "task/finger_4_action",
    "finger_5_action": "task/finger_5_action",
    "horizon": "training/horizon",
    "lr": "training/lr",
    "wall_seconds": "training/time",
    "sim_steps": "training/collected_sim_steps",
    "norm_count": "training/norm_count",
    "delta_norm_count": "training/delta_norm_count",
    "supervision_window": "training/supervision_window",
}


def organize_log_scalars(scalars, iteration):
    """Group scalar history keys into the W&B sections used for training runs."""
    organized = {}
    for name, value in scalars.items():
        if torch.is_tensor(value) and value.numel() != 1:
            continue
        if name.startswith("reward_parity/"):
            name = "reward/" + name.split("/", 1)[1]
        elif name.startswith("holdout/"):
            name = "wm/holdout_" + name.split("/", 1)[1]
        else:
            name = LOG_SECTION_RENAMES.get(name, name)
        organized[name] = float(value)
    organized["training/iteration"] = iteration
    return organized


def wandb_run_name(task: str, log_dir: str | Path) -> str:
    """Build a concise W&B run name that starts with the task label."""
    label = WANDB_TASK_LABELS.get(task, task)
    return f"{label} {Path(log_dir).name}"


def format_training_log(iteration, max_iterations, stats):
    """Format console metrics without changing TensorBoard scalar precision."""
    elapsed = int(stats["training/time"])
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    lines = [
        "",
        "=" * 72,
        f"Iteration {iteration:,} / {max_iterations:,}"
        f"  |  Time {hours:02d}:{minutes:02d}:{seconds:02d}"
        f"  |  Steps {stats['training/collected_sim_steps']:,.0f}",
        "-" * 72,
    ]
    groups = (
        (
            "Rollout",
            (
                "wm/reward",
                "wm/reward_parity",
                "task/drop_rate",
                "task/rot_error",
            ),
        ),
        (
            "Policy",
            (
                "policy/actor_loss",
                "policy/critic_loss",
                "wm/survival",
                "training/horizon",
            ),
        ),
    )
    for title, keys in groups:
        values = [f"{key}={stats[key]:.6g}" for key in keys if key in stats]
        if values:
            lines.append(f"{title}:")
            for offset in range(0, len(values), 2):
                lines.append("  " + "  |  ".join(values[offset : offset + 2]))
    lines.extend(
        [
            "World model:",
            f"  {'Metric':<20} {'Train':>16} {'Holdout':>16}",
        ]
    )
    for key in (
        "wm_loss",
        "failure_loss",
        "state_loss",
        "contact_digit_loss",
    ):
        train = f"{stats['wm/' + key]:.6g}" if "wm/" + key in stats else "--"
        holdout_key = "wm/holdout_" + key
        holdout = f"{stats[holdout_key]:.6g}" if holdout_key in stats else "--"
        lines.append(f"  {key:<20} {train:>16} {holdout:>16}")
    lines.append("=" * 72)
    return "\n".join(lines)


class DexSuperMimicRunner:
    def __init__(self, env, cfg, log_dir=None):
        if cfg.algorithm != "world_model":
            raise ValueError("DexSuperMimicRunner requires algorithm=world_model")
        self.env = env
        self.cfg = cfg
        self.device = env.device
        if torch.distributed.is_initialized():
            raise ValueError("DexSuperMimic currently supports one GPU/process")
        self.obs = env.reset()[0]["policy"]
        self.is_pen = cfg.task == "hand2-pen-spin"
        self.flatten_memory = flatten_memory
        self.unflatten_memory = unflatten_memory
        if self.is_pen:
            from .pen import PenObservations, PenSimReader, flatten_pen_memory, unflatten_pen_memory
            self.flatten_memory = flatten_pen_memory
            self.unflatten_memory = unflatten_pen_memory
            self.history_from_obs = lambda obs: obs[:, None]
            self.reader = PenSimReader(env)
        else:
            from .sim import SimReader

            self.reader = SimReader(env)
        self.mdp = self.reader.mdp
        if self.is_pen:
            self.observations = PenObservations(self.reader)
            self.obs = self.reader.policy_observation()
        else:
            from .hand1_reader import Hand1Observations

            self.observations = Hand1Observations(self.reader)
        self.ppo_memory = None
        self.ppo_critic = None
        initial_context = self.observations.context(
            self.reader.state(), self.reader.memory(), self.observations.auxiliary(), self.reader.xi
        )
        initial_dynamics = self.observations.dynamics(
            self.reader.state(), self.reader.xi, self.observations.disturbance(self.observations.auxiliary())
        )
        self.reader.attach_capture()
        ppo_enabled = cfg.algorithm == "ppo"
        action_dim = len(self.reader.robot.joint_names)
        self.actor = (
            GaussianPPOActor(
                self.obs.shape[-1],
                action_dim,
                init_std=cfg.ppo_init_std,
                std_range=(cfg.ppo_std_min, cfg.ppo_std_max),
            ).to(self.device)
            if ppo_enabled
            else Actor(self.obs.shape[-1]).to(self.device)
        )
        contact_dim = self.reader.contact().shape[-1]
        dynamics_dim = initial_dynamics.shape[-1]
        self.dynamics_dim = dynamics_dim
        context_dim = self.observations.context_dim
        self.context_dim = context_dim
        self.contact_dim = contact_dim
        self.wm = WorldModel(dynamics_dim, contact_dim, cfg.wm_hidden, cfg.amp).to(self.device)
        self.fail_head = FailureHead(context_dim).to(self.device)
        self.torque_norm = RunningNorm(20, eps=1e-3).to(self.device)
        self.critic = Critic(self.observations.critic(initial_context).shape[-1], cfg.amp).to(self.device)
        self.finger_action_ids = torch.stack(
            [
                torch.tensor(
                    [
                        i
                        for i, name in enumerate(self.reader.robot.joint_names)
                        if name.endswith(f"finger{finger}_joint{joint}")
                    ],
                    device=self.device,
                )
                for finger in range(1, 6)
                for joint in range(1, 5)
            ],
            -1,
        ).reshape(5, 4)
        self.optimizers = {
            name: torch.optim.Adam(module.parameters(), lr=cfg.lr) for name, module in self.modules().items()
        }
        if ppo_enabled:
            # Upstream Wuji-MJLab uses one Adam over actor+critic with a fixed LR.
            # The real critic group is available only after the env managers exist,
            # so construct it here from the current (pre-action) critic frame.
            initial_critic_obs = env.observation_manager.compute(update_history=False)["critic"]
            self.ppo_critic = Critic(initial_critic_obs.shape[-1], empirical_norm=True).to(self.device)
            self.optimizers = {
                "ppo": torch.optim.Adam(
                    list(self.actor.parameters()) + list(self.ppo_critic.parameters()), lr=cfg.ppo_lr
                )
            }
        else:
            self.optimizers["actor"] = torch.optim.AdamW(
                self.actor.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0
            )
        self.cfg = cfg
        self.buffer = ReplayBuffer(cfg.buffer_steps, env.num_envs, self.device, cfg.holdout_fraction)
        self.ppo_memory = None
        self.log_dir = Path(log_dir or cfg.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if cfg.wandb_mode:
            os.environ["WANDB_MODE"] = cfg.wandb_mode
        self.writer = MetricLogger(
            cfg.logger,
            log_dir=self.log_dir,
            config=cfg.to_dict(),
            run_name=wandb_run_name(cfg.task, self.log_dir),
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
        )
        self.iteration = 0
        self.sim_steps = 0
        self.wall_offset = 0.0
        self.started_at = time.perf_counter()
        if cfg.algorithm == "ppo":
            # Curriculum uses the real horizon: rollout_steps per PPO update.
            self.env.cfg.auto_reset = True
            self.env.max_common_steps = cfg.warmup_steps + cfg.max_iterations * cfg.ppo_rollout_steps
        else:
            self.env.cfg.auto_reset = False
            self.env.max_common_steps = cfg.warmup_steps + int(cfg.max_iterations / cfg.utd)
        self.step_wm = torch.compile(self.wm) if cfg.compile and str(self.device).startswith("cuda") else self.wm
        self.step_actor = (
            torch.compile(self.actor) if cfg.compile and str(self.device).startswith("cuda") else self.actor
        )
        if cfg.algorithm == "ppo":
            # The deterministic inference path is the Gaussian mean, not a sample.
            self.step_actor = self.actor.mean
        self.reward_step = self.mdp.rewards
        self.task_features = self.observations.context
        self.wm_features = self.observations.dynamics
        if cfg.compile_policy and str(self.device).startswith("cuda"):
            options = {"dynamic": cfg.compile_policy_dynamic} if cfg.compile_policy_dynamic else {}
            self.reward_step = torch.compile(self.mdp.rewards, **options)
            self.task_features = torch.compile(self.observations.context, **options)
            self.wm_features = torch.compile(self.observations.dynamics, **options)
        if cfg.checkpoint:
            self.load(cfg.checkpoint)
        if cfg.model_checkpoint:
            self.load_model_only(cfg.model_checkpoint)
        (self.log_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
        # Preserve the entire task configuration, including callable identities and DR.
        import dataclasses

        def serialize(value):
            if dataclasses.is_dataclass(value):
                return {f.name: serialize(getattr(value, f.name)) for f in dataclasses.fields(value)}
            if isinstance(value, dict):
                return {str(k): serialize(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [serialize(v) for v in value]
            if callable(value):
                return f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', repr(value))}"
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return repr(value)

        (self.log_dir / "environment_config.json").write_text(json.dumps(serialize(env.cfg), indent=2))
        import cloudpickle

        (self.log_dir / "environment_config.pkl").write_bytes(cloudpickle.dumps(env.cfg))
        (self.log_dir / "context.json").write_text(json.dumps(self.reader.context_names, indent=2))
        layout = {
            "base_terms": self.observations.base_terms if self.is_pen else BASE_TERMS,
            "current_terms": {} if self.is_pen else CURRENT_TERMS,
            "wm_history": 1,
            "critic_history": 1,
            "wm_input_dim": self.wm.norm.mean.numel(),
            "critic_input_dim": self.critic.norm.mean.numel(),
            "dynamics_input_dim": dynamics_dim,
            "dr_params": [] if self.is_pen else list(DR_PARAM_NAMES),
            "wm_extra_terms": {"control_target": 20},
            "simulator_context": self.reader.context_names,
        }
        (self.log_dir / "input_layout.json").write_text(json.dumps(layout, indent=2))

    def format_version(self):
        return 4 if self.cfg.algorithm == "ppo" else 3

    def _ensure_ppo_critic(self, observation):
        """PPO consumes the environment's real critic group, not imagined features."""
        if observation.shape[-1] != self.ppo_critic.norm._mean.shape[-1]:
            raise ValueError("PPO critic observation dimension changed during rollout")
        return self.ppo_critic

    def modules(self):
        if self.cfg.algorithm == "ppo" and self.ppo_critic is not None:
            return dict(actor=self.actor, critic=self.ppo_critic)
        return dict(actor=self.actor, wm=self.wm, fail_head=self.fail_head, critic=self.critic)

    def critic_features(self, context):
        return self.observations.critic(context)

    def replay_dynamics(self, w, x, j):
        """Dynamics features from predicted state; only startup/exogenous context is replayed."""
        simulator = w["sim_context"] if "sim_context" in w else self.reader.xi[w["env"]]
        return self.wm_features(x, simulator, self.observations.disturbance(w["aux"][:, j]))

    def advance_memory(self, x1, memory, action, target, cage, next_goal, soft=True):
        """Advance task/controller state entirely from the imagined transition."""
        updated = self.mdp.command_step(x1, memory, next_goal, soft=soft)
        updated.update(
            c1=target,
            c2=memory["c1"],
            a1=action,
            a2=memory["a1"],
            cage=cage,
            age=memory["age"] + 1,
        )
        return updated

    @staticmethod
    def advance_disturbance(auxiliary, event, world_direction, world_velocity):
        """Replay or sample the task's exogenous interval-disturbance cache."""
        return torch.cat(
            (
                auxiliary[:, :12],
                torch.where(event, world_direction, auxiliary[:, 12:15]),
                torch.where(
                    event,
                    torch.cat((world_velocity, torch.zeros_like(world_velocity)), -1),
                    auxiliary[:, 15:21],
                ),
                auxiliary[:, 21:],
            ),
            -1,
        )

    @staticmethod
    def history_from_obs(obs):
        return torch.cat(
            [v.reshape(*v.shape[:-1], 3, d) for v, d in zip(obs.split((60, 60, 9, 18, 60), -1), (20, 20, 3, 6, 20), strict=False)],
            -1,
        )

    @torch.no_grad()
    def collect(self, steps):
        gaps = []
        term_gaps = {}
        rewards = []
        drops = []
        rotation_errors = []
        finger_actions = []
        episode_metrics = {}
        for _ in range(steps):
            self.reader.refresh_frame()
            x = self.reader.state()
            m = self.reader.memory()
            auxiliary = self.observations.auxiliary()
            context = self.observations.context(x, m, auxiliary, self.reader.xi)
            require_finite("collect policy observation", self.obs)
            self.actor.norm.update(self.obs)
            action = (self.actor(self.obs) + self.cfg.action_noise * torch.randn_like(m["a1"])).clamp(-1, 1)
            require_finite("collect policy action", action)
            obs, reward, terminated, truncated, _ = self.env.step(action)
            cap = self.reader.reward_snapshot
            c = self.env.action_manager.get_term("joint_pos").processed_action.clone()
            u = self.reader.executed_target() if self.is_pen else c - self.reader.robot.data.encoder_bias
            next_x = self.reader.state()
            valid = (
                (~cap["nan"])
                & torch.isfinite(x).all(-1)
                & torch.isfinite(cap["physics"]).all(-1)
            )
            valid = valid & torch.isfinite(cap["contact"]).all(-1) & torch.isfinite(cap["torque"]).all(-1)
            valid &= torch.isfinite(u).all(-1) & torch.isfinite(c).all(-1)
            valid &= torch.isfinite(action).all(-1) & torch.isfinite(self.flatten_memory(m)).all(-1)
            valid &= torch.isfinite(next_x).all(-1) & torch.isfinite(self.obs).all(-1)
            valid &= torch.isfinite(context).all(-1)
            for name, value in {
                "state": x,
                "physics": cap["physics"],
                "contact": cap["contact"],
                "torque": cap["torque"],
                "target": u,
                "processed_action": c,
                "memory": self.flatten_memory(m),
                "next_state": next_x,
                "next_policy_observation": obs["policy"],
                "context": context,
            }.items():
                require_finite(f"collect {name}", value)
            if cap["nan"].any():
                raise FloatingPointError("Simulator reported numerical instability during collect")
            # Held-out environments never contribute to training statistics.
            train_valid = valid.clone()
            train_valid[self.buffer.train_end :] = False
            self.torque_norm.update(cap["torque"][train_valid])
            if self.cfg.variant == "fixed":
                dynamics = self.observations.dynamics(
                    x[train_valid], self.reader.xi[train_valid], self.observations.disturbance(auxiliary[train_valid])
                )
                self.wm.norm.update(self.wm.inputs(dynamics, u[train_valid]))
                self.wm.delta_norm.update(difference(x[train_valid], cap["physics"][train_valid]))
                with torch.no_grad():
                    selected_m = {key: value[train_valid] for key, value in m.items()}
                    _, next_cage, _ = self.mdp.rewards(
                        cap["physics"][train_valid],
                        selected_m,
                        action[train_valid],
                        c[train_valid],
                        cap["contact"][train_valid],
                        cap["torque"][train_valid],
                    )
                    next_memory = self.advance_memory(
                        cap["physics"][train_valid],
                        selected_m,
                        action[train_valid],
                        c[train_valid],
                        next_cage,
                        m["goal"][train_valid].reshape(-1, 3, 3),
                        soft=False,
                    )
                    next_context = self.observations.context(
                        cap["physics"][train_valid],
                        next_memory,
                        auxiliary[train_valid],
                        self.reader.xi[train_valid],
                    )
                self.fail_head.norm.update(
                    self.fail_head.features(cap["physics"][train_valid], next_context)
                )
            done = terminated | truncated
            self.buffer.add(
                dict(
                    x=x,
                    x1=cap["physics"],
                    push=next_x - cap["physics"],
                    u=u,
                    c=c,
                    action=action,
                    m=self.flatten_memory(m),
                    obs=self.obs,
                    aux=auxiliary,
                    contact=cap["contact"],
                    torque=cap["torque"],
                    reward=reward,
                    done=done,
                    terminated=terminated,
                    valid=valid,
                )
            )
            # Match original SuperMimic: reward uses the same post-step state
            # that the world model predicts.
            an, _, analytic_terms = self.mdp.rewards(
                cap["physics"], m, action, c, cap["contact"], cap["torque"]
            )
            denom = valid.float().sum().clamp_min(1)
            rotation_errors.append(
                torch.where(
                    valid,
                    angle_error(unpack(cap["physics"])[3], m["goal"].reshape(-1, 3, 3)),
                    0.0,
                ).sum()
                / denom
            )
            finger_actions.append(
                torch.where(valid[:, None], action, 0.0)[:, self.finger_action_ids].mean(0)
            )
            for name, value in analytic_terms.items():
                error = (value - cap["terms"][name]).abs()
                term_gaps.setdefault(name, []).append(torch.where(valid, error, 0.0).sum() / denom)
            gaps.append(torch.where(valid, (an - cap["reward"]).abs(), 0.0).sum() / denom)
            rewards.append((reward * valid).sum() / denom)
            drops.append(cap["drop"].float().mean())
            self.obs = obs["policy"]
            self.reader.refresh_frame()
            ids = done.nonzero(as_tuple=False).flatten()
            if ids.numel():
                post, reset_info = self.env.reset(env_ids=ids)
                # UniLab publishes source-native metrics at reset. Each
                # value already averages within an episode, then across
                # these reset rows. Weight reset batches by their row
                # counts so every completed episode has equal weight.
                # Preserve the native reductions for Hand1 as for Pen Spin;
                # do not turn an averaged counter into a new success rate.
                count = ids.numel()
                for name, value in reset_info.get("log", {}).items():
                    if name.startswith("Episode_Metrics/"):
                        key = "task/episode_" + name.split("/", 1)[1]
                        total, weight = episode_metrics.get(key, (0.0, 0))
                        episode_metrics[key] = (total + float(value) * count, weight + count)
                self.obs = torch.where(done[:, None], post["policy"], self.obs)
                self.reader.refresh_frame()
            self.sim_steps += self.env.num_envs
        return dict(
            **{"reward_parity/" + k: torch.stack(v).mean() for k, v in term_gaps.items()},
            # Absent when no episode finished during this collection call.
            **{key: total / count for key, (total, count) in episode_metrics.items()},
            norm_count=self.wm.norm.count.clone(),
            delta_norm_count=self.wm.delta_norm.count.clone(),
            reward=torch.stack(rewards).mean(),
            reward_parity=torch.stack(gaps).mean(),
            drop_rate=torch.stack(drops).mean(),
            rot_error=torch.stack(rotation_errors).mean(),
            **{
                "finger_" + str(i + 1) + "_action": value
                for i, value in enumerate(torch.stack(finger_actions).mean(0))
            },
        )

    def fit_world_model(self, holdout=False):
        c = self.cfg
        window = c.wm_window
        w = self.buffer.sample(
            c.wm_batch, window, holdout,
            failure_frac=0.0 if holdout or c.variant == "legacy" else c.wm_failure_frac,
            failure_horizon=c.wm_failure_horizon,
        )
        memory = self.unflatten_memory(w["m"][:, 0])
        auxiliary = w["aux"][:, 0]
        simulator = w["sim_context"] if "sim_context" in w else self.reader.xi[w["env"]]
        if not holdout and c.variant == "legacy":
            # Deliberately reproduce the old WM statistics bug in the control arm.
            # The fixed arm never updates statistics from replay.
            initial_dynamics = self.observations.dynamics(
                w["x"][:, 0], simulator, self.observations.disturbance(w["aux"][:, 0])
            )
            self.wm.norm.update(self.wm.inputs(initial_dynamics, w["u"][:, 0]))
            self.wm.delta_norm.update(difference(w["x"], w["x1"])[w["mask"].bool()])
        x = w["x"][:, 0]
        losses = []
        state_losses = []
        contact_losses = []
        failure_features = []
        std = torch.maximum(self.wm.delta_norm.std(), self.wm.error_floor)
        for j in range(window):
            dynamics = self.wm_features(x, simulator, self.observations.disturbance(auxiliary))
            x1, contact = self.step_wm(x, w["u"][:, j], dynamics)
            state_loss = (difference(x1, w["x1"][:, j]) / std).square().mean(-1)
            contact_loss = F.binary_cross_entropy_with_logits(contact, w["contact"][:, j], reduction="none").sum(-1)
            losses.append(
                c.final_state_weight * state_loss
                + c.contact_weight * contact_loss
            )
            state_losses.append(state_loss)
            contact_losses.append(contact_loss)
            torque_kwargs = {"limits": memory["limits"]} if self.is_pen else {}
            tau = self.mdp.pd_torque(x, w["u"][:, j], simulator, **torque_kwargs)
            _, cage, _ = self.reward_step(
                x1,
                memory,
                w["action"][:, j],
                w["c"][:, j],
                contact.sigmoid(),
                tau,
                soft=True,
            )
            # Keep the current goal during supervised open-loop replay; it is
            # not a teacher-forced geometry/context input, only the exogenous
            # command realization when a switch would occur.
            next_goal = memory["goal"].reshape(-1, 3, 3)
            memory = self.advance_memory(
                x1, memory, w["action"][:, j], w["c"][:, j], cage, next_goal
            )
            failure_features.append(
                self.fail_head.features(
                    x1.detach(),
                    self.observations.context(x1.detach(), memory, auxiliary, simulator),
                )
            )
            push = w["push"][:, j]
            event = push[..., -6:-3].abs().sum(-1, keepdim=True) > 0
            world_velocity = push[..., -6:-3]
            world_direction = torch.where(
                event,
                world_velocity / world_velocity.norm(dim=-1, keepdim=True).clamp_min(1e-8),
                world_velocity,
            )
            if self.is_pen:
                auxiliary = w["aux"][:, min(j + 1, window - 1)]
            else:
                auxiliary = self.advance_disturbance(auxiliary, event, world_direction, world_velocity)
            x = x1 + w["push"][:, j]
        mask = w["mask"]
        denom = mask.sum().clamp_min(1)
        wm_loss = (torch.stack(losses, 1) * mask).sum() / mask.shape[0]
        state_component = (torch.stack(state_losses, 1) * mask).sum() / denom
        contact_component = (torch.stack(contact_losses, 1) * mask).sum() / denom
        if not holdout:
            self.optimize("wm", wm_loss)
        true = self.replay_failure_features(w, simulator)
        predicted = torch.stack(failure_features, 1).detach()
        if not holdout and c.variant == "legacy":
            self.fail_head.norm.update(true[w["mask"].bool()])
        failure_logit_true = self.fail_head(true)
        failure_logit_pred = self.fail_head(predicted)
        label = w["terminated"].float()
        failure_loss = (
            (
                F.binary_cross_entropy_with_logits(failure_logit_true, label, reduction="none")
                + F.binary_cross_entropy_with_logits(failure_logit_pred, label, reduction="none")
            )
            * mask
            * w["importance"]
        ).sum() / (2 * denom)
        if not holdout:
            self.optimize("fail_head", failure_loss)
        return dict(
            supervision_window=window,
            wm_loss=wm_loss.detach(),
            failure_loss=failure_loss.detach(),
            state_loss=state_component.detach(),
            contact_digit_loss=contact_component.detach(),
        )

    @torch.no_grad()
    def replay_failure_features(self, w, simulator):
        # Recorded states are independent across time. Batch B*T together so
        # observation/FK kernels launch once rather than once per window step.
        batch, window = w["x1"].shape[:2]
        x = w["x1"].flatten(0, 1)
        context = self.observations.context(
            x,
            self.unflatten_memory(w["m"].flatten(0, 1)),
            w["aux"].flatten(0, 1),
            simulator[:, None].expand(-1, window, -1).reshape(batch * window, -1),
        )
        return self.fail_head.features(x, context).reshape(batch, window, -1)

    def load_model_only(self, path):
        """A/B starts actor, critic and optimizers fresh; source checkpoint is read-only."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint_task(state["config"]) != self.cfg.task:
            raise ValueError("Model initialization task mismatch")
        if getattr(self, "is_pen", False) and state.get("reference_signature") != self.mdp.reference_signature:
            raise ValueError("Pen-spin reference data mismatch")
        if (
            state["format_version"] not in (2, 3)
            or state["joint_names"] != list(self.reader.robot.joint_names)
            or state["context_names"] != self.reader.context_names
        ):
            raise ValueError("Model initialization layout mismatch")
        if state["config"]["control_hz"] != self.cfg.control_hz:
            raise ValueError("Model initialization frequency mismatch")
        for name in ("wm", "fail_head"):
            self.modules()[name].load_state_dict(state["modules"][name])
        if "torque_norm" in state:
            self.torque_norm.load_state_dict(state["torque_norm"])
        if self.cfg.reset_model_statistics:
            for norm in (self.wm.norm, self.wm.delta_norm, self.fail_head.norm, self.torque_norm):
                norm.mean.zero_()
                norm.var.fill_(1.0)
                norm.count.zero_()
                norm._host_count = 0.0

    def optimize(self, name, loss):
        optimizer = self.optimizers[name]
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {name} loss")
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.modules()[name].parameters(), 1.0)
        if name == "actor":
            nn.utils.clip_grad_value_(self.actor.parameters(), 0.1)
        if not torch.isfinite(norm):
            raise FloatingPointError(f"Non-finite {name} gradient")
        optimizer.step()

    def lr_at(self, iteration):
        c = self.cfg
        progress = min(iteration / c.lr_decay_iters, 1.0)
        return c.lr * (c.lr_final_ratio + (1 - c.lr_final_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))

    def sample_starts(self):
        # Like SuperMimic, draw uniformly from all eligible replay states,
        # including held-out envs (the WM fit itself remains held out).
        rows = (self.buffer.ptr - self.buffer.size + torch.arange(self.buffer.size, device=self.device)) % self.buffer.capacity
        eligible = self.buffer.data["valid"][rows].nonzero(as_tuple=False)
        if not len(eligible):
            raise RuntimeError("Replay contains no finite imagination starts")
        selected = eligible[torch.randint(len(eligible), (self.cfg.pi_batch,), device=self.device)]
        row, env = rows[selected[:, 0]], selected[:, 1]
        out = {k: v[row, env] for k, v in self.buffer.data.items()}
        out.update(env=env, sim_context=self.reader.xi[env])
        return out

    def policy_update(self):
        c = self.cfg
        w = self.sample_starts()
        x = w["x"]
        m = self.unflatten_memory(w["m"])
        history = self.history_from_obs(w["obs"])
        auxiliary = w["aux"]
        simulator = w["sim_context"]
        bias = self.reader.robot.data.encoder_bias[w["env"]]
        horizon = c.horizon
        frozen = (self.wm, self.fail_head, self.critic)
        for mod in frozen:
            mod.requires_grad_(False)
        xs = []
        rewards = []
        continues = []
        survival = x.new_ones(x.shape[0])
        objective = x.new_zeros(x.shape[0])
        # These quantities are constant for this update; keep them out of the
        # recurrent loop to avoid repeating host work and Python-side control flow.
        gamma_powers = [c.gamma**t for t in range(horizon + 1)]
        scale = min(
            1,
            max(
                0,
                (self.env.common_step_counter / self.env.max_common_steps - 0.05) / 0.75,
            ),
        )
        adaptive = float(getattr(self.env, "_adaptive_episode_curriculum", 0.05))
        try:
            for t in range(horizon):
                context = self.task_features(x, m, auxiliary, simulator)
                xs.append(self.critic_features(context))
                a = self.step_actor(self.mdp.flatten_history(history))
                target, u = self.mdp.control_target(a, m, bias) if self.is_pen else ema_target(
                    a,
                    m["c1"],
                    self.mdp.default,
                    self.mdp.lower,
                    self.mdp.upper,
                    bias,
                    self.mdp.alpha,
                    m["age"].squeeze(-1),
                    self.mdp.dt,
                )
                x1, contact = self.step_wm(
                    x, u, self.wm_features(x, simulator, self.observations.disturbance(auxiliary))
                )
                torque_kwargs = {"limits": m["limits"]} if self.is_pen else {}
                tau = self.mdp.pd_torque(x, u, simulator, **torque_kwargs)
                r, cage, _ = self.reward_step(x1, m, a, target, contact.sigmoid(), tau, soft=True)
                # Original termination reads the PRE-reward counter.
                if self.is_pen:
                    d = self.mdp.termination(x1, m)
                else:
                    d = torch.sigmoid((m["cage"].squeeze(-1) - self.mdp.max_outside + 0.5) / 0.25)
                if self.is_pen:
                    # A finite demonstration ends the imagined return; it is
                    # excluded from the physical-failure head's labels.
                    reference_end = (m["time"] + 1 >= m["end"] - m["start"]).squeeze(-1)
                    d = torch.where(reference_end, 1.0, d)
                next_goal = quat_to_mat(F.normalize(torch.randn(x.shape[0], 4, device=x.device), dim=-1))
                next_memory = self.advance_memory(
                    x1,
                    m,
                    a,
                    target,
                    cage,
                    next_goal,
                )
                if c.failure_head and self.iteration >= c.wm_only_iters + c.failure_warmup_iters:
                    next_context = self.task_features(x1, next_memory, auxiliary, simulator)
                    p_fail = self.fail_head(
                        self.fail_head.features(x1, next_context)
                    ).sigmoid().clamp(max=c.failure_clip)
                    d = 1 - (1 - d) * (1 - p_fail)
                # A timeout is not a physical termination; the critic bootstraps it.
                objective = objective + gamma_powers[t] * survival * r
                survival = survival * (1 - d)
                rewards.append(r.detach())
                continues.append((1 - d).detach())
                updated = next_memory
                # Match interval velocity disturbances independently of the physical model.
                push = torch.zeros_like(x1)
                speed = 0.05 + torch.rand(x.shape[0], 1, device=x.device) * 0.1 * scale * adaptive
                dv = F.normalize(torch.randn(x.shape[0], 3, device=x.device), dim=-1) * speed
                event = torch.rand(x.shape[0], 1, device=x.device) < self.mdp.dt / 1.2
                active = event & (
                    updated["age"] * self.mdp.dt >= 3
                )
                push[..., -6:-3] = 0 if self.is_pen else dv * active
                # The task caches the most recent sampled world-frame disturbance.
                # A sampled event before warmup clears the cache as in the simulator.
                palm_rotation = auxiliary[:, 3:12].reshape(-1, 3, 3)
                world_dv = mv(palm_rotation, dv * active)
                world_dir = mv(palm_rotation, F.normalize(dv, dim=-1) * active)
                if not self.is_pen:
                    auxiliary = self.advance_disturbance(auxiliary, event, world_dir, world_dv)
                x = x1 + push
                m = updated
                observation = (
                    self.mdp.policy_observation(x, m, auxiliary) if self.is_pen
                    else self.mdp.noisy_observation(self.mdp.observation(x, m))
                )
                history = torch.cat((history[:, 1:], observation[:, None]), 1)
            context = self.task_features(x, m, auxiliary, simulator)
            final = self.critic_features(context)
            xs.append(final)
            objective = objective + (c.gamma**horizon) * survival * self.critic(final)
            actor_loss = -objective.mean()
            self.optimize("actor", actor_loss)
        finally:
            for mod in frozen:
                mod.requires_grad_(True)
        inputs = torch.stack(xs).detach()
        rs = torch.stack(rewards)
        cs = torch.stack(continues)
        with torch.no_grad():
            self.critic.norm.update(inputs)
            values = self.critic(inputs)
            ret = values[-1]
            targets = []
            for t in reversed(range(horizon)):
                ret = rs[t] + c.gamma * cs[t] * ((1 - c.lam) * values[t + 1] + c.lam * ret)
                targets.append(ret)
            targets = torch.stack(targets[::-1])
        features = inputs[:-1].flatten(0, 1)
        targets = targets.flatten()
        count = features.shape[0]
        minibatch = count // c.critic_minibatches
        for _ in range(c.critic_epochs):
            permutation = torch.randperm(count, device=self.device)
            for i in range(c.critic_minibatches):
                ids = permutation[i * minibatch : (i + 1) * minibatch]
                critic_loss = (self.critic(features[ids]) - targets[ids]).square().mean()
                self.optimize("critic", critic_loss)
        return dict(
            actor_loss=actor_loss.detach(),
            critic_loss=critic_loss.detach(),
            survival=survival.detach().mean(),
            horizon=horizon,
        )

    def learn(self):
        if self.cfg.algorithm == "ppo":
            return self.learn_ppo()
        start = self.started_at
        phase_times = {}
        phase_counts = {}

        class PhaseTimer:
            def __init__(self, name):
                self.name = name

            def __enter__(self):
                if torch.cuda.is_initialized():
                    torch.cuda.synchronize()
                self.started = time.perf_counter()
                return self

            def __exit__(self, *_):
                if torch.cuda.is_initialized():
                    torch.cuda.synchronize()
                phase_times[self.name] = phase_times.get(self.name, 0.0) + time.perf_counter() - self.started
                phase_counts[self.name] = phase_counts.get(self.name, 0) + 1

        self.collect(max(self.cfg.warmup_steps, self.cfg.wm_window))
        step_acc = 0.0
        stats = {}
        for it in range(self.iteration, self.cfg.max_iterations):
            self.iteration = it
            lr = self.lr_at(it)
            for optimizer in self.optimizers.values():
                for group in optimizer.param_groups:
                    group["lr"] = lr
            step_acc += 1.0 / self.cfg.utd
            collect_steps = int(step_acc)
            step_acc -= collect_steps
            if self.cfg.profile:
                with PhaseTimer("collect"):
                    stats = self.collect(collect_steps)
                with PhaseTimer("wm"):
                    for _ in range(self.cfg.wm_batches):
                        stats.update(self.fit_world_model())
                if it >= self.cfg.wm_only_iters:
                    with PhaseTimer("policy"):
                        stats.update(self.policy_update())
            else:
                stats = self.collect(collect_steps)
                for _ in range(self.cfg.wm_batches):
                    stats.update(self.fit_world_model())
                if it >= self.cfg.wm_only_iters:
                    stats.update(self.policy_update())
            if it % self.cfg.log_interval == 0:
                if self.cfg.holdout_fraction:
                    with torch.no_grad():
                        stats.update({"holdout/" + k: v for k, v in self.fit_world_model(True).items()})
                stats.update(
                    lr=lr,
                    wall_seconds=self.wall_offset + time.perf_counter() - start,
                    sim_steps=self.sim_steps,
                )
                if self.cfg.profile:
                    elapsed = max(sum(phase_times.values()), 1e-12)
                    stats.update(
                        {
                            **{f"profile/seconds/{k}": v for k, v in phase_times.items()},
                            **{f"profile/calls/{k}": v for k, v in phase_counts.items()},
                            **{
                                f"profile/fraction/{k}": v / elapsed
                                for k, v in phase_times.items()
                            },
                        }
                    )
                scalar = organize_log_scalars(stats, it)
                self.writer.log_scalars(scalar, it)
                print(format_training_log(it, self.cfg.max_iterations, scalar), flush=True)
            if (it + 1) % self.cfg.save_interval == 0:
                self.save(self.log_dir / f"model_{it + 1}.pt", it + 1)
        self.save(self.log_dir / "model_final.pt", self.cfg.max_iterations)
        self.writer.close()
        summary = {
            "completed": True,
            "iteration": self.cfg.max_iterations,
            "sim_steps": self.sim_steps,
            "wall_time": self.wall_offset + time.perf_counter() - start,
            "task": self.cfg.task,
            "algorithm": "world_model",
            "seed": self.cfg.seed,
        }
        for name in ("wm_loss", "state_loss", "contact_digit_loss", "failure_loss", "actor_loss", "critic_loss"):
            if name in stats:
                summary[name] = float(stats[name])
        (self.log_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    @torch.no_grad()
    def collect_ppo(self, steps):
        """Collect a complete on-policy rollout for one PPO update."""
        obs_dim = self.obs.shape[-1]
        action_dim = len(self.reader.robot.joint_names)
        if self.ppo_memory is None:
            self.ppo_memory = {
                "obs": self.obs.new_empty(steps, self.env.num_envs, obs_dim),
                "critic_obs": self.obs.new_empty(steps, self.env.num_envs, self.ppo_critic.norm._mean.shape[-1]),
                "actions": self.obs.new_empty(steps, self.env.num_envs, action_dim),
                "log_prob": self.obs.new_empty(steps, self.env.num_envs),
                "value": self.obs.new_empty(steps, self.env.num_envs),
                "reward": self.obs.new_empty(steps, self.env.num_envs),
                "terminated": self.obs.new_empty(steps, self.env.num_envs),
                "truncated": self.obs.new_empty(steps, self.env.num_envs),
            }
        elif self.ppo_memory["obs"].shape[0] != steps:
            self.ppo_memory = {
                name: value.new_empty(steps, *value.shape[1:]) for name, value in self.ppo_memory.items()
            }
        memory = self.ppo_memory

        # PPO uses conventional vectorized auto-reset. Reuse the same actor,
        # normalization, action dimensions, and task configuration.
        self.env.cfg.auto_reset = True
        self.env._manual_reset_pending.zero_()
        gamma = self.cfg.ppo_gamma
        for t in range(steps):
            self.actor.norm.update(self.obs)
            distribution = self.actor.distribution(self.obs)
            action = distribution.sample()
            critic_obs = self._critic_observation()
            critic = self._ensure_ppo_critic(critic_obs)
            self.ppo_critic.norm.update(critic_obs)
            value = critic(critic_obs)
            obs, reward, terminated, truncated, _ = self.env.step(action)
            # rsl-rl treats timeouts as done but adds a bootstrapping reward
            # correction before GAE. This reproduces upstream Wuji-MJLab exactly.
            reward = reward + gamma * value * truncated.float()
            memory["obs"][t] = self.obs
            memory["critic_obs"][t] = critic_obs
            memory["actions"][t] = action
            memory["log_prob"][t] = distribution.log_prob(action).sum(-1)
            memory["value"][t] = value
            memory["reward"][t] = reward
            memory["terminated"][t] = terminated
            memory["truncated"][t] = truncated
            self.obs = obs["policy"]
            self.sim_steps += self.env.num_envs

        self.reader.refresh_frame()
        drop_name = "reference_end" if self.is_pen else "cage_drop"
        drop = self.env.termination_manager.get_term(drop_name).float().mean()
        return dict(
            reward=memory["reward"].mean(),
            drop_rate=drop,
            actor_loss=memory["log_prob"].square().mean(),
            critic_loss=memory["value"].square().mean(),
            entropy=torch.zeros((), device=self.device),
            clip_fraction=torch.zeros((), device=self.device),
            kl=torch.zeros((), device=self.device),
            explained_variance=torch.zeros((), device=self.device),
            horizon=steps,
        )

    def _critic_observation(self):
        return self.env.observation_manager.compute(update_history=False)["critic"]

    def ppo_update(self):
        c = self.cfg
        memory = self.ppo_memory
        batch, num_envs = memory["reward"].shape
        flat_actor_obs = memory["obs"].reshape(-1, memory["obs"].shape[-1])
        flat_obs = memory["critic_obs"].reshape(-1, memory["critic_obs"].shape[-1])
        flat_actions = memory["actions"].reshape(-1, memory["actions"].shape[-1])
        flat_old_log_prob = memory["log_prob"].reshape(-1)
        flat_values = memory["value"].reshape(-1)

        # rsl-rl GAE with timeout reward bootstrapping already applied above.
        rewards = memory["reward"]
        dones = (memory["terminated"].bool() | memory["truncated"].bool()).float()
        critic = self._ensure_ppo_critic(memory["critic_obs"])
        final_value = critic(self._critic_observation()).flatten()
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(num_envs, device=self.device)
        for t in reversed(range(batch)):
            next_non_terminal = 1.0 - dones[t]
            next_value = final_value if t == batch - 1 else memory["value"][t + 1]
            delta = rewards[t] + c.ppo_gamma * next_non_terminal * next_value - memory["value"][t]
            gae = delta + c.ppo_gamma * c.lam * next_non_terminal * gae
            advantages[t] = gae
        flat_advantages = advantages.reshape(-1)
        returns = flat_advantages + flat_values
        if c.ppo_normalize_advantage:
            flat_advantages = (flat_advantages - flat_advantages.mean()) / flat_advantages.std().clamp_min(1e-8)

        count = batch * num_envs
        usable_count = (count // c.ppo_minibatches) * c.ppo_minibatches
        minibatch_size = max(1, count // c.ppo_minibatches)
        actor_losses, critic_losses, entropies, clip_fractions, kls = [], [], [], [], []
        for _ in range(c.ppo_epochs):
            permutation = torch.randperm(usable_count, device=self.device)
            for start in range(0, usable_count, minibatch_size):
                ids = permutation[start : start + minibatch_size]
                distribution = self.actor.distribution(flat_actor_obs[ids])
                log_prob = distribution.log_prob(flat_actions[ids]).sum(-1)
                with torch.no_grad():
                    old_distribution = self.actor.distribution(flat_actor_obs[ids])
                ratio = (log_prob - flat_old_log_prob[ids]).exp()
                policy_loss = -torch.min(
                    ratio * flat_advantages[ids],
                    ratio.clamp(1.0 - c.ppo_clip, 1.0 + c.ppo_clip) * flat_advantages[ids],
                ).mean()
                entropy = distribution.entropy().sum(-1).mean()
                values = critic(flat_obs[ids])
                value_loss = (values - returns[ids]).square().mean()
                loss = policy_loss + c.ppo_value_coef * value_loss - c.ppo_entropy * entropy
                self.optimizers["ppo"].zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    tuple(self.actor.parameters()) + tuple(critic.parameters()), c.ppo_max_grad_norm
                )
                self.optimizers["ppo"].step()
                with torch.no_grad():
                    clip_fractions.append(((ratio - 1.0).abs() > c.ppo_clip).float().mean())
                    kls.append(
                        torch.distributions.kl_divergence(old_distribution, distribution).sum(-1).mean()
                    )
                actor_losses.append(policy_loss.detach())
                critic_losses.append(value_loss.detach())
                entropies.append(entropy.detach())

        target_variance = returns.var()
        explained = 1.0 - (returns - critic(flat_obs)).square().mean() / target_variance.clamp_min(1e-8)
        return dict(
            actor_loss=torch.stack(actor_losses).mean(),
            critic_loss=torch.stack(critic_losses).mean(),
            entropy=torch.stack(entropies).mean(),
            clip_fraction=torch.stack(clip_fractions).mean(),
            kl=torch.stack(kls).mean(),
            explained_variance=explained.detach(),
            horizon=batch,
        )

    def learn_ppo(self):
        start = self.started_at
        for it in range(self.iteration, self.cfg.max_iterations):
            self.iteration = it
            stats = self.collect_ppo(self.cfg.ppo_rollout_steps)
            stats.update(self.ppo_update())
            stats.update(
                lr=self.cfg.ppo_lr,
                wall_seconds=self.wall_offset + time.perf_counter() - start,
                sim_steps=self.sim_steps,
            )
            if it % self.cfg.log_interval == 0:
                scalar = organize_log_scalars(stats, it)
                self.writer.log_scalars(scalar, it)
                print(format_training_log(it, self.cfg.max_iterations, scalar), flush=True)
            if (it + 1) % self.cfg.save_interval == 0:
                self.save(self.log_dir / f"model_{it + 1}.pt", it + 1)
        self.save(self.log_dir / "model_final.pt", self.cfg.max_iterations)
        self.writer.close()

    def save(self, path, next_iteration=None):
        state = dict(
            format_version=self.format_version(),
            reference_signature=getattr(self.mdp, "reference_signature", None),
            torque_norm=self.torque_norm.state_dict(),
            config=self.cfg.to_dict(),
            iteration=self.iteration if next_iteration is None else next_iteration,
            sim_steps=self.sim_steps,
            context_dim=self.context_dim,
            contact_dim=self.contact_dim,
            modules={k: v.state_dict() for k, v in self.modules().items()},
            optimizers={k: v.state_dict() for k, v in self.optimizers.items()},
            context_names=self.reader.context_names,
            joint_names=list(self.reader.robot.joint_names),
            wall_seconds=self.wall_offset + time.perf_counter() - self.started_at,
            common_step=self.env.common_step_counter,
            curriculum_scales={
                k: getattr(self.env, k)
                for k in (
                    "_adaptive_episode_curriculum",
                    "_reorient_success_curriculum",
                )
                if hasattr(self.env, k)
            },
            cuda_rng=torch.cuda.get_rng_state_all() if str(self.device).startswith("cuda") else None,
            torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),
        )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def close(self):
        self.writer.close()
        self.reader.detach_capture()

    def load(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint_task(state["config"]) != self.cfg.task:
            raise ValueError("Checkpoint task mismatch")
        if getattr(self, "is_pen", False) and state.get("reference_signature") != self.mdp.reference_signature:
            raise ValueError("Pen-spin reference data mismatch")
        expected_version = self.format_version()
        if expected_version == 3 and (
            "utd" not in state["config"] or "critic_minibatches" not in state["config"]
        ):
            raise ValueError(
                "Checkpoint predates the SuperMimic training recipe. Start a fresh run, "
                "or use --model-checkpoint to initialize only the world model and failure head."
            )
        if state["format_version"] != expected_version or state["joint_names"] != list(
            self.reader.robot.joint_names
        ):
            raise ValueError("Checkpoint schema/joint order mismatch")
        if state["context_names"] != self.reader.context_names:
            raise ValueError("Context layout mismatch")
        if state["config"]["control_hz"] != self.cfg.control_hz:
            raise ValueError("Checkpoint control frequency mismatch")
        if "torque_norm" in state:
            self.torque_norm.load_state_dict(state["torque_norm"])
        for k, mod in self.modules().items():
            mod.load_state_dict(state["modules"][k])
        for k, opt in self.optimizers.items():
            opt.load_state_dict(state["optimizers"][k])
        self.iteration = state["iteration"]
        self.sim_steps = state["sim_steps"]
        self.env.common_step_counter = state.get("common_step", 0)
        self.wall_offset = state.get("wall_seconds", 0.0)
        self.started_at = time.perf_counter()
        for key, value in state.get("curriculum_scales", {}).items():
            setattr(self.env, key, value)
        if state.get("cuda_rng") is not None and str(self.device).startswith("cuda"):
            torch.cuda.set_rng_state_all([v.to(device="cpu", dtype=torch.uint8) for v in state["cuda_rng"]])
        torch.set_rng_state(state["torch_rng"].to(device="cpu", dtype=torch.uint8))
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        # Replay is intentionally refilled in the current randomized environments.

"""Dense reorientation rewards and termination terms."""

from __future__ import annotations

import torch
from unilab.managers import ManagerTermBase

from .commands import task
from .math import conjugate, rotate


def _resolve_curriculum_env_ids(env, env_ids) -> torch.Tensor:
    """Normalize a UniLab curriculum reset selector to concrete indices."""
    device = torch.device(getattr(env, "device", "cpu"))
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=device, dtype=torch.long).reshape(-1)


def reward(
    env,
    kind: str,
    max_outside_steps: int = 10,
    drop_scale: float = 4.0,
    up_axis: int = 0,
    max_count: float = 15.0,
) -> torch.Tensor:
    del up_axis, max_count
    state = task(env)
    error = state.error()
    if kind in ("orientation", "orientation_alignment"):
        excess_error = torch.clamp(error - 0.2, min=0.0)
        return torch.clamp(1.0 - excess_error / torch.pi, min=0.0, max=1.0)
    if kind in ("hold", "hold_escalation"):
        return (
            ((state.window > 0) & state.within).float()
            # Keep the pre-switch value for the final successful-window
            # transition.  ``reward_hold_counter`` is cleared when a new
            # goal is sampled, while the source command exposes this
            # snapshot specifically so the escalation reward is not dropped.
            * state.reward_hold_counter_snapshot.float()
        )
    if kind in ("cage", "cage_escape"):
        return (
            state.outside.float()
            * state.cage_counter
            / max(float(max_outside_steps), 1.0)
            * float(drop_scale)
        )
    if kind == "hand_pose":
        joints = env.scene.read_plan.joint_tensor_view("robot").joint_pos
        default = state.robot.data.default_joint_pos(env.device)
        default = default[0] if default.ndim == 2 else default
        return torch.square(joints - default).sum(-1)
    if kind == "action_rate":
        action = env.action_manager
        delta = torch.square(action.action - action.prev_action).sum(-1)
        previous_previous = getattr(action, "prev_prev_action", action.prev_action)
        jerk = torch.square(action.action - 2 * action.prev_action + previous_previous).sum(-1)
        return delta + jerk
    if kind in ("action_hf", "action_high_frequency"):
        # The source ActionHighFreqPenalty keeps its own c1/c2 history and
        # evaluates the current EMA target after action processing.  Reusing
        # only ``memory_tensor()['c1']`` would compute a first difference and
        # silently remove the second-order penalty.
        try:
            term = env.action_manager.get_term("joint_pos")
            jerk = term.target - 2.0 * term.previous_target + term.previous_previous_target
            age = env.episode_length_buf
            return torch.square(jerk).sum(-1) * (age >= 2)
        except (AttributeError, KeyError, RuntimeError):
            action = env.action_manager
            previous_previous = getattr(action, "prev_prev_action", action.prev_action)
            return torch.square(action.action - 2 * action.prev_action + previous_previous).sum(-1)
    if kind in ("torque", "torque_penalty"):
        try:
            torque = state.torque_tensor()
            limits = getattr(getattr(state, "mdp", None), "effort_limit", None)
            if limits is not None:
                torque = torque / limits.clamp_min(1e-3)
        except (AttributeError, RuntimeError):
            torque = env.action_manager.action.new_zeros((env.num_envs, 20))
        return torch.square(torque).sum(-1)
    if kind in ("finger_collision", "finger_self_collision"):
        try:
            contact = state.contact_tensor()
            return contact[..., 7:].sum(-1)
        except (AttributeError, RuntimeError):
            return state.cage_counter.new_zeros(state.cage_counter.shape)
    if kind in ("palm_detach", "palm_detach_reward"):
        try:
            contact = state.contact_tensor()
            return (1.0 - contact[..., 5]).clamp_min(0.0) * contact[..., 6]
        except (AttributeError, RuntimeError):
            return state.cage_counter.new_zeros(state.cage_counter.shape)
    if kind in ("tip_slide", "tip_slide_penalty"):
        try:
            # The source uses tip-site velocities (including each site's
            # offset from its distal body), not distal body-origin velocity.
            tips = state.tip_velocity_tensor()
            relative_velocity = torch.linalg.vector_norm(
                tips - state.cube_velocity()[:, None, :], dim=-1
            )
            in_contact = state.contact_tensor()[:, :5] > 0
            return (relative_velocity * in_contact.float()).sum(-1)
        except (AttributeError, RuntimeError, ValueError):
            return state.cage_counter.new_zeros(state.cage_counter.shape)
    if kind == "joint_velocity":
        return torch.square(env.scene.read_plan.joint_tensor_view("robot").joint_vel).mean(-1)
    raise ValueError(f"Unknown Hand2 reward kind: {kind}")


def nan_detection(env) -> torch.Tensor:
    """Detect invalid physical or action values before they reach the learner."""
    state = task(env)
    values = [state.state_tensor(), state.context_tensor()]
    action = getattr(env.action_manager, "action", None)
    if action is not None:
        values.append(action)
    return ~torch.isfinite(torch.cat(values, dim=-1)).all(dim=-1)

class CageDrop(ManagerTermBase):
    """Terminate after the cube leaves a palm-centered bounding cage."""

    def __init__(self, cfg, env):
        super().__init__(env)
        self.state = task(env)

    @property
    def entity_name(self) -> str:
        return "robot"

    @property
    def tensor_state_entity(self) -> str:
        return "robot"

    @property
    def tensor_body_names(self):
        # Match Dex-SuperMimic's cage IDs: palm and the four links on each
        # finger.  Calibration/tag bodies are deliberately excluded; they are
        # metadata frames and must not enlarge the physical hand AABB.
        return tuple(
            name for name in self.state.robot.body_names
            if name.endswith("right_palm_link")
            or any(name.endswith(f"right_finger{finger}_link{link}")
                   for finger in range(1, 6) for link in range(1, 5))
        )

    def __call__(
        self,
        env,
        margin: float = 0.01,
        max_outside_steps: int = 10,
        max_count: float = 15.0,
        up_axis: int = 0,
        up_margin: float = 0.03,
    ):
        p, q = self.state.palm_pose()
        bodies = env.scene.read_plan.body_tensor_view("robot", self.tensor_body_names)
        cube = self.state.cube_position()
        local = rotate(conjugate(q)[:, None, :], bodies.pos_w - p[:, None, :])
        cube_local = rotate(conjugate(q), cube - p)
        low = local.amin(dim=1) - torch.tensor(
            (margin, margin, margin), device=env.device, dtype=cube.dtype
        )
        high = local.amax(dim=1) + float(margin)
        high[:, int(up_axis)] = local[:, :, int(up_axis)].amax(dim=1) + float(up_margin)
        outside = torch.any((cube_local < low) | (cube_local > high), dim=-1)
        self.state.outside.copy_(outside)
        if max_count < max_outside_steps:
            raise ValueError("CageDrop max_count must be >= max_outside_steps")
        # UniLab evaluates termination before reward terms.  Dex-SuperMimic's
        # cage_drop reads the counter from the previous reward update, then
        # CageEscapePenalty advances it for the current reward.  Preserve that
        # one-step ordering even though this term owns the shared counter.
        terminated = self.state.cage_counter >= max_outside_steps
        self.state.cage_counter.copy_(torch.clamp(
            self.state.cage_counter + torch.where(outside, 1.0, -0.5),
            0.0,
            float(max_count),
        ))
        return terminated


def metric(env, kind: str) -> torch.Tensor:
    state = task(env)
    if kind in ("success", "goal_reach_count"):
        return state.goal_count.float()
    if kind == "orientation_error":
        return state.error()
    if kind in ("cube_survival_steps", "survival"):
        return env.episode_length_buf.float()
    if kind == "joint_vel_rms":
        vel = env.scene.read_plan.joint_tensor_view("robot").joint_vel
        return torch.sqrt(torch.square(vel).mean(-1))
    if kind == "action_delta_rms":
        action = env.action_manager
        return torch.sqrt(torch.square(action.action - action.prev_action).mean(-1))
    if kind == "action_jerk_rms":
        action = env.action_manager
        previous_previous = getattr(action, "prev_prev_action", action.prev_action)
        jerk = action.action - 2 * action.prev_action + previous_previous
        return torch.sqrt(torch.square(jerk).mean(-1))
    if kind == "cage_escape_frequency":
        return state.outside.float()
    if kind == "fingertip_contact_count":
        return (state.contact_tensor()[:, :5] > 0).sum(-1).float()
    if kind == "torque_saturation_ratio":
        torque = state.torque_tensor().abs()
        limits = getattr(getattr(state, "mdp", None), "force_limit", None)
        if limits is None:
            return torch.zeros(torque.shape[0], device=torque.device)
        return (torque > 0.9 * limits).float().mean(-1)
    if kind == "joint_acceleration_rms":
        # qacc is not currently part of UniLab's public read plan. Keep the
        # metric shape-compatible until that capability is exposed.
        velocity = env.scene.read_plan.joint_tensor_view("robot").joint_vel
        return torch.zeros_like(velocity[..., 0])
    if kind == "cube_height_above_palm":
        return state.state_tensor()[:, 40:43][:, 2]
    if kind == "finger_collision_frequency":
        return (state.contact_tensor()[:, 7:].sum(-1) > 0).float()
    if kind == "palm_detach_frequency":
        contact = state.contact_tensor()
        return ((contact[:, 5] <= 0) & (contact[:, 6] > 0)).float()
    if kind == "success_interval":
        return state.goal_timer.float()
    raise ValueError(f"Unknown Hand2 metric kind: {kind}")


def success_curriculum(
    env,
    env_ids=None,
    count_threshold: int = 3,
    delta_per_loop: float = 0.08,
    **kwargs,
):
    command_name = kwargs.pop("command_name", "reorient")
    if kwargs:
        raise TypeError(f"Unknown success curriculum parameters: {tuple(kwargs)}")
    state = task(env)
    if not hasattr(env, "_reorient_success_curriculum"):
        env._reorient_success_curriculum = 0.0
    rows = _resolve_curriculum_env_ids(env, env_ids)
    if rows.numel() == 0:
        return {"value": float(env._reorient_success_curriculum)}

    # The source consumes command metrics at the reset boundary.  Keep the
    # state-owned goal count as a fallback for old/lightweight command terms.
    command = env.command_manager.get_term(command_name)
    metrics = getattr(command, "metrics", {})
    counts = metrics.get("goal_reach_count")
    if counts is None:
        counts = state.goal_count
    counts = counts[rows]
    success = counts >= float(count_threshold)
    delta_per_env = float(delta_per_loop) / max(float(env.num_envs), 1.0)
    delta = torch.where(
        success,
        torch.full_like(counts, delta_per_env, dtype=torch.float32),
        torch.full_like(counts, -delta_per_env, dtype=torch.float32),
    ).sum()
    env._reorient_success_curriculum = float(
        max(min(env._reorient_success_curriculum + float(delta), 1.0), 0.0)
    )
    state.difficulty = env._reorient_success_curriculum
    return {
        "value": float(env._reorient_success_curriculum),
        "mean_goal_reach_count": float(counts.float().mean().item()),
        "success_env_frac": float(success.float().mean().item()),
        "delta": float(delta),
    }


def adaptive_episode_curriculum(
    env,
    env_ids=None,
    target_steps: int = 800,
    inc_rate: float = 0.1,
    dec_rate: float = 0.2,
    min_scale: float = 0.05,
    drop_gamma: float = 1.0,
):
    """Adjust disturbance scale from completed and early-dropped episodes.

    The scalar is global, while the statistics and delta use only the rows
    whose episodes are actually being reset.  This mirrors mjlab's selected
    reset contract and prevents live episodes from being counted as drops.
    """
    state = task(env)
    rows = _resolve_curriculum_env_ids(env, env_ids)
    if rows.numel() == 0:
        return {"value": float(state.adaptive)}
    episode_steps = env.episode_length_buf.float()[rows]
    progress = torch.clamp(episode_steps / max(float(target_steps), 1.0), max=1.0)
    reached = episode_steps >= float(target_steps)
    terminated = getattr(getattr(env, "termination_manager", None), "terminated", None)
    if terminated is None:
        terminated = torch.zeros_like(reached)
    else:
        terminated = terminated[rows]
    early_drop = terminated & ~reached
    delta = (
        reached.float() * float(inc_rate)
        - early_drop.float() * (1.0 - progress).pow(float(drop_gamma)) * float(dec_rate)
    ).sum() / max(float(env.num_envs), 1.0)
    state.adaptive = float(min(max(state.adaptive + float(delta), min_scale), 1.0))
    return {
        "value": state.adaptive,
        "mean_episode_steps": float(episode_steps.mean().item()),
        "mean_progress": float(progress.mean().item()),
        "reached_target_frac": float(reached.float().mean().item()),
        "early_drop_frac": float(early_drop.float().mean().item()),
        "target_steps": float(target_steps),
        "delta": float(delta),
    }


__all__ = [
    "CageDrop",
    "adaptive_episode_curriculum",
    "metric",
    "nan_detection",
    "reward",
    "success_curriculum",
]

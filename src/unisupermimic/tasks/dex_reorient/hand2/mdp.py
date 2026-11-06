"""Public Hand 2 MDP symbols and the differentiable SuperMimic adapter.

The manager terms in this package operate on live UniLab entities.  The
``Hand2SuperMimicMDP`` below is deliberately entity free: it consumes packed
physics state plus a flattened task memory and can therefore be used by a
world-model rollout on CPU, CUDA, or an MPS bridge.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping

import torch

from unisupermimic.rl.algos.supermimic.geometry import (
    matmul,
    matvec,
    so3_log,
)

from .analytic import (
    CONTACT_DIM,
    CONTEXT_DIM,
    GOAL_FRAME_DIM,
    NUM_JOINTS,
    OBS_DIM,
    STATE_DIM,
    ema_target,
    integrate,
    observation,
    pack_state,
    unpack_state,
)
from .memory import MEMORY_DIM, MEMORY_DIMS, flatten_memory, unflatten_memory
from .observations import feature
from .rewards import CageDrop, metric, reward


def _state(value: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Return a flat ``[q, qd, p, R, v, w]`` state tensor."""
    if isinstance(value, torch.Tensor):
        if value.shape[-1] != STATE_DIM:
            raise ValueError(f"Hand2 state must have width {STATE_DIM}, got {value.shape[-1]}")
        return value
    if len(value) != 6:
        raise ValueError("Hand2 state tuple must contain q, qd, p, R, v and w")
    return pack_state(*value[0:2], value[2], value[3], value[4], value[5])


def _parts(value: torch.Tensor | tuple[torch.Tensor, ...]):
    flat = _state(value)
    return unpack_state(flat)


def _clone_memory(memory: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.clone() for name, value in memory.items()}


def _rotation_angle(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Stable SO(3) angle used by the source Hand2 reward.

    ``acos(trace(R))`` has an unbounded derivative as two rotations become
    aligned.  The source implementation uses the skew part and ``atan2``;
    retaining that form is essential for differentiable imagination because a
    near-successful cube state is encountered frequently during training.
    """
    relative = matmul(first, second.transpose(-1, -2))
    vee = 0.5 * torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        -1,
    )
    sine = torch.linalg.vector_norm(vee, dim=-1)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.atan2(sine, cosine)


class Hand2SuperMimicMDP:
    """Differentiable Hand 2 dynamics context, observations, and rewards.

    This class follows the source ``DexMDP`` equations while keeping all
    mutable controller state in the caller-owned memory dictionary.  The
    environment adapter is responsible for collecting ``xi`` (46 values),
    auxiliary metadata (21 values), contacts, and the pre-step state.
    """

    num_joints = NUM_JOINTS
    context_dim = CONTEXT_DIM
    obs_dim = OBS_DIM
    # Hand2's actor receives only its 69D observation history.  A zero-width
    # goal makes this explicit for generic tooling; the custom Hand2 runner
    # does not use the G1 goal encoder.
    goal_frame_dim = 0
    max_step_reward = 15.0 / 50.0
    memory_dim = MEMORY_DIM
    state_dim = STATE_DIM
    delta_dim = 52
    contact_dim = CONTACT_DIM
    wm_input_dim = STATE_DIM + CONTEXT_DIM + 9 + NUM_JOINTS
    wm_output_dim = delta_dim + contact_dim

    def __init__(
        self,
        default_joint_pos: torch.Tensor,
        lower: torch.Tensor | None = None,
        upper: torch.Tensor | None = None,
        tag_pos: torch.Tensor | None = None,
        tag_rot: torch.Tensor | None = None,
        *,
        default_joint_vel: torch.Tensor | None = None,
        kp_default: torch.Tensor | None = None,
        kd_default: torch.Tensor | None = None,
        force_limit: torch.Tensor | None = None,
        effort_limit: torch.Tensor | None = None,
        gear: torch.Tensor | None = None,
        dt: float = 0.02,
        hold_steps: int = 13,
        switch_steps: int = 50,
        max_outside: float = 25.0,
        max_count: float = 37.5,
        hold_weight: float = 4.694,
        action_rate_weight: float = 1.0,
        action_hf_weight: float = 30.0,
        torque_weight: float = 24.0,
        cage_weight: float = 500.0,
        finger_collision_weight: float = 1.0,
        palm_detach_weight: float = 0.5,
        action_alpha: float = 0.5,
        warmup_time_s: float = 0.4,
        cage_low: torch.Tensor | None = None,
        cage_high: torch.Tensor | None = None,
        kinematics=None,
        cage_up_axis: int = 1,
        tip_offset: torch.Tensor | None = None,
        tip_scale: float = 5.0,
    ) -> None:
        if default_joint_pos.shape[-1] != NUM_JOINTS:
            raise ValueError("Hand2 default_joint_pos must have width 20")
        device = default_joint_pos.device
        self.default = default_joint_pos.detach().clone().reshape(-1).to(device=device)
        self.lower = (torch.full_like(self.default, -torch.inf) if lower is None else lower).detach().clone()
        self.upper = (torch.full_like(self.default, torch.inf) if upper is None else upper).detach().clone()
        self.tag_pos = torch.zeros(3, device=device) if tag_pos is None else tag_pos.detach().clone()
        self.tag_rot = torch.eye(3, device=device) if tag_rot is None else tag_rot.detach().clone()
        # The source privileged frame uses ``joint_vel_rel`` and therefore
        # subtracts the robot's configured default velocity.  Hand2's MJCF
        # currently uses zero defaults, but retaining the value explicitly is
        # necessary for exact parity when a deployment asset changes it.
        self.default_joint_vel = (
            torch.zeros(NUM_JOINTS, device=device)
            if default_joint_vel is None
            else default_joint_vel.detach().clone().to(device=device)
        )
        self.kp_default = torch.ones(NUM_JOINTS, device=device) if kp_default is None else kp_default.detach().clone()
        self.kd_default = torch.zeros(NUM_JOINTS, device=device) if kd_default is None else kd_default.detach().clone()
        self.force_limit = None if force_limit is None else force_limit.detach().clone()
        # ``force_limit`` is the actuator PD clamp.  The source reward only
        # normalizes torque when the simulator exposes a separate effort-limit
        # field; UniLab's MuJoCo bridge currently does not, so keep this
        # distinction explicit instead of silently dividing by forcerange.
        self.effort_limit = None if effort_limit is None else effort_limit.detach().clone()
        self.gear = torch.ones(NUM_JOINTS, device=device) if gear is None else gear.detach().clone()
        self.dt = float(dt)
        self.max_step_reward = 15.0 * self.dt
        self.hold_steps = int(hold_steps)
        self.switch_steps = int(switch_steps)
        self.max_outside = float(max_outside)
        self.max_count = float(max_count)
        self.hold_weight = float(hold_weight)
        self.action_rate_weight = float(action_rate_weight)
        self.action_hf_weight = float(action_hf_weight)
        self.torque_weight = float(torque_weight)
        self.cage_weight = float(cage_weight)
        self.finger_collision_weight = float(finger_collision_weight)
        self.palm_detach_weight = float(palm_detach_weight)
        self.action_alpha = float(action_alpha)
        self.warmup_steps = round(float(warmup_time_s) / max(self.dt, 1e-6))
        self.cage_low = None if cage_low is None else cage_low.detach().clone()
        self.cage_high = None if cage_high is None else cage_high.detach().clone()
        self.kinematics = kinematics
        self.cage_up_axis = int(cage_up_axis)
        self.tip_offset = None if tip_offset is None else tip_offset.detach().clone()
        self.tip_scale = float(tip_scale)
        self._noise = torch.cat(
            (torch.full((20,), 0.06, device=device), torch.full((20,), 0.0, device=device),
             torch.full((3,), 0.008, device=device), torch.full((6,), 0.05, device=device),
             torch.zeros(20, device=device))
        )
        self._normalization = torch.ones(CONTEXT_DIM, device=device)

    def to(self, device: str | torch.device) -> "Hand2SuperMimicMDP":
        device = torch.device(device)
        if device == self.default.device:
            return self
        result = copy.copy(self)
        for name in (
            "default", "lower", "upper", "tag_pos", "tag_rot", "kp_default", "kd_default",
            "default_joint_vel", "force_limit", "gear", "cage_low", "cage_high", "_noise", "_normalization",
            "tip_offset", "effort_limit",
        ):
            value = getattr(self, name)
            setattr(result, name, None if value is None else value.to(device))
        return result

    @staticmethod
    def flatten_memory(memory: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return flatten_memory(memory)

    @staticmethod
    def unflatten_memory(value: torch.Tensor) -> dict[str, torch.Tensor]:
        return unflatten_memory(value)

    @staticmethod
    def flatten_state(state: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
        return _state(state)

    @staticmethod
    def unflatten_state(value: torch.Tensor):
        return unpack_state(value)

    @staticmethod
    def world_model_targets(state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        q, qd, p, rotation, v, w = unpack_state(_state(state))
        q1, qd1, p1, rotation1, v1, w1 = unpack_state(_state(next_state))
        return torch.cat(
            (q1 - q, qd1 - qd, p1 - p, so3_log(matmul(rotation.transpose(-1, -2), rotation1)), v1 - v, w1 - w),
            -1,
        )

    @staticmethod
    def state_error(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return Hand2SuperMimicMDP.world_model_targets(target, predicted)

    def world_model_inputs(
        self,
        state: torch.Tensor,
        action_target: torch.Tensor,
        context: torch.Tensor,
        disturbance: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat((self.dynamics(state, context, disturbance), action_target), -1)

    @staticmethod
    def integrate_state(state: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return integrate(_state(state), delta)

    world_model_step = integrate_state

    def control_target(
        self,
        action: torch.Tensor,
        previous_target: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        age: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the source controller target and executed actuator target.

        Dex-SuperMimic keeps the EMA target ``c`` in task memory and gives the
        actuator the encoder-corrected value ``u = c - bias``.  Keeping this
        operation here prevents collection and imagination from accidentally
        using different target semantics.
        """
        target = ema_target(
            self.default,
            action,
            previous_target,
            lower=self.lower,
            upper=self.upper,
            age=age,
            warmup_steps=self.warmup_steps,
            alpha=self.action_alpha,
        )
        if bias is None:
            bias = torch.zeros_like(target)
        if bias.shape != target.shape:
            raise ValueError(
                f"Hand2 encoder bias must match target shape {tuple(target.shape)}, "
                f"got {tuple(bias.shape)}"
            )
        return target, target - bias

    def advance_memory(
        self,
        state: torch.Tensor,
        memory: Mapping[str, torch.Tensor] | torch.Tensor,
        action: torch.Tensor,
        target: torch.Tensor,
        cage: torch.Tensor,
        next_goal: torch.Tensor,
        *,
        soft: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Advance command state and the two-step controller history.

        The source runner's ordering is significant: command counters are
        updated from ``state`` first, then ``c1/c2`` and ``a1/a2`` are shifted
        and the episode age increments.  Centralising it makes supervised WM
        fitting and imagined policy rollouts use the exact same transition.
        """
        current = self._memory(memory)
        updated = self.command_step(state, current, next_goal, soft=soft)
        updated.update(
            c1=target,
            c2=current["c1"],
            a1=action,
            a2=current["a1"],
            cage=cage,
            age=current["age"] + 1.0,
        )
        return updated

    def _memory(self, memory: Mapping[str, torch.Tensor] | torch.Tensor) -> dict[str, torch.Tensor]:
        return unflatten_memory(memory) if isinstance(memory, torch.Tensor) else dict(memory)

    def obs(
        self,
        state: torch.Tensor,
        memory: Mapping[str, torch.Tensor] | torch.Tensor,
        auxiliary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the 69D source actor observation."""
        q, _qd, p, rotation, _v, _w = _parts(state)
        m = self._memory(memory)
        center = (self.lower + self.upper) * 0.5
        # Match Dex-SuperMimic's ``half_range + 1e-6`` exactly.  A clamp
        # changes the normalization for very narrow limits and can alter the
        # actor input even though both expressions remain finite.
        half = (self.upper - self.lower) * 0.5 + 1e-6
        qn = ((q - center) / half).clamp(-1.0, 1.0)
        c1 = ((m["c1"] - center) / half).clamp(-1.0, 1.0)
        pos = matvec(self.tag_rot.transpose(-1, -2), p - self.tag_pos)
        goal = m["goal"].reshape(*q.shape[:-1], 3, 3)
        error = matmul(
            matmul(
                matmul(self.tag_rot.transpose(-1, -2), rotation), goal.transpose(-1, -2)
            ),
            self.tag_rot,
        )
        return observation(qn, qn - c1, pos, error[..., 1:, :].flatten(-2), m["a2"])

    def add_noise(self, obs: torch.Tensor) -> torch.Tensor:
        """Inject the source policy noise and its 2% pose outliers."""
        q, e, p, r, a = obs.split((20, 20, 3, 6, 20), dim=-1)
        q = q + (torch.rand_like(q) * 2.0 - 1.0) * 0.06
        p = torch.where(
            torch.rand_like(p[..., :1]) < 0.02,
            torch.rand_like(p) - 0.5,
            p,
        )
        random_quat = torch.nn.functional.normalize(
            torch.randn(*r.shape[:-1], 4, device=r.device, dtype=r.dtype), dim=-1
        )
        from unisupermimic.rl.algos.supermimic.geometry import quat_to_mat

        random_rotation = quat_to_mat(random_quat)[..., 1:, :].flatten(-2)
        r = torch.where(
            torch.rand_like(r[..., :1]) < 0.02,
            random_rotation,
            r,
        )
        p = p + (torch.rand_like(p) * 2.0 - 1.0) * 0.008
        return torch.cat((q, e, p, r + torch.randn_like(r) * 0.05, a), dim=-1)

    def goal(self, frames: torch.Tensor | None = None, num_frames: int = 0) -> torch.Tensor:
        if num_frames:
            raise ValueError("Hand2 SuperMimic has no reference-frame goal encoder")
        device = self.default.device
        shape = () if frames is None else frames.shape
        return torch.zeros((*shape, 0), device=device)

    def normalize_context(self, context: torch.Tensor) -> torch.Tensor:
        if context.shape[-1] != CONTEXT_DIM:
            raise ValueError(f"Hand2 context must have width {CONTEXT_DIM}")
        return context

    def dynamics(self, state: torch.Tensor, context: torch.Tensor, disturbance: torch.Tensor) -> torch.Tensor:
        state = _state(state)
        if context.shape[-1] != CONTEXT_DIM or disturbance.shape[-1] != 9:
            raise ValueError("Hand2 dynamics expects context=46 and disturbance=9")
        return torch.cat((state, context, disturbance), dim=-1)

    def pd_torque(self, state: torch.Tensor, target: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        q, qd, *_ = _parts(state)
        kp, kd = self.kp_default, self.kd_default
        if context is not None:
            kp = kp * context[..., -40:-20]
            kd = kd * context[..., -20:]
        torque = self.gear * (kp * (target - q) + kd * qd)
        if self.force_limit is not None:
            torque = torque.clamp(-self.force_limit, self.force_limit)
        return torque

    def _cage(self, state: torch.Tensor, counter: torch.Tensor, soft: bool) -> tuple[torch.Tensor, torch.Tensor]:
        q, qd, p, _rotation, _v, _w = _parts(state)
        if self.kinematics is not None:
            body_pos = self.kinematics.forward(q, qd)[0]
            low = body_pos.amin(-2) - 0.01
            margin = p.new_full((3,), 0.01)
            margin[self.cage_up_axis] = 0.03
            high = body_pos.amax(-2) + margin
        else:
            low = self.cage_low if self.cage_low is not None else p.new_tensor((-0.20, -0.20, -0.20))
            high = self.cage_high if self.cage_high is not None else p.new_tensor((0.20, 0.20, 0.20))
        signed = torch.maximum(low - p, p - high).amax(-1, keepdim=True)
        outside = torch.sigmoid(signed / 0.002) if soft else (signed > 0).to(p.dtype)
        nxt = (outside * (counter + 1.0) + (1.0 - outside) * (counter - 0.5).clamp_min(0.0)).clamp_max(self.max_count)
        return outside * 4.0 * nxt / self.max_outside, nxt

    def tip_kinematics(self, state: torch.Tensor, auxiliary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return palm-frame fingertip positions/velocities for the analytic reward.

        The packed physical state stores the cube translation and velocity in
        the palm frame (see :meth:`Hand2ReorientCommand.state_tensor`).  The
        source ``HandKinematics`` returns FK velocities in that same frame,
        and ``DexMDP.rewards`` subtracts the palm-frame cube velocity directly.
        Keeping this helper in the palm frame avoids mixing a world-frame FK
        value with the relative state velocity during imagination.
        """
        q, qd, _p, _rotation, _v, _w = _parts(state)
        if self.kinematics is None:
            zeros = q.new_zeros((*q.shape[:-1], 5, 3))
            return zeros, zeros
        del auxiliary
        return self.kinematics.tips(self.kinematics.forward(q, qd))

    def privileged_tip_positions(self, state: torch.Tensor, auxiliary: torch.Tensor) -> torch.Tensor:
        """Return source critic fingertip body origins in env-relative world coordinates."""
        q, qd, *_ = _parts(state)
        if self.kinematics is None:
            return q.new_zeros((*q.shape[:-1], 5, 3))
        body_pos = self.kinematics.tip_body_positions(self.kinematics.forward(q, qd))
        palm_pos = auxiliary[..., :3]
        palm_rot = auxiliary[..., 3:12].reshape(*q.shape[:-1], 3, 3)
        return palm_pos[..., None, :] + torch.matmul(
            palm_rot[..., None, :, :], body_pos[..., :, :, None]
        ).squeeze(-1)

    def reward(
        self,
        state: torch.Tensor,
        memory: Mapping[str, torch.Tensor] | torch.Tensor,
        action: torch.Tensor,
        target: torch.Tensor | None = None,
        contact: torch.Tensor | None = None,
        torque: torch.Tensor | None = None,
        auxiliary: torch.Tensor | None = None,
        *,
        soft: bool = True,
    ) -> torch.Tensor:
        """Compute differentiable source reward from a predicted state."""
        m = self._memory(memory)
        q, _qd, _p, rotation, v, _w = _parts(state)
        goal = m["goal"].reshape(*q.shape[:-1], 3, 3)
        theta = _rotation_angle(rotation, goal)
        cage, _ = self._cage(state, m["cage"], soft)
        if torque is None:
            if target is None:
                target = m["c1"]
            torque = self.pd_torque(state, target)
        if contact is None:
            contact = q.new_zeros((*q.shape[:-1], CONTACT_DIM))
        if target is None:
            target = m["c1"]
        previous = m["a1"]
        terms = {
            "orientation_alignment": 15.0 * (1.0 - (theta - 0.2).clamp_min(0.0) / torch.pi).clamp_min(0.0),
            "hand_pose": -0.2 * (q - self.default).square().sum(-1),
            "action_rate": -self.action_rate_weight * (
                (action - previous).square().sum(-1)
                + (action - 2 * previous + m["a2"]).square().sum(-1)
            ),
            "action_hf": -self.action_hf_weight * (target - 2 * m["c1"] + m["c2"]).square().sum(-1) * (m["age"].squeeze(-1) >= 2),
            "torque": -self.torque_weight * (
                (torque / self.effort_limit.clamp_min(1e-3)).square().sum(-1)
                if self.effort_limit is not None
                else torque.square().sum(-1)
            ),
            "tip_slide": (
                -0.3
                * (
                    torch.linalg.vector_norm(
                        self.tip_kinematics(state, auxiliary)[1]
                        - v.unsqueeze(-2),
                        dim=-1,
                    )
                    * contact[..., :5]
                ).sum(-1)
                if self.kinematics is not None and auxiliary is not None
                else q.new_zeros(q.shape[:-1])
            ),
            "cage_escape": -self.cage_weight * cage.squeeze(-1),
            "finger_collision": -self.finger_collision_weight * contact[..., 7:].sum(-1),
            "hold_escalation": self.hold_weight * (m["window"] * m["within"] * m["reward_snapshot"]).squeeze(-1),
            "palm_detach": self.palm_detach_weight * (1.0 - contact[..., 5]) * contact[..., 6],
        }
        return sum(terms.values()) * self.dt

    def command_step(
        self,
        state: torch.Tensor,
        memory: Mapping[str, torch.Tensor] | torch.Tensor,
        next_goal: torch.Tensor,
        *,
        soft: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Advance hold/window counters and EMA controller memory."""
        m = self._memory(memory)
        rotation = _parts(state)[3]
        goal = m["goal"].reshape(*rotation.shape[:-2], 3, 3)
        err = _rotation_angle(rotation, goal).unsqueeze(-1)
        within = torch.sigmoid((0.2 - err) / 0.02) if soft else (err < 0.2).to(rotation.dtype)
        window = m["window"]
        approaching = 1.0 - window
        hold = approaching * within * (m["hold"] + 1.0) + window * m["hold"]
        succeeded = approaching * (torch.sigmoid((hold - self.hold_steps + 0.5) / 0.25) if soft else (hold >= self.hold_steps).to(rotation.dtype))
        window = window + (1.0 - window) * succeeded
        timer = (1.0 - succeeded) * m["window_timer"] + window
        count = (1.0 - succeeded) * (m["reward_hold"] + window * within)
        switch = window * (torch.sigmoid((timer - self.switch_steps + 0.5) / 0.25) if soft else (timer >= self.switch_steps).to(rotation.dtype))
        out = _clone_memory(m)
        out["goal"] = torch.where(switch > 0.5, next_goal.reshape_as(out["goal"]), m["goal"])
        out.update(
            hold=(1.0 - switch) * (1.0 - succeeded) * hold,
            goal_timer=(1.0 - switch) * (m["goal_timer"] + 1.0),
            window=(1.0 - switch) * window,
            window_timer=(1.0 - switch) * timer,
            reward_window=timer,
            reward_hold=(1.0 - switch) * count,
            reward_snapshot=count,
            within=within,
            count=m["count"] + succeeded,
        )
        return out

    def _critic_feature_tensors(
        self,
        state: torch.Tensor,
        memory: Mapping[str, torch.Tensor] | torch.Tensor,
        auxiliary: torch.Tensor,
        normalized_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Build critic features and their finite checks without a host sync.

        The runner can compile this tensor-only boundary and keep exception
        reporting in :meth:`_validate_critic_features`.  Returning the named
        inputs/intermediates preserves the detailed diagnostics on failure;
        the normal path needs only the aggregated scalar check.
        """
        m = self._memory(memory)
        q, qd, _p, _rotation, v, w = _parts(state)
        if auxiliary.shape[-1] < 21:
            auxiliary = torch.cat(
                (auxiliary, q.new_zeros((*q.shape[:-1], 21 - auxiliary.shape[-1]))), -1
            )
        palm_rot = auxiliary[..., 3:12].reshape(*q.shape[:-1], 3, 3)
        perturb_dir, perturb = auxiliary[..., 12:15], auxiliary[..., 15:21]
        checks = {"state": state, "context": normalized_context, "auxiliary": auxiliary}
        for name, value in m.items():
            checks[f"memory.{name}"] = value
        policy = self.obs(state, m, auxiliary)
        checks["policy"] = policy
        # Tip FK is not required by the environment contract; callers with a
        # kinematics implementation may pass a 15D tip slice in auxiliary.
        tips = (
            (self.privileged_tip_positions(state, auxiliary) - q.new_tensor((0.0, 0.0, 0.5)))
            .mul(self.tip_scale)
            .flatten(-2)
        )
        progress = torch.cat(
            (
                m["hold"] / self.hold_steps,
                m["reward_window"] / self.switch_steps,
                m["age"] / 2500.0,
                m["count"],
            ),
            -1,
        )
        palm_linear_velocity = matvec(palm_rot, v)
        palm_angular_velocity = matvec(palm_rot, w)
        checks["palm_linear_velocity"] = palm_linear_velocity
        checks["palm_angular_velocity"] = palm_angular_velocity
        current = torch.cat(
            (
                policy[..., 40:49],
                palm_linear_velocity,
                palm_angular_velocity,
                perturb_dir,
                perturb,
                progress,
                (m["cage"] / self.max_outside).clamp(max=1.0),
                normalized_context[..., :6],
                palm_rot[..., 1:, :].flatten(-2),
            ),
            -1,
        )
        checks["critic_current"] = current
        base = torch.cat((policy, q - self.default, qd - self.default_joint_vel, tips), -1)
        frame = torch.cat((base, current), -1)
        checks["critic_frame"] = frame
        if frame.shape[-1] != 165:
            raise RuntimeError(f"Hand2 critic frame must have width 165, got {frame.shape[-1]}")
        if normalized_context.shape[-1] != CONTEXT_DIM:
            raise ValueError(f"Hand2 context must have width {CONTEXT_DIM}")
        finite = torch.stack([torch.isfinite(value).all() for value in checks.values()]).all()
        return torch.cat((frame, normalized_context), -1), finite, checks

    @staticmethod
    def _validate_critic_features(finite: torch.Tensor, checks: Mapping[str, torch.Tensor]) -> None:
        """Synchronize once, locating the original offending field on failure."""
        if bool(finite):
            return
        for name, value in checks.items():
            if not bool(torch.isfinite(value).all()):
                finite_values = value.detach()[torch.isfinite(value)]
                max_value = (
                    float(finite_values.abs().max()) if finite_values.numel() else float("nan")
                )
                raise FloatingPointError(
                    f"Hand2 critic feature '{name}' contains NaN/Inf (finite_absmax={max_value})"
                )

    def critic_features(
        self,
        state: torch.Tensor,
        memory: Mapping[str, torch.Tensor] | torch.Tensor,
        auxiliary: torch.Tensor,
        normalized_context: torch.Tensor,
    ) -> torch.Tensor:
        """Return source privileged frame (165D) plus simulator context (46D)."""
        features, finite, checks = self._critic_feature_tensors(
            state, memory, auxiliary, normalized_context
        )
        self._validate_critic_features(finite, checks)
        return features

    def task_features(
        self,
        state: torch.Tensor,
        memory: Mapping[str, torch.Tensor] | torch.Tensor,
        auxiliary: torch.Tensor,
        normalized_context: torch.Tensor,
    ) -> torch.Tensor:
        """Return the source task context used by the failure head.

        ``TaskObservations.context`` in Dex-SuperMimic is ordered as
        ``[simulator_context(46), privileged_frame(165)]``.  The critic uses
        the opposite order, ``[frame, simulator_context]``.  Keep both
        layouts explicit: callers should use this method for FailureHead and
        :meth:`critic_features` for value estimation.
        """
        critic = self.critic_features(state, memory, auxiliary, normalized_context)
        frame = critic[..., :165]
        if normalized_context.shape[-1] != CONTEXT_DIM:
            raise ValueError(f"Hand2 context must have width {CONTEXT_DIM}")
        result = torch.cat((normalized_context, frame), -1)
        if result.shape[-1] != 211:
            raise RuntimeError(f"Hand2 task context must have width 211, got {result.shape[-1]}")
        return result

    def soft_termination(self, state: torch.Tensor, memory: Mapping[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((self._memory(memory)["cage"].squeeze(-1) - self.max_outside + 0.5) / 0.25)

    def valid_imagination_start(self, age: torch.Tensor, horizon: int, max_steps: int = 2500) -> torch.Tensor:
        return age.reshape(-1) + horizon < max_steps


__all__ = [
    "CageDrop", "CONTACT_DIM", "CONTEXT_DIM", "GOAL_FRAME_DIM", "MEMORY_DIM", "MEMORY_DIMS",
    "NUM_JOINTS", "OBS_DIM", "STATE_DIM", "Hand2SuperMimicMDP", "ema_target", "feature",
    "flatten_memory", "integrate", "metric", "observation", "pack_state", "reward",
    "unflatten_memory", "unpack_state",
]

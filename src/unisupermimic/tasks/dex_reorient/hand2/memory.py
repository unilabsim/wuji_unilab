"""Task memory state kept outside the differentiable world model."""

from __future__ import annotations

from dataclasses import dataclass

import torch

# This layout mirrors the memory captured by Dex-SuperMimic's ``SimReader``.
# Keep scalar entries as one-element trailing tensors so flattening works for
# both real environment batches and imagined batches without special cases.
MEMORY_DIMS: dict[str, int] = {
    "c1": 20,
    "c2": 20,
    "a1": 20,
    "a2": 20,
    "goal": 9,
    "hold": 1,
    "goal_timer": 1,
    "window": 1,
    "window_timer": 1,
    "reward_window": 1,
    "reward_hold": 1,
    "reward_snapshot": 1,
    "within": 1,
    "cage": 1,
    "age": 1,
    "count": 1,
}
MEMORY_DIM = sum(MEMORY_DIMS.values())


def flatten_memory(memory: dict[str, torch.Tensor]) -> torch.Tensor:
    """Pack task/controller memory into the stable 98D replay layout."""
    missing = [name for name in MEMORY_DIMS if name not in memory]
    if missing:
        raise KeyError(f"Hand2 memory is missing fields: {missing}")
    values = []
    for name, width in MEMORY_DIMS.items():
        value = memory[name]
        if value.shape[-1:] != (width,):
            raise ValueError(
                f"Hand2 memory field {name!r} must end in {width} values, got {tuple(value.shape)}"
            )
        values.append(value)
    return torch.cat(values, dim=-1)


def unflatten_memory(value: torch.Tensor) -> dict[str, torch.Tensor]:
    """Inverse of :func:`flatten_memory`, preserving arbitrary batch axes."""
    if value.shape[-1] != MEMORY_DIM:
        raise ValueError(f"Hand2 memory must have width {MEMORY_DIM}, got {value.shape[-1]}")
    chunks = value.split(tuple(MEMORY_DIMS.values()), dim=-1)
    return dict(zip(MEMORY_DIMS, chunks, strict=True))


def memory_from_command(command, action_manager=None) -> dict[str, torch.Tensor]:
    """Capture a command's dynamic fields for a replay transition.

    The helper intentionally accepts a small duck-typed command object so it
    can be used by both the MuJoCo and mjwarp environment adapters.  Values are
    cloned to prevent subsequent manager updates from mutating replay rows.
    """
    action = action_manager
    if action is None:
        action = getattr(command, "action_manager", None)
    def scalar(value):
        value = torch.as_tensor(value, device=command.goal.device)
        if value.ndim == 0:
            value = value.expand(command.goal.shape[0])
        return value.float().reshape(*value.shape, 1)
    if action is None:
        zeros = command.goal.new_zeros(command.goal.shape[0], 20)
        c1 = c2 = a1 = a2 = zeros
    else:
        term = action.get_term("joint_pos") if hasattr(action, "get_term") else action
        c1 = getattr(term, "target", getattr(term, "processed_action", None))
        a1 = getattr(action, "action", getattr(term, "raw", None))
        a2 = getattr(action, "prev_action", a1)
        c1 = torch.zeros_like(a1) if c1 is None else c1
        c2 = getattr(term, "previous_target", c1)
    encoder_bias = getattr(
        getattr(command, "robot", None),
        "data",
        None,
    )
    encoder_bias = getattr(encoder_bias, "encoder_bias_tensor", None)
    if encoder_bias is None:
        encoder_bias = c1.new_zeros(c1.shape)
    else:
        encoder_bias = torch.as_tensor(encoder_bias, device=c1.device, dtype=c1.dtype)
    goal = command.goal
    # ``goal`` is a quaternion in the manager; the MDP adapter converts it to
    # a matrix in the palm frame when a tag transform is available.
    if goal.shape[-1] == 4:
        from unisupermimic.rl.algos.supermimic.geometry import quat_to_mat
        goal = quat_to_mat(goal)
    if goal.shape[-1] == 9:
        goal = goal.reshape(*goal.shape[:-1], 9)
    # The live command stores an elapsed window counter in ``window``;
    # the source learner's memory field is the binary in-window flag.
    live_window = getattr(command, "window", 0)
    window_flag = (torch.as_tensor(live_window, device=command.goal.device) > 0).float()
    fields = {
        "c1": c1, "c2": c2, "a1": a1, "a2": a2,
        "u": c1 - encoder_bias,
        "bias": encoder_bias,
        "goal": goal,
        "hold": scalar(getattr(command, "hold", 0)),
        "goal_timer": scalar(getattr(command, "goal_timer", 0)),
        "window": scalar(window_flag),
        "window_timer": scalar(getattr(command, "window", 0)),
        "reward_window": scalar(
            getattr(command, "reward_window_timer", getattr(command, "window", 0))
        ),
        "reward_hold": scalar(
            getattr(command, "reward_hold_counter", getattr(command, "reward_timer", 0))
        ),
        "reward_snapshot": scalar(
            getattr(
                command,
                "reward_hold_counter_snapshot",
                getattr(command, "reward_timer", 0),
            )
        ),
        "within": scalar(getattr(command, "within", 0)),
        "cage": scalar(getattr(command, "cage_counter", 0)),
        "age": scalar(getattr(command, "episode_length_buf", 0)),
        "count": scalar(getattr(command, "goal_count", 0)),
    }
    return {name: value.detach().clone() for name, value in fields.items()}


@dataclass
class Hand2Memory:
    """Batch-shaped control and goal counters for imagined rollouts."""

    target: torch.Tensor
    previous_action: torch.Tensor
    age: torch.Tensor
    hold: torch.Tensor
    window: torch.Tensor

    @classmethod
    def zeros(cls, num_envs: int, device: torch.device | str = "cpu") -> "Hand2Memory":
        target = torch.zeros((num_envs, 20), device=device)
        return cls(
            target=target,
            previous_action=torch.zeros_like(target),
            age=torch.zeros(num_envs, dtype=torch.long, device=device),
            hold=torch.zeros(num_envs, dtype=torch.long, device=device),
            window=torch.zeros(num_envs, dtype=torch.long, device=device),
        )

    def advance(self, target: torch.Tensor, action: torch.Tensor) -> None:
        self.target.copy_(target)
        self.previous_action.copy_(action)
        self.age.add_(1)


__all__ = [
    "Hand2Memory",
    "MEMORY_DIMS",
    "MEMORY_DIM",
    "flatten_memory",
    "unflatten_memory",
    "memory_from_command",
]

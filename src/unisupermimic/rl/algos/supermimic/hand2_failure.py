"""Learned physical-failure head for Hand 2 imagined rollouts.

The source Dex-SuperMimic runner predicts real termination separately from
the contact/state world model.  Hand 2 keeps the same separation: the head
sees only the predicted physical state and simulator context, so imagined
rollouts never query a backend termination manager.
"""

from __future__ import annotations

import torch
from torch import nn

from unisupermimic.rl.algos.supermimic.networks import RunningNorm, mlp, run_mlp
from unisupermimic.tasks.dex_reorient.hand2.analytic import STATE_DIM

# The source Dex-SuperMimic failure head receives the complete task context:
# the 46-D simulator context plus the 165-D privileged task frame.  The
# 46-D simulator slice alone is sufficient for the dynamics model, but it
# omits goal/progress/cage/kinematic information needed to predict physical
# termination and changes the source head's input contract from 269 to 104.
TASK_CONTEXT_DIM = 211


class Hand2FailureHead(nn.Module):
    """Predict the probability of physical termination at a Hand 2 state."""

    input_dim = STATE_DIM + TASK_CONTEXT_DIM

    def __init__(self, hidden: tuple[int, ...] = (256, 128)) -> None:
        super().__init__()
        self.norm = RunningNorm(self.input_dim)
        # Dex-SuperMimic's FailureHead uses the shared default MLP
        # activation (ELU).  The world model is the only Hand2 head with
        # SiLU; using SiLU here silently changes the failure logits and its
        # contribution to imagined termination.
        self.net = mlp(self.input_dim, hidden, 1)

    def features(self, state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] != STATE_DIM:
            raise ValueError(f"Hand2 failure state must have width {STATE_DIM}")
        if context.shape[-1] != TASK_CONTEXT_DIM:
            raise ValueError(f"Hand2 failure context must have width {TASK_CONTEXT_DIM}")
        return torch.cat((state, context), dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"Hand2 failure features must have width {self.input_dim}")
        return run_mlp(self.net, self.norm(features)).squeeze(-1)

    def predict(self, state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Return logits while building features from state and context."""
        return self(self.features(state, context))


__all__ = ["Hand2FailureHead", "TASK_CONTEXT_DIM"]

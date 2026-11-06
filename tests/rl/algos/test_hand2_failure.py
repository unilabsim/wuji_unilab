"""Contract tests for the Hand 2 physical-failure head."""

from __future__ import annotations

import torch

from unisupermimic.rl.algos.supermimic.hand2_failure import Hand2FailureHead


def test_hand2_failure_head_shapes_and_gradients() -> None:
    head = Hand2FailureHead(hidden=(16,))
    state = torch.randn(4, 58, requires_grad=True)
    # FailureHead consumes source's complete 211-D task context (46-D
    # simulator context plus 165-D privileged frame).
    context = torch.randn(4, 211)
    logits = head.predict(state, context)
    assert logits.shape == (4,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
    loss.backward()
    assert torch.isfinite(state.grad).all()
    assert torch.isfinite(logits).all()


def test_hand2_failure_head_updates_running_statistics() -> None:
    head = Hand2FailureHead(hidden=(8,))
    state = torch.zeros(2, 58)
    context = torch.ones(2, 211)
    assert head.norm.count.item() == 0
    head.norm.update(head.features(state, context))
    assert head.norm.count.item() == 2

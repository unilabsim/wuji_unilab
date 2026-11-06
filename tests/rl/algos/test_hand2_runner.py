"""Focused tests for the Hand 2 world-model runner tensor boundary."""

from __future__ import annotations

import torch

from unisupermimic.rl.algos.supermimic.hand2_runner import Hand2SuperMimicRunner


def test_term_major_policy_history_unpacks_newest_frame() -> None:
    """UniLab concatenates each term's history before the next term."""
    widths = (20, 20, 3, 6, 20)
    blocks = []
    expected = []
    for width, offset in zip(widths, (0, 100, 200, 300, 400), strict=True):
        block = torch.arange(3 * width, dtype=torch.float32).reshape(1, 3, width) + offset
        blocks.append(block.flatten(-2))
        expected.append(block[:, -1])
    packed = torch.cat(blocks, dim=-1)

    frame = Hand2SuperMimicRunner._single_frame_observation(packed)

    torch.testing.assert_close(frame, torch.cat(expected, dim=-1))

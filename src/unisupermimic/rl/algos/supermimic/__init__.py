"""SuperMimic world-model reinforcement learning.

Hand2's runner is imported lazily.  Task manager modules import the geometry
submodule during environment registration; eagerly importing the runner there
would recurse back into the Hand2 MDP and observations modules.
"""

from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig, SuperMimicLearner


def __getattr__(name: str):
    if name in {"Hand2SuperMimicConfig", "Hand2SuperMimicRunner"}:
        from unisupermimic.rl.algos.supermimic.hand2_runner import (
            Hand2SuperMimicConfig,
            Hand2SuperMimicRunner,
        )

        return {"Hand2SuperMimicConfig": Hand2SuperMimicConfig, "Hand2SuperMimicRunner": Hand2SuperMimicRunner}[name]
    raise AttributeError(name)

__all__ = [
    "Hand2SuperMimicConfig",
    "Hand2SuperMimicRunner",
    "SuperMimicConfig",
    "SuperMimicLearner",
]

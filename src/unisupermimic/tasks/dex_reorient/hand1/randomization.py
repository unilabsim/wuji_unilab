"""Hand 1 randomization entry points.

The first migration slice leaves randomization to UniLab's public Entity
bindings.  These helpers provide explicit ranges for downstream event terms.
"""

DEFAULT_RANGES = {
    "friction": (0.7, 1.3),
    "cube_mass": (0.4, 1.6),
    "cube_size": (0.85, 1.15),
    "kp": (0.75, 1.5),
    "kd": (0.5, 2.0),
    "encoder": (-0.01, 0.01),
}

__all__ = ["DEFAULT_RANGES"]

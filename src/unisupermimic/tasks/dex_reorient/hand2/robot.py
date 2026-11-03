"""Stable Hand 2 joint/body names used by config and diagnostics."""

JOINT_NAMES = tuple(
    f"robot/right_finger{finger}_joint{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
)
ACTUATOR_NAMES = tuple(f"{name}_actuator" for name in JOINT_NAMES)
PALM_BODY = "robot/right_palm_link"
TAG_BODY = "robot/right_wrist_tag_frame"
TIP_BODIES = tuple(f"robot/right_finger{finger}_link4" for finger in range(1, 6))
TIP_SITES = tuple(f"robot/right_finger{finger}_tip" for finger in range(1, 6))

__all__ = ["ACTUATOR_NAMES", "JOINT_NAMES", "PALM_BODY", "TAG_BODY", "TIP_BODIES", "TIP_SITES"]

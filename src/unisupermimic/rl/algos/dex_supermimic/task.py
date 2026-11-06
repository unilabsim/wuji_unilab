"""Task identities carried by source-format Dex SuperMimic checkpoints."""

from typing import Literal

TaskName = Literal["hand1-reorient", "hand2-reorient", "hand2-pen-spin"]
DEFAULT_TASK: TaskName = "hand2-reorient"


def checkpoint_task(config) -> TaskName:
    task = config.get("task", DEFAULT_TASK)
    if task not in ("hand1-reorient", "hand2-reorient", "hand2-pen-spin"):
        raise ValueError(f"Unknown checkpoint task: {task!r}")
    return task

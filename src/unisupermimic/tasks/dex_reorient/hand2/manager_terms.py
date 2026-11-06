"""Compatibility facade grouping Hand 2 Manager terms by function."""

from .actions import WujiAction, WujiActionCfg
from .commands import Hand2ReorientCommand, Hand2ReorientCommandCfg, task
from .observations import TipsFeature, feature
from .rewards import CageDrop, metric, nan_detection, reward

__all__ = [
    "CageDrop", "Hand2ReorientCommand", "Hand2ReorientCommandCfg", "TipsFeature",
    "WujiAction", "WujiActionCfg", "feature", "metric", "nan_detection", "reward", "task",
]

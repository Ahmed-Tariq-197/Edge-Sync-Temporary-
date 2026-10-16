"""EdgeSync Perception Interfaces."""
from .base_detector import BaseDetector, Detection
from .base_tracker import BaseTracker, Track

__all__ = ['BaseDetector', 'Detection', 'BaseTracker', 'Track']

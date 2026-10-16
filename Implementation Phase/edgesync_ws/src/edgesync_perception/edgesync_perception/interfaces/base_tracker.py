"""Abstract Base Class for Multi-Object Tracking (SOLID: DIP & OCP)."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
from .base_detector import Detection


@dataclass
class Track:
    """Represents an active multi-object track."""
    track_id: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float
    is_activated: bool = True

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)


class BaseTracker(ABC):
    """Abstract interface for all multi-object tracking backends."""

    @abstractmethod
    def update(self, detections: List[Detection]) -> List[Track]:
        """Update tracker state with new frame detections.

        Args:
            detections: List of 2D bounding box detections for the current frame.

        Returns:
            List of confirmed active Tracks.
        """
        pass

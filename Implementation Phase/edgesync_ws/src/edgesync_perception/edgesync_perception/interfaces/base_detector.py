"""Abstract Base Class for Object Detection (SOLID: DIP & OCP)."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class Detection:
    """Represents a 2D bounding box detection."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float
    class_id: int
    class_name: str = "person"

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Return (xmin, ymin, xmax, ymax)."""
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)


class BaseDetector(ABC):
    """Abstract interface for all perception detector backends."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Execute 2D object detection on an input RGB image frame.

        Args:
            image: HxWx3 uint8 numpy array in RGB format.

        Returns:
            List of validated Detection objects.
        """
        pass

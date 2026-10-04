"""Abstract Base Class for Re-Identification Feature Extraction (SOLID: DIP & OCP)."""
from abc import ABC, abstractmethod
import numpy as np


class BaseIdentityExtractor(ABC):
    """Abstract interface for visual appearance Re-ID feature extractors."""

    @abstractmethod
    def extract_embedding(self, image_crop: np.ndarray) -> np.ndarray:
        """Extract a 512-dimensional L2-normalized feature embedding from an image crop.

        Args:
            image_crop: HxWx3 uint8 numpy array in RGB format.

        Returns:
            np.ndarray: 512-element 1D float32 array with unit L2 norm (||v||_2 = 1.0).
        """
        pass

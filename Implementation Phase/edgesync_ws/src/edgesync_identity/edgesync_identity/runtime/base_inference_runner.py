"""Abstract inference runner (SOLID: DIP). Extractors depend on this, not on TensorRT."""
from abc import ABC, abstractmethod
import numpy as np


class BaseInferenceRunner(ABC):
    """Executes a single-input, single-output network on one preprocessed tensor."""

    @abstractmethod
    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run the network.

        Args:
            input_tensor: float32 array matching the engine input shape.

        Returns:
            1D float32 array holding the raw (not yet normalised) network output.
        """

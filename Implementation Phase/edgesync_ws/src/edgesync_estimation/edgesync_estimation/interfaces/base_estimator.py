"""Abstract Base Class for 3D Target State Estimation (SOLID: DIP & OCP)."""
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class BaseStateEstimator(ABC):
    """Abstract interface for target 3D kinematics filtering (e.g., EKF/UKF)."""

    @abstractmethod
    def predict(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Propagate state and covariance forward by dt seconds.

        Args:
            dt: Time delta in seconds.

        Returns:
            Tuple of (predicted_state [6,], predicted_covariance [6,6]).
        """
        pass

    @abstractmethod
    def update(self, measurement_map: np.ndarray, r_total: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Incorporate new 3D position measurement with total inflated covariance.

        Args:
            measurement_map: [x, y, z] target position in shared 'map' frame.
            r_total: 3x3 measurement covariance matrix (including ego-motion propagation).

        Returns:
            Tuple of (updated_state [6,], updated_covariance [6,6]).
        """
        pass

    @abstractmethod
    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return current state vector and 6x6 covariance matrix."""
        pass

    @abstractmethod
    def get_upper_triangular_covariance(self) -> np.ndarray:
        """Return 21-element 1D array of the upper-triangular symmetric covariance."""
        pass

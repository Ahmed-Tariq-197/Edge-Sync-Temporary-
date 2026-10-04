"""Abstract Base Class for Time-To-Occlusion (TTO) Prediction."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class TTOResult:
    """Encapsulates analytical Time-To-Occlusion computation results."""
    tto_hat_s: float                    # Nominal point estimate of TTO (seconds)
    sigma_tto_s: float                  # Uncertainty standard deviation in TTO (seconds)
    tto_safe_s: float                   # Conservative lower bound: max(0, tto_hat - k_tau * sigma)
    exit_boundary: str                  # 'horizontal_left', 'horizontal_right', 'vertical', 'range_max', 'obstacle'
    predicted_trajectory: np.ndarray    # Nx3 forward projected coordinates in map frame


class BaseTTOPredictor(ABC):
    """Abstract interface for analytical TTO prediction."""

    @abstractmethod
    def compute_tto(self, target_state: np.ndarray,
                    target_cov: np.ndarray,
                    camera_position_map: np.ndarray,
                    camera_orientation_matrix: np.ndarray,
                    obstacles: Optional[List] = None) -> TTOResult:
        """Compute conservative Time-To-Occlusion.

        Args:
            target_state: 6-DoF state [x, y, z, vx, vy, vz] in 'map' frame.
            target_cov: 6x6 covariance matrix in 'map' frame.
            camera_position_map: [x, y, z] camera origin in 'map' frame.
            camera_orientation_matrix: 3x3 rotation matrix from 'map' to camera optical frame.
            obstacles: Optional list of 2D/3D obstacle hulls.

        Returns:
            TTOResult with safe TTO lower bound and exit classification.
        """
        pass

"""Analytical Time-To-Occlusion (TTO) Predictor (Math/Protocol Spec v2.3 §1.3).

Forecasts when the target will leave the 3D FOV frustum or become occluded,
computing the uncertainty-aware conservative lower bound:
    T_TTO^safe = max(0, T_hat_TTO - k_tau * sigma_TTO)
"""
from typing import List, Optional, Tuple
import numpy as np
from ..interfaces.base_tto_predictor import BaseTTOPredictor, TTOResult


class AnalyticalTTOCalculator(BaseTTOPredictor):
    """Calculates 3D FOV frustum intersection and conservative TTO."""

    def __init__(self,
                 hfov_deg: float = 60.0,
                 vfov_deg: float = 49.5,
                 min_range_m: float = 0.40,
                 max_range_m: float = 5.50,
                 k_tau_safety: float = 2.0,
                 prediction_horizon_s: float = 4.0,
                 dt_step_s: float = 0.05):
        """Initialize TTO Calculator with optical parameters.

        Args:
            hfov_deg: Full horizontal field of view in degrees.
            vfov_deg: Full vertical field of view in degrees.
            min_range_m: Minimum sensing distance.
            max_range_m: Maximum sensing distance (effective tracking boundary).
            k_tau_safety: Multiplier for conservative uncertainty margin (planning value = 2.0).
            prediction_horizon_s: Maximum forward projection horizon.
            dt_step_s: Discrete integration step size.
        """
        self.alpha_fov = np.radians(hfov_deg / 2.0)  # Horizontal half-angle
        self.beta_fov = np.radians(vfov_deg / 2.0)   # Vertical half-angle
        self.min_range_m = min_range_m
        self.max_range_m = max_range_m
        self.k_tau_safety = k_tau_safety
        self.prediction_horizon_s = prediction_horizon_s
        self.dt_step_s = dt_step_s

    def is_in_frustum(self, point_camera: np.ndarray) -> Tuple[bool, str]:
        """Check if 3D camera-frame point lies strictly inside the frustum."""
        xc, yc, zc = float(point_camera[0]), float(point_camera[1]), float(point_camera[2])

        if zc <= self.min_range_m:
            return False, "range_min"
        if zc >= self.max_range_m:
            return False, "range_max"

        norm_xy = np.sqrt(xc ** 2 + zc ** 2)
        if norm_xy > self.max_range_m:
            return False, "range_max"

        azimuth = np.arctan2(xc, zc)
        if azimuth > self.alpha_fov:
            return False, "horizontal_right"
        if azimuth < -self.alpha_fov:
            return False, "horizontal_left"

        elevation = np.arctan2(yc, zc)
        if elevation > self.beta_fov:
            return False, "vertical_bottom"
        if elevation < -self.beta_fov:
            return False, "vertical_top"

        return True, "inside"

    def compute_tto(self, target_state: np.ndarray,
                    target_cov: np.ndarray,
                    camera_position_map: np.ndarray,
                    camera_orientation_matrix: np.ndarray,
                    obstacles: Optional[List] = None) -> TTOResult:
        """Compute conservative Time-To-Occlusion.

        Args:
            target_state: [x, y, z, vx, vy, vz] in 'map' frame.
            target_cov: 6x6 covariance matrix in 'map' frame.
            camera_position_map: [x, y, z] position of camera optical center in 'map'.
            camera_orientation_matrix: 3x3 rotation matrix R_map^cam.
            obstacles: Optional obstacle polygons/hulls.

        Returns:
            TTOResult with nominal T_hat, uncertainty sigma, and conservative T_safe.
        """
        pos0 = target_state[:3]
        vel0 = target_state[3:]
        speed = float(np.linalg.norm(vel0))

        # Discretize horizon
        num_steps = int(self.prediction_horizon_s / self.dt_step_s) + 1
        times = np.linspace(0.0, self.prediction_horizon_s, num_steps)

        # Forward project trajectory in map frame: p_m(t) = pos0 + t * vel0
        traj_map = pos0[None, :] + times[:, None] * vel0[None, :]

        # Transform entire trajectory to camera optical frame: p_c = R_map^cam * (p_m - t_cam)
        diff_map = traj_map - camera_position_map[None, :]
        traj_cam = (camera_orientation_matrix @ diff_map.T).T

        t_hat = self.prediction_horizon_s
        exit_reason = "horizon_limit"

        # Check initial point
        in_frustum, reason = self.is_in_frustum(traj_cam[0])
        if not in_frustum:
            return TTOResult(
                tto_hat_s=0.0,
                sigma_tto_s=0.0,
                tto_safe_s=0.0,
                exit_boundary=reason,
                predicted_trajectory=traj_map
            )

        # Find earliest step where trajectory exits frustum
        for i in range(1, num_steps):
            in_frustum, reason = self.is_in_frustum(traj_cam[i])
            if not in_frustum:
                t_hat = float(times[i])
                exit_reason = reason
                break

        # Uncertainty propagation: sigma_tto
        # Project spatial covariance along target velocity direction
        if speed > 0.05:
            v_dir = vel0 / speed
            sigma_pos_along_v = float(np.sqrt(max(1e-6, v_dir @ target_cov[:3, :3] @ v_dir)))
            sigma_vel_along_v = float(np.sqrt(max(1e-6, v_dir @ target_cov[3:, 3:] @ v_dir)))
            sigma_tto = (sigma_pos_along_v + t_hat * sigma_vel_along_v) / speed
        else:
            # Stationary target: uncertainty bounded by sensor range uncertainty
            sigma_tto = 0.10

        sigma_tto = float(np.clip(sigma_tto, 0.02, 1.0))
        t_safe = float(max(0.0, t_hat - self.k_tau_safety * sigma_tto))

        return TTOResult(
            tto_hat_s=round(t_hat, 3),
            sigma_tto_s=round(sigma_tto, 3),
            tto_safe_s=round(t_safe, 3),
            exit_boundary=exit_reason,
            predicted_trajectory=traj_map
        )

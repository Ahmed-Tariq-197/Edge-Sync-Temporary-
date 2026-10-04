"""Pure mathematical module for Ego-Motion Covariance Propagation (Math/Protocol Spec v2.3 §1.2).

Computes Jacobians J_c and J_xi and projects measurement noise into 'map' frame:
    R_total = J_c * R_cam * J_c^T + J_xi * Sigma_xi * J_xi^T
Guarantees zero double-counting of ego-motion noise with tf2.
"""
from typing import Tuple
import numpy as np


class EgoCovariancePropagator:
    """Computes total measurement covariance R_total accounting for robot chassis uncertainty."""

    @staticmethod
    def compute_jacobians(point_camera: np.ndarray,
                          r_cam_to_base: np.ndarray,
                          t_cam_to_base: np.ndarray,
                          robot_yaw: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute observation Jacobian J_c and pose Jacobian J_xi.

        Args:
            point_camera: [X_c, Y_c, Z_c] in camera optical frame.
            r_cam_to_base: 3x3 rotation matrix from camera optical frame to robot base_link.
            t_cam_to_base: 3-element translation vector from base_link to camera.
            robot_yaw: Robot orientation yaw theta_r in 'map' frame (radians).

        Returns:
            Tuple of (J_c [3,3], J_xi [3,3]).
        """
        # Point expressed in base_link frame: p_b = R_cam^base * p_c + t_cam^base
        p_b = r_cam_to_base @ point_camera + t_cam_to_base
        xb, yb = float(p_b[0]), float(p_b[1])

        cos_yaw = float(np.cos(robot_yaw))
        sin_yaw = float(np.sin(robot_yaw))

        # Base to map rotation matrix R_base^map for 2D planar motion
        r_base_to_map = np.array([
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw,  cos_yaw, 0.0],
            [0.0,      0.0,     1.0]
        ], dtype=np.float64)

        # 1. Observation Jacobian: J_c = d(p_m)/d(p_c) = R_base^map * R_cam^base
        j_c = r_base_to_map @ r_cam_to_base

        # 2. Pose Jacobian: J_xi = d(p_m)/d([x_r, y_r, theta_r])
        dx_dtheta = -xb * sin_yaw - yb * cos_yaw
        dy_dtheta =  xb * cos_yaw - yb * sin_yaw

        j_xi = np.array([
            [1.0, 0.0, dx_dtheta],
            [0.0, 1.0, dy_dtheta],
            [0.0, 0.0, 0.0]
        ], dtype=np.float64)

        return j_c, j_xi

    @classmethod
    def propagate_covariance(cls, point_camera: np.ndarray,
                             r_cam: np.ndarray,
                             sigma_robot_pose: np.ndarray,
                             r_cam_to_base: np.ndarray,
                             t_cam_to_base: np.ndarray,
                             robot_yaw: float,
                             enable_ego_propagation: bool = True) -> np.ndarray:
        """Propagate camera observation covariance and robot pose uncertainty to map frame.

        Args:
            point_camera: [X_c, Y_c, Z_c] in camera frame.
            r_cam: 3x3 measurement covariance in camera optical frame.
            sigma_robot_pose: 3x3 covariance of robot pose [x, y, yaw] in map.
            r_cam_to_base: 3x3 extrinsic rotation matrix.
            t_cam_to_base: 3-element extrinsic translation vector.
            robot_yaw: Robot heading in map (radians).
            enable_ego_propagation: If False, skips ego-pose inflation (ablation baseline).

        Returns:
            np.ndarray: 3x3 total measurement covariance R_total in 'map' frame.
        """
        j_c, j_xi = cls.compute_jacobians(point_camera, r_cam_to_base, t_cam_to_base, robot_yaw)

        # Camera observation noise rotated into map: J_c * R_cam * J_c^T
        r_cam_map = j_c @ r_cam @ j_c.T

        if not enable_ego_propagation or sigma_robot_pose is None:
            r_total = r_cam_map
        else:
            # Ego-motion noise: J_xi * Sigma_xi * J_xi^T
            r_ego = j_xi @ sigma_robot_pose @ j_xi.T
            r_total = r_cam_map + r_ego

        # Enforce exact mathematical symmetry
        r_total = 0.5 * (r_total + r_total.T)
        return r_total

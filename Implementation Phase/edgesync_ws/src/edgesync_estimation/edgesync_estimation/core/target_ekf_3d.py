"""6-DoF Target Extended Kalman Filter (EKF) in shared 'map' frame.

Math/Protocol Spec v2.3 §1.1 & §1.2:
- 6-DoF constant velocity state: s = [x, y, z, vx, vy, vz]^T
- Target acceleration noise Q_target (no chassis noise in Q)
- Measurement covariance R_total incorporates propagated ego-motion noise
- Joseph stabilized covariance update for numerical robustness (positive semi-definite)
- Upper triangular 21-element extraction for ROS 2 messages
"""
from typing import Tuple, Optional
import numpy as np
from ..interfaces.base_estimator import BaseStateEstimator


class TargetEKF3D(BaseStateEstimator):
    """Linear 6-DoF Constant Velocity Kalman Filter for 3D Target Tracking."""

    def __init__(self,
                 target_id: str = "target_human",
                 shared_frame: str = "map",
                 initial_pos: Optional[np.ndarray] = None,
                 q_accel: float = 1.0,
                 init_pos_var: float = 0.25,
                 init_vel_var: float = 1.0):
        """Initialize 6-DoF EKF.

        Args:
            target_id: Identifier string for tracked subject.
            shared_frame: Common world coordinate frame ("map").
            initial_pos: Optional initial [x, y, z] position.
            q_accel: Target acceleration power spectral density (m^2/s^3).
            init_pos_var: Initial position variance prior (m^2).
            init_vel_var: Initial velocity variance prior ((m/s)^2).
        """
        self.target_id = target_id
        self.shared_frame = shared_frame
        self.q_accel = q_accel

        # State vector: [x, y, z, vx, vy, vz]^T
        self.x = np.zeros(6, dtype=np.float64)
        if initial_pos is not None:
            self.x[:3] = initial_pos

        # State covariance matrix P (6x6)
        self.P = np.diag([
            init_pos_var, init_pos_var, init_pos_var,
            init_vel_var, init_vel_var, init_vel_var
        ]).astype(np.float64)

        # Measurement matrix H (3x6) mapping state to 3D position [x, y, z]
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[:3, :3] = np.eye(3, dtype=np.float64)

        self.last_update_stamp_ns: int = 0
        self.is_initialized: bool = initial_pos is not None

    def _compute_f_and_q(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute state transition matrix F and continuous white noise Q."""
        f = np.eye(6, dtype=np.float64)
        f[0, 3] = dt
        f[1, 4] = dt
        f[2, 5] = dt

        dt2 = (dt ** 2) / 2.0
        dt3 = (dt ** 3) / 3.0

        q_block_pos = dt3 * self.q_accel * np.eye(3, dtype=np.float64)
        q_block_cross = dt2 * self.q_accel * np.eye(3, dtype=np.float64)
        q_block_vel = dt * self.q_accel * np.eye(3, dtype=np.float64)

        q = np.zeros((6, 6), dtype=np.float64)
        q[:3, :3] = q_block_pos
        q[:3, 3:] = q_block_cross
        q[3:, :3] = q_block_cross
        q[3:, 3:] = q_block_vel
        return f, q

    def predict(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Propagate state and covariance forward by dt seconds."""
        if dt <= 0.0:
            return self.x.copy(), self.P.copy()

        f, q = self._compute_f_and_q(dt)
        self.x = f @ self.x
        self.P = f @ self.P @ f.T + q

        # Enforce exact symmetry
        self.P = 0.5 * (self.P + self.P.T)
        return self.x.copy(), self.P.copy()

    def update(self, measurement_map: np.ndarray, r_total: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update state using 3D measurement in map frame and inflated R_total."""
        z = np.asarray(measurement_map, dtype=np.float64).reshape(3)

        if not self.is_initialized:
            self.x[:3] = z
            self.x[3:] = 0.0
            self.P[:3, :3] = r_total
            self.is_initialized = True
            return self.x.copy(), self.P.copy()

        # Innovation: y = z - H * x
        y = z - (self.H @ self.x)

        # Innovation covariance: S = H * P * H^T + R_total
        s = self.H @ self.P @ self.H.T + r_total
        s = 0.5 * (s + s.T)

        # Kalman gain: K = P * H^T * S^-1
        k = self.P @ self.H.T @ np.linalg.inv(s)

        # State update
        self.x = self.x + (k @ y)

        # Joseph stabilized covariance update: P = (I - KH) * P * (I - KH)^T + K * R_total * K^T
        i_kh = np.eye(6, dtype=np.float64) - (k @ self.H)
        self.P = (i_kh @ self.P @ i_kh.T) + (k @ r_total @ k.T)
        self.P = 0.5 * (self.P + self.P.T)

        return self.x.copy(), self.P.copy()

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (state [6,], covariance [6,6])."""
        return self.x.copy(), self.P.copy()

    def get_upper_triangular_covariance(self) -> np.ndarray:
        """Extract upper triangular 21 floats from symmetric 6x6 covariance matrix."""
        indices = np.triu_indices(6)
        return self.P[indices].astype(np.float32)

    @classmethod
    def reconstruct_covariance_from_upper_triangle(cls, ut: np.ndarray) -> np.ndarray:
        """Helper to reconstruct full 6x6 symmetric covariance from 21 elements."""
        p = np.zeros((6, 6), dtype=np.float64)
        indices = np.triu_indices(6)
        p[indices] = ut
        # Symmetrize
        p = p + p.T - np.diag(np.diag(p))
        return p

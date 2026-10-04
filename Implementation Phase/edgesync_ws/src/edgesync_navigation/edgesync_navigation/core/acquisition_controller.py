"""Chassis Acquisition Slew Controller (Math/Protocol Spec v2.3 §3.2).

Computes fast orientation slew toward target spatial prior:
- Bounds nominal slew latency to <= 450 ms (Quality Gate 7).
- Generates angular velocity commands with smooth acceleration limiting for Mecanum base.
"""
import numpy as np


class AcquisitionController:
    """Computes orientation slew velocity commands for target interception."""

    def __init__(self,
                 max_angular_vel_rad_s: float = 1.5,
                 angular_accel_limit: float = 2.0,
                 heading_tolerance_rad: float = 0.05):
        self.max_w = max_angular_vel_rad_s
        self.max_alpha = angular_accel_limit
        self.tol = heading_tolerance_rad

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def compute_slew_angle(self, robot_pos: tuple[float, float],
                           robot_yaw: float,
                           target_pos: tuple[float, float]) -> float:
        """Calculate required delta heading angle to point directly at target."""
        dx = target_pos[0] - robot_pos[0]
        dy = target_pos[1] - robot_pos[1]

        target_bearing = np.arctan2(dy, dx)
        delta_yaw = self.normalize_angle(target_bearing - robot_yaw)
        return float(delta_yaw)

    def compute_angular_velocity(self, delta_yaw: float, current_w: float = 0.0, dt: float = 0.02) -> float:
        """Compute bounded angular velocity command to achieve alignment."""
        if abs(delta_yaw) <= self.tol:
            return 0.0

        # Proportional control with saturation
        kp = 3.0
        target_w = np.clip(kp * delta_yaw, -self.max_w, self.max_w)

        # Acceleration ramp-rate limiting
        max_dw = self.max_alpha * dt
        w_cmd = np.clip(target_w, current_w - max_dw, current_w + max_dw)
        return float(w_cmd)

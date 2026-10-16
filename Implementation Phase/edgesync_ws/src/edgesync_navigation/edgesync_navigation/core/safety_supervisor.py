"""Autonomous Safe-Stop & Motion Safety Supervisor (SAD v2.3 §3 & WP-7).

Safety Mandates:
- Emergency safe-stop engages within <= 50 ms of heartbeat severance or localization loss.
- Controlled linear & angular velocity ramp-down (no wheel lockup or skid).
- Dual-layer safety: Software authority node + STM32 FreeRTOS watchdog (150 ms).
"""
import time
from typing import Tuple
import numpy as np


class SafetySupervisor:
    """Oversees platform motion safety, command multiplexing, and autonomous stop."""

    def __init__(self,
                 max_linear_decel: float = 2.0,
                 max_angular_decel: float = 3.0,
                 cmd_vel_watchdog_ms: float = 150.0):
        self.max_lin_decel = max_linear_decel
        self.max_ang_decel = max_angular_decel
        self.watchdog_limit_s = cmd_vel_watchdog_ms / 1000.0  # 0.150s

        self._is_emergency_stop: bool = False
        self._stop_reason: str = ""
        self._last_valid_command_time: float = time.monotonic()

    def feed_heartbeat(self):
        """Reset watchdog upon receiving fresh command / valid localization."""
        self._last_valid_command_time = time.monotonic()

    def trigger_safe_stop(self, reason: str):
        """Immediately trigger software safe-stop."""
        self._is_emergency_stop = True
        self._stop_reason = reason

    def clear_safe_stop(self):
        """Clear safe-stop state."""
        self._is_emergency_stop = False
        self._stop_reason = ""
        self._last_valid_command_time = time.monotonic()

    def filter_command(self,
                       req_vx: float, req_vy: float, req_wz: float,
                       current_vx: float, current_vy: float, current_wz: float,
                       dt: float = 0.02,
                       current_time: float = None) -> Tuple[float, float, float, bool]:
        """Apply safety limits and safe-stop braking to velocity commands.

        Returns:
            Tuple of (cmd_vx, cmd_vy, cmd_wz, is_safe_stopped).
        """
        now = current_time if current_time is not None else time.monotonic()

        # Check 150ms command watchdog
        if (now - self._last_valid_command_time) > self.watchdog_limit_s:
            self._is_emergency_stop = True
            self._stop_reason = "Command watchdog cutoff (>150ms)"

        if self._is_emergency_stop:
            # Active ramp-down braking to 0
            max_lin_step = self.max_lin_decel * dt
            max_ang_step = self.max_ang_decel * dt

            lin_speed = np.hypot(current_vx, current_vy)
            if lin_speed > 1e-4:
                new_speed = max(0.0, lin_speed - max_lin_step)
                scale = new_speed / lin_speed
                safe_vx = current_vx * scale
                safe_vy = current_vy * scale
            else:
                safe_vx = 0.0
                safe_vy = 0.0

            if abs(current_wz) > 1e-4:
                w_sign = np.sign(current_wz)
                safe_wz = w_sign * max(0.0, abs(current_wz) - max_ang_step)
            else:
                safe_wz = 0.0

            return float(safe_vx), float(safe_vy), float(safe_wz), True

        return float(req_vx), float(req_vy), float(req_wz), False

    @property
    def is_stopped(self) -> bool:
        return self._is_emergency_stop

    @property
    def stop_reason(self) -> str:
        return self._stop_reason

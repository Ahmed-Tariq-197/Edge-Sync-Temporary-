"""10 Hz Liveness Watchdog (Anti-Deadlock Invariant 3).

Math/Protocol Spec v2.3 §3.3:
- Armed upon receipt of HANDOFF_RESPONSE=ACCEPT.
- Monitors incoming AcqStatus.msg heartbeats at 10 Hz (period = 100 ms).
- 3 consecutive missed heartbeats (300 ms cutoff) triggers ERR_CANDIDATE_DEAD.
- Prevents owner hanging in COMMITTED/ACQUISITION/VERIFICATION on peer failure.
"""
from typing import Callable, Optional
import time


class LivenessWatchdog:
    """Anti-deadlock watchdog monitoring peer acquisition liveness."""

    def __init__(self,
                 heartbeat_period_ms: float = 100.0,
                 max_missed_heartbeats: int = 3,
                 on_timeout_callback: Optional[Callable[[], None]] = None):
        self.heartbeat_period_ms = heartbeat_period_ms
        self.max_missed = max_missed_heartbeats
        self.cutoff_duration_s = (heartbeat_period_ms * max_missed_heartbeats) / 1000.0  # 0.300s
        self.on_timeout = on_timeout_callback

        self._is_armed: bool = False
        self._last_heartbeat_time: float = 0.0
        self._missed_count: int = 0

    def arm(self):
        """Arm watchdog at the instant of ACCEPT."""
        self._is_armed = True
        self._last_heartbeat_time = time.monotonic()
        self._missed_count = 0

    def disarm(self):
        """Disarm watchdog upon transaction completion or cancellation."""
        self._is_armed = False
        self._missed_count = 0

    def feed(self):
        """Reset watchdog timer upon valid AcqStatus.msg heartbeat."""
        if self._is_armed:
            self._last_heartbeat_time = time.monotonic()
            self._missed_count = 0

    def check(self, current_time: Optional[float] = None) -> bool:
        """Periodic check (e.g. at 20-50 Hz). Returns True if timed out."""
        if not self._is_armed:
            return False

        now = current_time if current_time is not None else time.monotonic()
        elapsed = now - self._last_heartbeat_time

        if elapsed >= self.cutoff_duration_s:
            self._is_armed = False
            if self.on_timeout is not None:
                self.on_timeout()
            return True

        return False

    @property
    def is_armed(self) -> bool:
        return self._is_armed

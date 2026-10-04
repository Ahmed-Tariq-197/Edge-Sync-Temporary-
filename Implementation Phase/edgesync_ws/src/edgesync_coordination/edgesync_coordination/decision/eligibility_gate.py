"""Threshold-Gated Candidate Eligibility & Analytical Feasibility (Math/Protocol Spec v2.3 §2).

Computes Hard Gate G(j) and Feasibility:
    G(j) = 1 if CPU <= 90% and Battery >= 15% and Nav == Available and Link is Fresh
    Feasible(j) = (t_exec_hat + delta_t_safety <= T_TTO_safe)
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class PeerTelemetry:
    """Telemetry data received from peer robot via StateHeartbeat."""
    cpu_load_pct: float
    battery_pct: float
    nav_status: int           # 1 = AVAILABLE, 0 = DEGRADED
    last_heartbeat_age_ms: float
    rtt_ms: float = 10.0


class EligibilityGate:
    """Evaluates candidate admission gate G(j) and execution budget feasibility."""

    def __init__(self,
                 max_cpu_pct: float = 90.0,
                 min_battery_pct: float = 15.0,
                 max_telemetry_age_ms: float = 2500.0,
                 dt_exec_nominal_ms: float = 866.0,
                 dt_safety_ms: float = 364.0):
        self.max_cpu_pct = max_cpu_pct
        self.min_battery_pct = min_battery_pct
        self.max_telemetry_age_ms = max_telemetry_age_ms
        self.dt_exec_nominal_ms = dt_exec_nominal_ms
        self.dt_safety_ms = dt_safety_ms
        self.total_budget_ms = dt_exec_nominal_ms + dt_safety_ms  # 1230 ms

    def evaluate_hard_gate(self, peer: PeerTelemetry) -> Tuple[bool, str]:
        """Evaluate hard admission gate G(j). Unsafe candidates are filtered out."""
        if peer.cpu_load_pct > self.max_cpu_pct:
            return False, f"CPU load ({peer.cpu_load_pct:.1f}%) exceeds gate limit ({self.max_cpu_pct}%)"

        if peer.battery_pct < self.min_battery_pct:
            return False, f"Battery ({peer.battery_pct:.1f}%) below minimum ({self.min_battery_pct}%)"

        if peer.nav_status != 1:
            return False, "Navigation stack degraded or unavailable"

        if peer.last_heartbeat_age_ms > self.max_telemetry_age_ms:
            return False, f"Peer telemetry stale ({peer.last_heartbeat_age_ms:.0f} ms > {self.max_telemetry_age_ms} ms)"

        return True, "Hard gate passed"

    def evaluate_feasibility(self, tto_safe_s: float, est_acq_s: float = 0.60) -> Tuple[bool, str]:
        """Check if total execution time plus safety margin fits within remaining TTO_safe."""
        # Remaining budget requirement in seconds: ~1.23s
        required_time_s = (self.dt_exec_nominal_ms + self.dt_safety_ms) / 1000.0

        if tto_safe_s < required_time_s:
            return False, f"TTO safe ({tto_safe_s:.2f}s) < required budget ({required_time_s:.2f}s)"

        return True, "Feasibility confirmed"

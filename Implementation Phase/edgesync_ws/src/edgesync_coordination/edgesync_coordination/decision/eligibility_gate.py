"""Threshold-Gated Candidate Eligibility & Analytical Feasibility (Math/Protocol Spec v2.3 §2).

Computes Hard Gate G(j) and Feasibility:
    G(j) = 1 if CPU < 90% and Battery > 15% and Nav == Available (1) and Link is Fresh
    Feasible(j) = (T_TTO_safe > t_txn_ms)
"""
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class PeerTelemetry:
    """Telemetry data received from peer robot via StateHeartbeat or local status."""
    cpu_load_pct: float
    battery_pct: float
    nav_status: int           # 1 = AVAILABLE / READY, 0 = DEGRADED
    last_heartbeat_age_ms: float = 0.0
    rtt_ms: float = 10.0


class EligibilityGate:
    """Evaluates candidate admission gate G(j) and execution budget feasibility."""

    def __init__(self,
                 max_cpu_pct: float = 90.0,
                 min_battery_pct: float = 15.0,
                 max_telemetry_age_ms: float = 2500.0,
                 dt_exec_nominal_ms: float = 866.0,
                 dt_safety_ms: float = 364.0,
                 t_txn_ms: Optional[float] = None):
        """Initialize eligibility gate with zero-hardcoded thresholds.

        Args:
            max_cpu_pct: Maximum allowable CPU load (default: 90.0%).
            min_battery_pct: Minimum required battery level (default: 15.0%).
            max_telemetry_age_ms: Telemetry age timeout (default: 2500 ms).
            dt_exec_nominal_ms: Nominal execution latency.
            dt_safety_ms: Safety margin latency.
            t_txn_ms: Transaction budget in milliseconds (e.g. 886.0 ms).
        """
        self.max_cpu_pct = max_cpu_pct
        self.min_battery_pct = min_battery_pct
        self.max_telemetry_age_ms = max_telemetry_age_ms
        self.dt_exec_nominal_ms = dt_exec_nominal_ms
        self.dt_safety_ms = dt_safety_ms
        self.t_txn_ms = float(t_txn_ms) if t_txn_ms is not None else (dt_exec_nominal_ms + dt_safety_ms)
        self.total_budget_ms = self.t_txn_ms

    def evaluate_hard_gate(self, peer: PeerTelemetry) -> Tuple[bool, str]:
        """Evaluate hard admission gate G(j).

        Strict checks:
        1. CPU load must be < max_cpu_pct (90%)
        2. Battery must be > min_battery_pct (15%)
        3. Navigation status must be AVAILABLE/READY (1)
        4. Telemetry must be fresh (<= max_telemetry_age_ms)
        """
        if peer.cpu_load_pct >= self.max_cpu_pct:
            return False, f"CPU load ({peer.cpu_load_pct:.1f}%) exceeds gate limit ({self.max_cpu_pct}%)"

        if peer.battery_pct <= self.min_battery_pct:
            return False, f"Battery ({peer.battery_pct:.1f}%) below minimum ({self.min_battery_pct}%)"

        if peer.nav_status != 1:
            return False, "Navigation stack degraded or unavailable"

        if peer.last_heartbeat_age_ms > self.max_telemetry_age_ms:
            return False, f"Peer telemetry stale ({peer.last_heartbeat_age_ms:.0f} ms > {self.max_telemetry_age_ms} ms)"

        return True, "Hard gate passed"

    def evaluate_feasibility(self, tto_safe_s: float, est_acq_s: float = 0.60) -> Tuple[bool, str]:
        """Check if remaining TTO_safe exceeds transaction budget.

        T_TTO_safe MUST be strictly greater than t_txn_ms (budget).
        """
        required_time_s = self.t_txn_ms / 1000.0

        if tto_safe_s <= required_time_s:
            return False, f"TTO safe ({tto_safe_s:.3f}s) <= required budget ({required_time_s:.3f}s)"

        return True, "Feasibility confirmed"

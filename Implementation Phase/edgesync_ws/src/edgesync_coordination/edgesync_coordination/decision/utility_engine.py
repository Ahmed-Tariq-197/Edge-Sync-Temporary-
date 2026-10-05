"""Threshold-Gated Multi-Candidate Utility Engine (Math/Protocol Spec v2.3 §2.2 & Sprint 4).

Formulates candidate rank score:
1. Candidate Receiver Evaluation (Sprint 4):
   U_j = W_fov * S_fov + W_dist * S_dist + W_battery * S_battery
   Weights: (W_fov=0.5, W_dist=0.3, W_battery=0.2)
   - FOV centrality S_fov: proximity to optical axis in camera frame.
   - Proximity S_dist: inverse of Euclidean distance to target.
   - Battery S_battery: normalized battery percentage [0, 1].

2. Sender Multi-Candidate Ranking:
   U_j = G(j) * Feasible(j) * [w1 * Phi_feas(j) + w2 * Phi_FOV(j) + w3 * Phi_net(j)]
"""
from typing import Optional, Tuple
import numpy as np
from .eligibility_gate import EligibilityGate, PeerTelemetry


class UtilityEngine:
    """Computes ranking utility score among eligible candidate receivers."""

    def __init__(self,
                 w_feasibility: float = 0.50,
                 w_fov: float = 0.30,
                 w_network: float = 0.20,
                 w_dist: Optional[float] = None,
                 w_battery: Optional[float] = None,
                 w_receiver_fov: Optional[float] = None,
                 w_receiver_dist: Optional[float] = None,
                 w_receiver_battery: Optional[float] = None,
                 t_ref_s: float = 1.0,
                 rtt_max_ms: float = 100.0,
                 max_fov_angle_rad: float = 0.5236,  # 30.0 degrees half-angle
                 d_ref_m: float = 1.0,
                 eligibility_gate: Optional[EligibilityGate] = None):
        """Initialize utility engine with zero hardcoded weights and reference values.

        Args:
            w_feasibility: Sender ranking feasibility weight (w1 = 0.5).
            w_fov: Sender ranking FOV weight (w2 = 0.3) or receiver FOV weight if w_dist passed.
            w_network: Sender ranking network weight (w3 = 0.2).
            w_dist: Receiver soft scoring Euclidean proximity weight (W_dist = 0.3).
            w_battery: Receiver soft scoring battery weight (W_battery = 0.2).
            w_receiver_fov: Optional explicit receiver FOV weight.
            w_receiver_dist: Optional explicit receiver distance weight.
            w_receiver_battery: Optional explicit receiver battery weight.
            t_ref_s: Reference margin time for feasibility scaling.
            rtt_max_ms: Max RTT for network quality scaling.
            max_fov_angle_rad: Camera half-HFOV angle in radians (default: 30 deg).
            d_ref_m: Reference distance for inverse proximity scaling (default: 1.0 m).
            eligibility_gate: Injected EligibilityGate instance.
        """
        # Sender multi-candidate weighting parameters (w1=0.5, w2=0.3, w3=0.2)
        self.w1 = w_feasibility
        self.w2 = 0.30 if (w_dist is not None or w_receiver_dist is not None) else w_fov
        self.w3 = w_network

        # Receiver soft-scoring weights (Sprint 4: W_fov=0.5, W_dist=0.3, W_battery=0.2)
        if w_receiver_fov is not None:
            self.w_fov = w_receiver_fov
        elif w_dist is not None:
            self.w_fov = w_fov
        else:
            self.w_fov = 0.50

        if w_receiver_dist is not None:
            self.w_dist = w_receiver_dist
        elif w_dist is not None:
            self.w_dist = w_dist
        else:
            self.w_dist = 0.30

        if w_receiver_battery is not None:
            self.w_battery = w_receiver_battery
        elif w_battery is not None:
            self.w_battery = w_battery
        else:
            self.w_battery = 0.20

        self.t_ref_s = t_ref_s
        self.rtt_max_ms = rtt_max_ms
        self.max_fov_angle_rad = max_fov_angle_rad
        self.d_ref_m = d_ref_m
        self.gate = eligibility_gate if eligibility_gate is not None else EligibilityGate()

    def compute_receiver_utility(self,
                                 point_camera: np.ndarray,
                                 battery_pct: float,
                                 distance: Optional[float] = None) -> Tuple[float, float, float, float]:
        """Compute candidate receiver utility U_j in [0.0, 1.0] (Sprint 4).

        Args:
            point_camera: [X_c, Y_c, Z_c] 3D coordinates in receiver's camera frame.
            battery_pct: Receiver battery level percentage [0.0 - 100.0].
            distance: Optional pre-computed Euclidean distance (meters).

        Returns:
            Tuple of (total_utility, s_fov, s_dist, s_battery).
        """
        xc = float(point_camera[0])
        yc = float(point_camera[1])
        zc = float(point_camera[2])

        # 1. FOV Centrality S_fov in [0, 1]
        if zc <= 0.05:
            # Target is behind or at camera optical center
            s_fov = 0.0
        else:
            # Angular offset from optical axis (+Z_c): theta = arctan(sqrt(xc^2 + yc^2) / zc)
            radial_dist = float(np.sqrt(xc ** 2 + yc ** 2))
            theta = float(np.arctan2(radial_dist, zc))
            s_fov = float(np.clip(1.0 - (theta / self.max_fov_angle_rad), 0.0, 1.0))

        # 2. Proximity S_dist in [0, 1] (Inverse of Euclidean distance)
        if distance is None:
            dist = float(np.sqrt(xc ** 2 + yc ** 2 + zc ** 2))
        else:
            dist = float(distance)

        d_clamped = max(self.d_ref_m, dist)
        s_dist = float(np.clip(self.d_ref_m / d_clamped, 0.0, 1.0))

        # 3. Battery Score S_battery in [0, 1]
        s_battery = float(np.clip(battery_pct / 100.0, 0.0, 1.0))

        # 4. Weighted Combination: U = W_fov * S_fov + W_dist * S_dist + W_battery * S_battery
        utility = (
            self.w_fov * s_fov +
            self.w_dist * s_dist +
            self.w_battery * s_battery
        )
        utility_clipped = float(np.clip(utility, 0.0, 1.0))

        return round(utility_clipped, 4), round(s_fov, 4), round(s_dist, 4), round(s_battery, 4)

    def compute_utility(self, peer: PeerTelemetry,
                        tto_safe_s: float,
                        phi_fov: float = 0.85,
                        packet_loss_rate: float = 0.0) -> float:
        """Compute scalar utility score U_j in [0.0, 1.0] for sender candidate ranking."""
        # 1. Hard Gate G(j)
        gate_passed, _ = self.gate.evaluate_hard_gate(peer)
        if not gate_passed:
            return 0.0

        # 2. Feasibility Check
        feas_passed, _ = self.gate.evaluate_feasibility(tto_safe_s)
        if not feas_passed:
            return 0.0

        # 3. Execution Margin Term: Phi_feas = clip((T_safe - t_exec - t_safety) / T_ref, 0, 1)
        required_time_s = self.gate.total_budget_ms / 1000.0
        spare_margin_s = tto_safe_s - required_time_s
        phi_feas = float(np.clip(spare_margin_s / self.t_ref_s, 0.0, 1.0))

        # 4. FOV Overlap Term: Phi_FOV in [0, 1]
        phi_fov_clipped = float(np.clip(phi_fov, 0.0, 1.0))

        # 5. Network Quality Term: Phi_net = (1 - p_loss) * clip(1 - RTT/RTT_max, 0, 1)
        rtt_factor = float(np.clip(1.0 - (peer.rtt_ms / self.rtt_max_ms), 0.0, 1.0))
        phi_net = (1.0 - packet_loss_rate) * rtt_factor

        # 6. Weighted Sum
        utility = self.w1 * phi_feas + self.w2 * phi_fov_clipped + self.w3 * phi_net
        return round(float(utility), 4)

"""Threshold-Gated Multi-Candidate Utility Engine (Math/Protocol Spec v2.3 §2.2).

Formulates candidate rank score:
    U_j = G(j) * Feasible(j) * [w1 * Phi_feas(j) + w2 * Phi_FOV(j) + w3 * Phi_net(j)]
Weights: w = (0.5, 0.3, 0.2)
"""
import numpy as np
from .eligibility_gate import EligibilityGate, PeerTelemetry


class UtilityEngine:
    """Computes ranking utility score among eligible candidate receivers."""

    def __init__(self,
                 w_feasibility: float = 0.50,
                 w_fov: float = 0.30,
                 w_network: float = 0.20,
                 t_ref_s: float = 1.0,
                 rtt_max_ms: float = 100.0,
                 eligibility_gate: EligibilityGate = None):
        self.w1 = w_feasibility
        self.w2 = w_fov
        self.w3 = w_network
        self.t_ref_s = t_ref_s
        self.rtt_max_ms = rtt_max_ms
        self.gate = eligibility_gate if eligibility_gate is not None else EligibilityGate()

    def compute_utility(self, peer: PeerTelemetry,
                        tto_safe_s: float,
                        phi_fov: float = 0.85,
                        packet_loss_rate: float = 0.0) -> float:
        """Compute scalar utility score U_j in [0.0, 1.0]."""
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

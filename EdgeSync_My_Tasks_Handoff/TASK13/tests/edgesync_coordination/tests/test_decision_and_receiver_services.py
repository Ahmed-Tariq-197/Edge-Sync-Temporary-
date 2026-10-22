"""Unit tests for Sprint 4: Decision & Receiver Services (Eligibility & Utility).

Verifies:
1. EligibilityGate Hard Constraints:
   - Rejection on CPU >= 90%.
   - Rejection on Battery <= 15%.
   - Rejection on Nav stack degraded (nav_status != 1).
   - Rejection on stale telemetry (> 2500 ms).
   - Acceptance on nominal telemetry.
2. Analytical Feasibility Timing Check:
   - Strict check: T_TTO_safe MUST be greater than t_txn_ms (886 ms).
   - Rejection when T_TTO_safe <= 886 ms.
   - Acceptance when T_TTO_safe > 886 ms.
3. Utility Function Soft Scoring:
   - Weights: W_fov=0.5, W_dist=0.3, W_battery=0.2.
   - FOV centrality: highest on optical axis (X=0, Y=0), decays with angular offset.
   - Proximity: inverse Euclidean distance.
   - Total utility bounded in [0, 1].
4. CoordinationNode Wiring:
   - Full acceptance pipeline with packed HandoffResponse.msg (ACCEPT, REASON_OK, Utility).
   - Immediate rejection on timing check failure (REJECT, REASON_TIME_INSUFFICIENT).
   - Immediate rejection on high CPU (REJECT, REASON_BUSY_CPU).
   - Immediate rejection on low battery (REJECT, REASON_LOW_BATTERY).
   - Immediate rejection on nav degraded (REJECT, REASON_NAV_UNAVAILABLE).
"""
import math
import numpy as np
import pytest

import rclpy
from edgesync_msgs.msg import (
    HandoffHeader, HandoffRequest, HandoffResponse
)
from edgesync_coordination.decision.eligibility_gate import EligibilityGate, PeerTelemetry
from edgesync_coordination.decision.utility_engine import UtilityEngine
from edgesync_coordination.fsm.ownership_fsm import (
    OwnershipFSM, ReceiverState, OwnershipRole
)
from edgesync_coordination.nodes.coordination_node import CoordinationNode


@pytest.fixture(autouse=True)
def rclpy_session():
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


# ==============================================================================
# 1. EligibilityGate Hard Constraints Tests
# ==============================================================================
class TestEligibilityGateHardConstraints:

    def test_rejection_on_high_cpu(self):
        gate = EligibilityGate(max_cpu_pct=90.0)
        # 90.0% or 95.0% must reject
        peer = PeerTelemetry(cpu_load_pct=95.0, battery_pct=80.0, nav_status=1)
        passed, reason = gate.evaluate_hard_gate(peer)
        assert passed is False
        assert "CPU" in reason

    def test_rejection_on_low_battery(self):
        gate = EligibilityGate(min_battery_pct=15.0)
        # 12.0% must reject
        peer = PeerTelemetry(cpu_load_pct=40.0, battery_pct=12.0, nav_status=1)
        passed, reason = gate.evaluate_hard_gate(peer)
        assert passed is False
        assert "Battery" in reason

    def test_rejection_on_degraded_nav(self):
        gate = EligibilityGate()
        peer = PeerTelemetry(cpu_load_pct=30.0, battery_pct=90.0, nav_status=0)
        passed, reason = gate.evaluate_hard_gate(peer)
        assert passed is False
        assert "Navigation" in reason

    def test_rejection_on_stale_telemetry(self):
        gate = EligibilityGate(max_telemetry_age_ms=2500.0)
        peer = PeerTelemetry(cpu_load_pct=30.0, battery_pct=90.0, nav_status=1, last_heartbeat_age_ms=3000.0)
        passed, reason = gate.evaluate_hard_gate(peer)
        assert passed is False
        assert "stale" in reason

    def test_acceptance_on_nominal_telemetry(self):
        gate = EligibilityGate(max_cpu_pct=90.0, min_battery_pct=15.0)
        peer = PeerTelemetry(cpu_load_pct=35.0, battery_pct=85.0, nav_status=1, last_heartbeat_age_ms=100.0)
        passed, reason = gate.evaluate_hard_gate(peer)
        assert passed is True
        assert "passed" in reason


# ==============================================================================
# 2. Analytical Feasibility Timing Check Tests (t_txn_ms = 886ms)
# ==============================================================================
class TestFeasibilityTimingCheck:

    def test_timing_rejection_below_budget(self):
        """T_TTO_safe (0.700s) <= t_txn_ms (0.886s) -> REJECT."""
        gate = EligibilityGate(t_txn_ms=886.0)
        passed, reason = gate.evaluate_feasibility(tto_safe_s=0.700)
        assert passed is False
        assert "budget" in reason

    def test_timing_rejection_exact_boundary(self):
        """T_TTO_safe (0.886s) MUST be strictly greater than t_txn_ms (0.886s)."""
        gate = EligibilityGate(t_txn_ms=886.0)
        passed, reason = gate.evaluate_feasibility(tto_safe_s=0.886)
        assert passed is False

    def test_timing_acceptance_above_budget(self):
        """T_TTO_safe (1.500s) > t_txn_ms (0.886s) -> CONFIRMED."""
        gate = EligibilityGate(t_txn_ms=886.0)
        passed, reason = gate.evaluate_feasibility(tto_safe_s=1.500)
        assert passed is True
        assert "confirmed" in reason.lower()


# ==============================================================================
# 3. Utility Function Soft Scoring Tests
# ==============================================================================
class TestUtilityFunctionSoftScoring:

    def test_perfect_optical_center_utility(self):
        """Target directly on optical axis at 1m with 100% battery should yield utility = 1.0."""
        engine = UtilityEngine(w_fov=0.50, w_dist=0.30, w_battery=0.20, d_ref_m=1.0)

        # Target: [0, 0, 1] -> dead center, dist=1m
        pt_cam = np.array([0.0, 0.0, 1.0])
        u, s_fov, s_dist, s_batt = engine.compute_receiver_utility(pt_cam, battery_pct=100.0)

        assert pytest.approx(s_fov, abs=1e-3) == 1.0
        assert pytest.approx(s_dist, abs=1e-3) == 1.0
        assert pytest.approx(s_batt, abs=1e-3) == 1.0
        assert pytest.approx(u, abs=1e-3) == 1.0

    def test_distance_decay(self):
        """As distance doubles from 1m to 2m, proximity term halves from 1.0 to 0.5."""
        engine = UtilityEngine(w_fov=0.50, w_dist=0.30, w_battery=0.20, d_ref_m=1.0)

        pt_near = np.array([0.0, 0.0, 1.0])
        _, _, s_dist_near, _ = engine.compute_receiver_utility(pt_near, battery_pct=100.0)

        pt_far = np.array([0.0, 0.0, 2.0])
        _, _, s_dist_far, _ = engine.compute_receiver_utility(pt_far, battery_pct=100.0)

        assert pytest.approx(s_dist_near, abs=1e-3) == 1.0
        assert pytest.approx(s_dist_far, abs=1e-3) == 0.50

    def test_off_axis_fov_decay(self):
        """Target off-axis has strictly lower FOV score than target on-axis."""
        engine = UtilityEngine(w_fov=0.50, w_dist=0.30, w_battery=0.20, max_fov_angle_rad=0.5236)

        pt_center = np.array([0.0, 0.0, 2.0])
        _, s_fov_center, _, _ = engine.compute_receiver_utility(pt_center, battery_pct=80.0)

        pt_off_axis = np.array([0.5, 0.0, 2.0])
        _, s_fov_off, _, _ = engine.compute_receiver_utility(pt_off_axis, battery_pct=80.0)

        assert s_fov_center > s_fov_off
        assert s_fov_off >= 0.0

    def test_utility_bounded_in_unit_interval(self):
        """Utility score is strictly clipped within [0.0, 1.0]."""
        engine = UtilityEngine(w_fov=0.50, w_dist=0.30, w_battery=0.20)

        # Extremely far, off-axis, low battery
        pt_extreme = np.array([10.0, 10.0, 1.0])
        u_low, _, _, _ = engine.compute_receiver_utility(pt_extreme, battery_pct=0.0)
        assert 0.0 <= u_low <= 1.0

        # Close, centered, max battery
        pt_close = np.array([0.0, 0.0, 0.5])
        u_high, _, _, _ = engine.compute_receiver_utility(pt_close, battery_pct=100.0)
        assert 0.0 <= u_high <= 1.0


# ==============================================================================
# 4. CoordinationNode Wiring Tests
# ==============================================================================
class TestCoordinationNodeReceiverWiring:

    def _create_mock_request(self, tto_remaining_ms: int = 1500, tto_sigma_ms: float = 50.0):
        req = HandoffRequest()
        req.header.protocol_version = 0x0203
        req.header.message_type = HandoffHeader.MSG_TYPE_REQUEST
        req.header.sender_id = 'ugv_1'
        req.header.receiver_id = 'ugv_2'
        req.header.transaction_id = 'txn-sprint4-001'
        req.header.ownership_epoch = 1

        req.tto_remaining_ms_at_send = tto_remaining_ms
        req.tto_sigma_ms = tto_sigma_ms
        req.position = [2.0, 0.0, 0.0]  # 2 meters forward in map
        return req

    def test_handoff_acceptance_pipeline(self):
        """Eligible receiver accepts request, packs ACCEPT and utility into HandoffResponse."""
        # Initialize Robot B as Standby Receiver
        fsm_b = OwnershipFSM(robot_id='ugv_2', is_initial_owner=False, initial_epoch=1)
        node = CoordinationNode(fsm=fsm_b, node_name='test_receiver_node')

        # Override parameters for receiver
        node.robot_id = 'ugv_2'
        node.peer_id = 'ugv_1'
        node.negotiation.self_id = 'ugv_2'

        # Set nominal telemetry (CPU=30%, Battery=85%, Nav=1)
        node._my_telemetry.cpu_load_pct = 30.0
        node._my_telemetry.battery_pct = 85.0
        node._my_telemetry.nav_status = 1

        published_responses = []
        node.pub_res.publish = lambda msg: published_responses.append(msg)

        # Incoming request with TTO_safe = 1500 - 2*50 = 1400 ms > 886 ms
        req = self._create_mock_request(tto_remaining_ms=1500, tto_sigma_ms=50.0)
        node._on_handoff_request(req)

        assert len(published_responses) == 1
        res: HandoffResponse = published_responses[0]

        assert res.decision == HandoffResponse.DECISION_ACCEPT
        assert res.reason == HandoffResponse.REASON_OK
        assert res.est_acquisition_ms == 450
        assert node.last_calculated_utility > 0.0
        assert node.fsm.receiver_state == ReceiverState.ACCEPTED
        assert node.fsm.role == OwnershipRole.CANDIDATE

        node.destroy_node()

    def test_timing_rejection_insufficient_tto(self):
        """TTO_safe (700ms) <= 886ms must result in DECISION_REJECT with REASON_TIME_INSUFFICIENT."""
        fsm_b = OwnershipFSM(robot_id='ugv_2', is_initial_owner=False, initial_epoch=1)
        node = CoordinationNode(fsm=fsm_b, node_name='test_receiver_tto_reject')
        node.robot_id = 'ugv_2'
        node.peer_id = 'ugv_1'
        node.negotiation.self_id = 'ugv_2'

        published_responses = []
        node.pub_res.publish = lambda msg: published_responses.append(msg)

        # TTO = 700ms, sigma = 0 -> TTO_safe = 700ms <= 886ms
        req = self._create_mock_request(tto_remaining_ms=700, tto_sigma_ms=0.0)
        node._on_handoff_request(req)

        assert len(published_responses) == 1
        res: HandoffResponse = published_responses[0]

        assert res.decision == HandoffResponse.DECISION_REJECT
        assert res.reason == HandoffResponse.REASON_TIME_INSUFFICIENT
        assert node.fsm.receiver_state == ReceiverState.IDLE
        assert node.fsm.role == OwnershipRole.STANDBY

        node.destroy_node()

    def test_telemetry_rejection_high_cpu(self):
        """Receiver with CPU >= 90% must reject with REASON_BUSY_CPU."""
        fsm_b = OwnershipFSM(robot_id='ugv_2', is_initial_owner=False, initial_epoch=1)
        node = CoordinationNode(fsm=fsm_b, node_name='test_receiver_cpu_reject')
        node.robot_id = 'ugv_2'
        node.peer_id = 'ugv_1'
        node.negotiation.self_id = 'ugv_2'

        node._my_telemetry.cpu_load_pct = 94.0  # Over limit

        published_responses = []
        node.pub_res.publish = lambda msg: published_responses.append(msg)

        req = self._create_mock_request(tto_remaining_ms=1500)
        node._on_handoff_request(req)

        assert len(published_responses) == 1
        res: HandoffResponse = published_responses[0]
        assert res.decision == HandoffResponse.DECISION_REJECT
        assert res.reason == HandoffResponse.REASON_BUSY_CPU
        assert node.fsm.receiver_state == ReceiverState.IDLE

        node.destroy_node()

    def test_telemetry_rejection_low_battery(self):
        """Receiver with Battery <= 15% must reject with REASON_LOW_BATTERY."""
        fsm_b = OwnershipFSM(robot_id='ugv_2', is_initial_owner=False, initial_epoch=1)
        node = CoordinationNode(fsm=fsm_b, node_name='test_receiver_batt_reject')
        node.robot_id = 'ugv_2'
        node.peer_id = 'ugv_1'
        node.negotiation.self_id = 'ugv_2'

        node._my_telemetry.battery_pct = 10.0  # Below minimum

        published_responses = []
        node.pub_res.publish = lambda msg: published_responses.append(msg)

        req = self._create_mock_request(tto_remaining_ms=1500)
        node._on_handoff_request(req)

        assert len(published_responses) == 1
        res: HandoffResponse = published_responses[0]
        assert res.decision == HandoffResponse.DECISION_REJECT
        assert res.reason == HandoffResponse.REASON_LOW_BATTERY

        node.destroy_node()

    def test_telemetry_rejection_degraded_navigation(self):
        """Receiver with nav_status != 1 must reject with REASON_NAV_UNAVAILABLE."""
        fsm_b = OwnershipFSM(robot_id='ugv_2', is_initial_owner=False, initial_epoch=1)
        node = CoordinationNode(fsm=fsm_b, node_name='test_receiver_nav_reject')
        node.robot_id = 'ugv_2'
        node.peer_id = 'ugv_1'
        node.negotiation.self_id = 'ugv_2'

        node._my_telemetry.nav_status = 0  # Degraded

        published_responses = []
        node.pub_res.publish = lambda msg: published_responses.append(msg)

        req = self._create_mock_request(tto_remaining_ms=1500)
        node._on_handoff_request(req)

        assert len(published_responses) == 1
        res: HandoffResponse = published_responses[0]
        assert res.decision == HandoffResponse.DECISION_REJECT
        assert res.reason == HandoffResponse.REASON_NAV_UNAVAILABLE

        node.destroy_node()

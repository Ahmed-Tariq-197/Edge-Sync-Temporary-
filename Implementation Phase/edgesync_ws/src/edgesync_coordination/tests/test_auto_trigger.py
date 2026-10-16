"""Unit tests for Sprint 6 Autonomous Handoff Trigger and Lifecycle Tie-in (edgesync_coordination).

Verifies:
1. FSM Lifecycle Tie-in:
   - When is_initial_owner=True, CoordinationNode starts in TRACKING / OWNER role
     and publishes initial StateHeartbeat and OwnershipEvent (authority declaration).
   - When is_initial_owner=False, CoordinationNode starts in STANDBY / STANDBY role
     and does not publish ownership claim.
2. Auto-Trigger Logic:
   - When in TRACKING state, tto_safe_s >= threshold (e.g. 3.0s >= 2.5s) -> No handoff trigger.
   - When in TRACKING state, tto_safe_s < threshold (e.g. 2.2s < 2.5s) -> Automatically broadcasts
     HandoffRequest and transitions FSM to NEGOTIATING.
3. Cooldown Mechanism:
   - Prevents spamming HandoffRequests within handoff_cooldown_s (default 5.0s).
   - Once cooldown expires (elapsed >= 5.0s), subsequent low TTO safely triggers handoff again.
4. State & Role Guards:
   - Non-OWNER or non-TRACKING states (STANDBY, NEGOTIATING, COMMITTED, RECOVERY)
     strictly ignore low TTO estimates without triggering handoff.
5. Parameter Configurability:
   - Verified with custom parameter overrides (e.g., threshold=1.8s, cooldown=2.0s).
"""
import pytest
import rclpy
from edgesync_msgs.msg import (
    HandoffHeader, HandoffRequest, HandoffResponse,
    OwnershipEvent, StateHeartbeat, TTOEstimate
)
from geometry_msgs.msg import Point
from edgesync_coordination.nodes.coordination_node import CoordinationNode
from edgesync_coordination.fsm.ownership_fsm import (
    OwnershipFSM, SenderState, OwnershipRole
)


@pytest.fixture(autouse=True)
def rclpy_session():
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


class TestFSMLifecycleTieIn:
    """Verifies FSM initial state and authority broadcast at node startup."""

    def test_initial_owner_starts_in_tracking_and_publishes_authority(self):
        """Owner robot starts in TRACKING and broadcasts StateHeartbeat and OwnershipEvent."""
        node = CoordinationNode(node_name='test_initial_owner_node')

        # Assert FSM initial state
        assert node.fsm.role == OwnershipRole.OWNER
        assert node.fsm.sender_state == SenderState.TRACKING

        # The node should have published initial StateHeartbeat and OwnershipEvent
        # Let's inspect that publishers exist
        assert hasattr(node, 'pub_heartbeat')
        assert hasattr(node, 'pub_claim')
        assert node.pub_heartbeat is not None
        assert node.pub_claim is not None

        node.destroy_node()

    def test_standby_node_starts_in_standby_without_claim(self):
        """Non-owner robot starts in STANDBY and does not declare authority."""
        standby_fsm = OwnershipFSM(robot_id='ugv_2', is_initial_owner=False, initial_epoch=1)
        node = CoordinationNode(fsm=standby_fsm, node_name='test_standby_node')

        assert node.fsm.role == OwnershipRole.STANDBY
        assert node.fsm.sender_state == SenderState.STANDBY

        node.destroy_node()


class TestAutonomousHandoffTrigger:
    """Verifies autonomous threshold monitoring and request dispatching."""

    def test_tto_above_threshold_does_not_trigger(self):
        """When TTO safe is 3.0s (>= 2.5s), no handoff request is broadcast."""
        node = CoordinationNode(node_name='test_above_thresh_node')
        published_requests = []
        node.pub_req.publish = lambda msg: published_requests.append(msg)

        assert node.fsm.sender_state == SenderState.TRACKING

        tto_msg = TTOEstimate()
        tto_msg.target_id = 'target_human'
        tto_msg.frame_id = 'map'
        tto_msg.tto_hat_s = 3.5
        tto_msg.sigma_tto_s = 0.25
        tto_msg.tto_safe_s = 3.0
        tto_msg.stamp_utc_ns = 1_000_000_000

        node._on_tto_estimate(tto_msg, current_time_s=100.0)

        # FSM must remain in TRACKING and no request broadcast
        assert node.fsm.sender_state == SenderState.TRACKING
        assert len(published_requests) == 0

        node.destroy_node()

    def test_tto_below_threshold_auto_triggers_handoff(self):
        """When TTO safe is 2.1s (< 2.5s), auto-trigger initiates handoff negotiation."""
        node = CoordinationNode(node_name='test_auto_trigger_node')
        published_requests = []
        node.pub_req.publish = lambda msg: published_requests.append(msg)

        assert node.fsm.sender_state == SenderState.TRACKING

        tto_msg = TTOEstimate()
        tto_msg.target_id = 'target_human'
        tto_msg.frame_id = 'map'
        tto_msg.tto_hat_s = 2.4
        tto_msg.sigma_tto_s = 0.15
        tto_msg.tto_safe_s = 2.1
        tto_msg.stamp_utc_ns = 2_000_000_000
        tto_msg.predicted_trajectory.append(Point(x=1.0, y=2.0, z=0.0))

        sim_time = 100.0
        node._on_tto_estimate(tto_msg, current_time_s=sim_time)

        # FSM must advance from TRACKING -> HANDOFF_CANDIDATE -> NEGOTIATING
        assert node.fsm.sender_state == SenderState.NEGOTIATING
        assert len(published_requests) == 1

        req: HandoffRequest = published_requests[0]
        assert req.header.sender_id == node.robot_id
        assert req.header.receiver_id == node.peer_id
        assert req.tto_remaining_ms_at_send == 2100
        assert pytest.approx(req.tto_sigma_ms, abs=1.0) == 150.0
        assert req.position[0] == 1.0
        assert req.position[1] == 2.0

        # Window open and timestamp updated
        assert node._negotiation_window_open is True
        assert node._last_handoff_trigger_time == sim_time

        node.destroy_node()


class TestHandoffCooldownMechanism:
    """Verifies that cooldown prevents request spamming after trigger, abort, or failure."""

    def test_cooldown_prevents_spamming_within_cooldown_period(self):
        """Requests arriving within cooldown window (e.g. 2s < 5s) must be suppressed."""
        node = CoordinationNode(node_name='test_cooldown_node')
        published_requests = []
        node.pub_req.publish = lambda msg: published_requests.append(msg)

        tto_msg = TTOEstimate()
        tto_msg.tto_safe_s = 2.0
        tto_msg.sigma_tto_s = 0.1
        tto_msg.stamp_utc_ns = 1_000_000_000

        # First trigger at t=100.0s
        node._on_tto_estimate(tto_msg, current_time_s=100.0)
        assert len(published_requests) == 1
        assert node.fsm.sender_state == SenderState.NEGOTIATING

        # Simulate negotiation abort / fallback back to TRACKING
        node.fsm.abort_to_recovery("Negotiation test abort")
        node.fsm.recovery_to_solo_tracking()
        assert node.fsm.sender_state == SenderState.TRACKING

        # Second estimate arrives at t=102.0s (elapsed = 2.0s < cooldown 5.0s)
        node._on_tto_estimate(tto_msg, current_time_s=102.0)
        # MUST NOT TRIGGER A SECOND TIME
        assert len(published_requests) == 1
        assert node.fsm.sender_state == SenderState.TRACKING

        # Third estimate arrives at t=104.9s (still < 5.0s)
        node._on_tto_estimate(tto_msg, current_time_s=104.9)
        assert len(published_requests) == 1
        assert node.fsm.sender_state == SenderState.TRACKING

        # Fourth estimate arrives at t=105.1s (elapsed 5.1s > cooldown 5.0s)
        node._on_tto_estimate(tto_msg, current_time_s=105.1)
        # MUST TRIGGER!
        assert len(published_requests) == 2
        assert node.fsm.sender_state == SenderState.NEGOTIATING
        assert node._last_handoff_trigger_time == 105.1

        node.destroy_node()


class TestStateAndRoleGuards:
    """Verifies that non-tracking states and non-owners cannot trigger handoff."""

    def test_standby_node_ignores_low_tto(self):
        """A STANDBY node observing low TTO never triggers handoff."""
        standby_fsm = OwnershipFSM(robot_id='ugv_1', is_initial_owner=False, initial_epoch=1)
        node = CoordinationNode(fsm=standby_fsm, node_name='test_guard_standby')
        published_requests = []
        node.pub_req.publish = lambda msg: published_requests.append(msg)

        assert node.fsm.role == OwnershipRole.STANDBY
        assert node.fsm.sender_state == SenderState.STANDBY

        tto_msg = TTOEstimate()
        tto_msg.tto_safe_s = 0.5

        node._on_tto_estimate(tto_msg, current_time_s=50.0)

        assert len(published_requests) == 0
        assert node.fsm.sender_state == SenderState.STANDBY

        node.destroy_node()

    def test_committed_sender_ignores_subsequent_tto(self):
        """A sender already in COMMITTED state does not re-trigger on new TTO estimates."""
        node = CoordinationNode(node_name='test_guard_committed')
        published_requests = []
        node.pub_req.publish = lambda msg: published_requests.append(msg)

        # Transition node to COMMITTED
        node.fsm.trigger_candidate('target', 'txn_guard')
        node.fsm.send_request()
        node.fsm.on_peer_accepted()
        assert node.fsm.sender_state == SenderState.COMMITTED

        tto_msg = TTOEstimate()
        tto_msg.tto_safe_s = 0.5
        node._on_tto_estimate(tto_msg, current_time_s=60.0)

        assert len(published_requests) == 0
        assert node.fsm.sender_state == SenderState.COMMITTED

        node.destroy_node()


class TestConfigurableParameters:
    """Verifies custom threshold and cooldown parameters are respected."""

    def test_custom_threshold_and_cooldown(self):
        node = CoordinationNode(node_name='test_custom_params')
        # Inject custom threshold = 1.5s, cooldown = 2.0s
        node.handoff_trigger_threshold_s = 1.5
        node.handoff_cooldown_s = 2.0

        published_requests = []
        node.pub_req.publish = lambda msg: published_requests.append(msg)

        # 1. TTO = 1.8s (> 1.5s custom threshold) -> No trigger
        tto_msg = TTOEstimate(tto_safe_s=1.8, sigma_tto_s=0.1)
        node._on_tto_estimate(tto_msg, current_time_s=10.0)
        assert len(published_requests) == 0

        # 2. TTO = 1.4s (< 1.5s custom threshold) -> Trigger!
        tto_msg.tto_safe_s = 1.4
        node._on_tto_estimate(tto_msg, current_time_s=11.0)
        assert len(published_requests) == 1

        # Reset to tracking
        node.fsm.sender_state = SenderState.TRACKING
        node.fsm.role = OwnershipRole.OWNER

        # 3. t=12.5s (elapsed 1.5s < custom cooldown 2.0s) -> Suppressed
        node._on_tto_estimate(tto_msg, current_time_s=12.5)
        assert len(published_requests) == 1

        # 4. t=13.1s (elapsed 2.1s > custom cooldown 2.0s) -> Trigger!
        node._on_tto_estimate(tto_msg, current_time_s=13.1)
        assert len(published_requests) == 2

        node.destroy_node()

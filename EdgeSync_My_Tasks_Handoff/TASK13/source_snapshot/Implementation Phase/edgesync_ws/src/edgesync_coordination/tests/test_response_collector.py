"""Unit tests for Sprint 5 Response Collector (edgesync_coordination).

Tests:
1. Multi-candidate response collection and selection of highest utility ACCEPT.
2. Filtering out REJECT responses even if raw utility is higher.
3. Timeout with zero responses triggers abort_to_recovery.
4. Timeout with only REJECT responses triggers abort_to_recovery.
5. broadcast_handoff_request arms negotiation window and transitions FSM.
"""
import pytest
import rclpy
from edgesync_msgs.msg import HandoffHeader, HandoffResponse, HandoffRequest
from edgesync_coordination.nodes.coordination_node import CoordinationNode
from edgesync_coordination.fsm.ownership_fsm import OwnershipFSM, SenderState, OwnershipRole


@pytest.fixture(scope="module", autouse=True)
def rclpy_init():
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def create_response(sender_id: str, decision: int, utility: float, txn_id: str = "txn_test", epoch: int = 1) -> HandoffResponse:
    res = HandoffResponse()
    res.header.protocol_version = 0x0203
    res.header.message_type = HandoffHeader.MSG_TYPE_RESPONSE
    res.header.sender_id = sender_id
    res.header.receiver_id = "ugv_1"
    res.header.transaction_id = txn_id
    res.header.ownership_epoch = epoch
    res.decision = decision
    res.reason = HandoffResponse.REASON_OK if decision == HandoffResponse.DECISION_ACCEPT else HandoffResponse.REASON_BUSY_CPU
    res.est_acquisition_ms = 400
    res.cpu_load = 25.0
    res.battery = 80.0
    res.utility = float(utility)
    return res


def test_response_collector_selects_highest_utility():
    fsm = OwnershipFSM(robot_id="ugv_1", is_initial_owner=True, initial_epoch=1)
    node = CoordinationNode(fsm=fsm, node_name="test_collector_highest_util")

    # Transition to NEGOTIATING
    assert fsm.trigger_candidate("target_0", "txn_1") is True
    assert fsm.send_request() is True
    assert fsm.sender_state == SenderState.NEGOTIATING

    # Open negotiation window
    node.start_negotiation_window(timeout_ms=80)
    assert node._negotiation_window_open is True

    # Send 3 responses: ugv_2 (0.65), ugv_3 (0.88), ugv_4 (0.45)
    r2 = create_response("ugv_2", HandoffResponse.DECISION_ACCEPT, 0.65)
    r3 = create_response("ugv_3", HandoffResponse.DECISION_ACCEPT, 0.88)
    r4 = create_response("ugv_4", HandoffResponse.DECISION_ACCEPT, 0.45)

    node._on_handoff_response(r2)
    node._on_handoff_response(r3)
    node._on_handoff_response(r4)

    assert len(node._collected_responses) == 3

    # Close window via timeout
    node._on_negotiation_timeout()

    # Winning candidate MUST be ugv_3 with highest utility
    assert node.selected_receiver_id == "ugv_3"
    assert node.fsm.sender_state == SenderState.COMMITTED
    assert node.watchdog.is_armed is True
    assert node._negotiation_window_open is False

    node.destroy_node()


def test_response_collector_filters_out_rejects():
    fsm = OwnershipFSM(robot_id="ugv_1", is_initial_owner=True, initial_epoch=1)
    node = CoordinationNode(fsm=fsm, node_name="test_collector_filter_reject")

    fsm.trigger_candidate("target_0", "txn_1")
    fsm.send_request()
    node.start_negotiation_window(timeout_ms=80)

    # ugv_2 sends ACCEPT with utility 0.72
    r2 = create_response("ugv_2", HandoffResponse.DECISION_ACCEPT, 0.72)
    # ugv_3 sends REJECT even though it calculated utility 0.95
    r3 = create_response("ugv_3", HandoffResponse.DECISION_REJECT, 0.95)

    node._on_handoff_response(r2)
    node._on_handoff_response(r3)

    node._on_negotiation_timeout()

    # Only ugv_2 was eligible for acceptance
    assert node.selected_receiver_id == "ugv_2"
    assert node.fsm.sender_state == SenderState.COMMITTED

    node.destroy_node()


def test_response_collector_zero_responses_aborts_to_recovery():
    fsm = OwnershipFSM(robot_id="ugv_1", is_initial_owner=True, initial_epoch=1)
    node = CoordinationNode(fsm=fsm, node_name="test_collector_timeout_zero")

    fsm.trigger_candidate("target_0", "txn_1")
    fsm.send_request()
    node.start_negotiation_window(timeout_ms=80)

    # 0 responses arrive during 80ms window
    node._on_negotiation_timeout()

    # Must abort to RECOVERY
    assert node.fsm.sender_state == SenderState.RECOVERY
    assert node.selected_receiver_id is None

    node.destroy_node()


def test_response_collector_all_rejected_aborts_to_recovery():
    fsm = OwnershipFSM(robot_id="ugv_1", is_initial_owner=True, initial_epoch=1)
    node = CoordinationNode(fsm=fsm, node_name="test_collector_all_rejected")

    fsm.trigger_candidate("target_0", "txn_1")
    fsm.send_request()
    node.start_negotiation_window(timeout_ms=80)

    # Two robots both reject
    r2 = create_response("ugv_2", HandoffResponse.DECISION_REJECT, 0.0)
    r3 = create_response("ugv_3", HandoffResponse.DECISION_REJECT, 0.0)
    node._on_handoff_response(r2)
    node._on_handoff_response(r3)

    node._on_negotiation_timeout()

    # Must abort to RECOVERY
    assert node.fsm.sender_state == SenderState.RECOVERY
    assert node.selected_receiver_id is None

    node.destroy_node()


def test_broadcast_handoff_request_starts_negotiation():
    fsm = OwnershipFSM(robot_id="ugv_1", is_initial_owner=True, initial_epoch=1)
    node = CoordinationNode(fsm=fsm, node_name="test_broadcast_handoff_req")

    req = HandoffRequest()
    req.header.protocol_version = 0x0203
    req.header.message_type = HandoffHeader.MSG_TYPE_REQUEST
    req.header.sender_id = "ugv_1"
    req.header.receiver_id = "*"
    req.header.transaction_id = "txn_broadcast_1"
    req.header.ownership_epoch = 1

    assert node.fsm.sender_state == SenderState.TRACKING
    node.broadcast_handoff_request(req)

    assert node.fsm.sender_state == SenderState.NEGOTIATING
    assert node._negotiation_window_open is True
    assert node._neg_timer is not None

    node.destroy_node()

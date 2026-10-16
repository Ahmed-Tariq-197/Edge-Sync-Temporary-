"""Unit tests for SafetyAuthorityNode (C-10 & Quality Gate 7).

Verifies:
1. Ultimate motion override: Critical safe stop engages when peer connection is lost > 2.0s during active handoff.
2. Immediate zero-velocity Twist published upon critical failure.
3. Peer navigation stack crash (NAV_STATUS_FAILED) triggers critical safe stop during handoff.
4. Local navigation stack crash triggers critical safe stop.
5. Watchdog timeout and emergency stop signals engage critical safe stop.
"""
import pytest
import rclpy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

from edgesync_msgs.msg import StateHeartbeat, OwnershipEvent, HandoffHeader
from edgesync_navigation.nodes.safety_authority_node import SafetyAuthorityNode


@pytest.fixture(scope="module", autouse=True)
def rclpy_init():
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def test_safety_authority_init():
    node = SafetyAuthorityNode(node_name="test_safety_init")
    assert node.robot_id == "ugv_1"
    assert node.peer_id == "ugv_2"
    assert node.peer_timeout_s == 2.0
    assert node.is_critical_stopped is False
    assert node.handoff_in_progress is False
    node.destroy_node()


def test_peer_heartbeat_timeout_during_handoff_triggers_critical_safe_stop():
    """If peer connection lost > 2.0s during active handoff, node MUST engage critical safe stop."""
    node = SafetyAuthorityNode(node_name="test_safety_peer_timeout")

    published_twists = []
    node.pub_cmd_safe.publish = lambda msg: published_twists.append(msg)

    # 1. Trigger handoff active via OwnershipEvent ROLE_CANDIDATE
    ev = OwnershipEvent()
    ev.role = OwnershipEvent.ROLE_CANDIDATE
    ev.header.transaction_id = "txn_safety_001"
    node._on_ownership_event(ev)
    assert node.handoff_in_progress is True

    # 2. Simulate peer heartbeat received at t = 100.0s
    node.last_peer_heartbeat_time = 100.0

    # 3. Check at t = 101.5s (< 2.0s timeout) -> Safe stop NOT triggered
    stopped = node.check_safety_conditions(current_time=101.5)
    assert stopped is False
    assert node.is_critical_stopped is False

    # 4. Check at t = 102.5s (> 2.0s timeout) -> CRITICAL SAFE STOP TRIGGERED!
    stopped = node.check_safety_conditions(current_time=102.5)
    assert stopped is True
    assert node.is_critical_stopped is True
    assert "Peer connection completely lost" in node.critical_stop_reason

    # Verify zero-velocity Twist published
    assert len(published_twists) >= 1
    last_cmd: Twist = published_twists[-1]
    assert last_cmd.linear.x == 0.0
    assert last_cmd.linear.y == 0.0
    assert last_cmd.angular.z == 0.0

    node.destroy_node()


def test_peer_heartbeat_maintained_no_safe_stop():
    """If peer heartbeats continuously during handoff, motion authority is preserved."""
    node = SafetyAuthorityNode(node_name="test_safety_peer_healthy")

    node.set_handoff_active(True, "txn_safety_healthy")
    assert node.handoff_in_progress is True

    # Continuous heartbeats
    for t_step in [100.0, 101.0, 102.0, 103.0]:
        hb = StateHeartbeat()
        hb.header.sender_id = "ugv_2"
        hb.nav_status = StateHeartbeat.NAV_STATUS_AVAILABLE
        node._on_peer_heartbeat(hb)
        node.last_peer_heartbeat_time = t_step
        stopped = node.check_safety_conditions(current_time=t_step + 0.8)
        assert stopped is False

    node.destroy_node()


def test_peer_nav_crash_triggers_critical_safe_stop():
    """If peer reports NAV_STATUS_DEGRADED during handoff, engage critical safe stop immediately."""
    node = SafetyAuthorityNode(node_name="test_safety_peer_crash")

    published_twists = []
    node.pub_cmd_safe.publish = lambda msg: published_twists.append(msg)

    node.set_handoff_active(True, "txn_peer_crash")

    hb = StateHeartbeat()
    hb.header.sender_id = "ugv_2"
    hb.nav_status = StateHeartbeat.NAV_STATUS_DEGRADED
    node._on_peer_heartbeat(hb)

    assert node.is_critical_stopped is True
    assert "NAV_STATUS_DEGRADED" in node.critical_stop_reason
    assert len(published_twists) >= 1
    assert published_twists[-1].linear.x == 0.0

    node.destroy_node()


def test_local_nav_crash_triggers_critical_safe_stop():
    """If local robot nav stack reports NAV_STATUS_DEGRADED, engage critical safe stop immediately."""
    node = SafetyAuthorityNode(node_name="test_safety_local_crash")

    published_twists = []
    node.pub_cmd_safe.publish = lambda msg: published_twists.append(msg)

    hb = StateHeartbeat()
    hb.header.sender_id = "ugv_1"
    hb.nav_status = StateHeartbeat.NAV_STATUS_DEGRADED
    node._on_self_heartbeat(hb)

    assert node.is_critical_stopped is True
    assert "Local navigation stack" in node.critical_stop_reason
    assert len(published_twists) >= 1
    assert published_twists[-1].angular.z == 0.0

    node.destroy_node()


def test_watchdog_and_estop_signals_trigger_safe_stop():
    """Verify watchdog timeout signal and emergency stop input engage safe stop."""
    node = SafetyAuthorityNode(node_name="test_safety_wd_estop")

    published_twists = []
    node.pub_cmd_safe.publish = lambda msg: published_twists.append(msg)

    # 1. Watchdog timeout signal
    wd_msg = Bool()
    wd_msg.data = True
    node._on_watchdog_timeout(wd_msg)
    assert node.is_critical_stopped is True
    assert "Watchdog timeout" in node.critical_stop_reason

    # Clear stop
    node.clear_safe_stop()
    assert node.is_critical_stopped is False

    # 2. Emergency stop signal
    estop_msg = Bool()
    estop_msg.data = True
    node._on_emergency_stop(estop_msg)
    assert node.is_critical_stopped is True
    assert "Emergency stop" in node.critical_stop_reason

    node.destroy_node()

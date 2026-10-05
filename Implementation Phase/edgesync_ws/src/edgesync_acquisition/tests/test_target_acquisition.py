"""Unit tests for Sprint 5 Receiver Acquisition Wiring (edgesync_acquisition).

Tests:
1. State machine transitions: IDLE -> SLEWING -> SEARCHING -> VERIFYING.
2. Target centered with verified Re-ID publishes HANDOFF_CONFIRM.
3. Target centered with Re-ID mismatch publishes HANDOFF_CANCEL (REASON_REID_MISMATCH).
4. Acquisition timeout (700ms cutoff) publishes HANDOFF_CANCEL (REASON_TRACK_LOST).
"""
import pytest
import numpy as np
import rclpy
from edgesync_msgs.msg import (
    HandoffRequest, HandoffConfirm, HandoffCancel, AcqStatus
)
from edgesync_msgs.srv import IdentityVerify
from edgesync_acquisition.nodes.target_acquisition_node import (
    TargetAcquisitionNode, AcquisitionState
)


@pytest.fixture(scope="module", autouse=True)
def rclpy_init():
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def test_acquisition_state_transitions_to_verifying():
    node = TargetAcquisitionNode(node_name="test_acq_transitions")
    assert node.state == AcquisitionState.IDLE

    # 1. Start acquisition
    node.start_acquisition(
        prior_pos_map=(2.5, 0.5, 0.0),
        txn_id="txn_sprint5_test",
        epoch=2,
        reference_vector=np.ones(512, dtype=np.float32),
        sender_id="ugv_1"
    )
    assert node.state == AcquisitionState.SLEWING

    # 2. Target enters FOV
    node.on_target_detected()
    assert node.state == AcquisitionState.SEARCHING

    # 3. Target centered -> transitions to VERIFYING
    node.center_and_verify()
    assert node.state == AcquisitionState.VERIFYING

    node.destroy_node()


def test_reid_match_publishes_confirm():
    node = TargetAcquisitionNode(node_name="test_acq_match_confirm")

    # Capture published confirm messages
    published_confirms = []
    node.pub_confirm.publish = lambda msg: published_confirms.append(msg)

    node.start_acquisition(
        prior_pos_map=(3.0, 1.0, 0.0),
        txn_id="txn_confirm_test",
        epoch=3,
        reference_vector=np.ones(512, dtype=np.float32),
        sender_id="ugv_1"
    )
    node.on_target_detected()
    node.center_and_verify()

    # Simulate /identity/verify service returning match_confirmed = True
    node.handle_verify_result(match_confirmed=True, similarity=0.88, message="Match confirmed")

    assert node.state == AcquisitionState.CONFIRMED
    assert len(published_confirms) == 1

    confirm: HandoffConfirm = published_confirms[0]
    assert confirm.header.transaction_id == "txn_confirm_test"
    assert confirm.header.ownership_epoch == 3
    assert confirm.header.receiver_id == "ugv_1"
    assert pytest.approx(confirm.reid_similarity, abs=1e-4) == 0.88

    node.destroy_node()


def test_reid_mismatch_publishes_cancel_with_reason_reid_mismatch():
    node = TargetAcquisitionNode(node_name="test_acq_mismatch_cancel")

    published_cancels = []
    node.pub_cancel.publish = lambda msg: published_cancels.append(msg)

    node.start_acquisition(
        prior_pos_map=(3.0, 1.0, 0.0),
        txn_id="txn_mismatch_test",
        epoch=4,
        reference_vector=np.ones(512, dtype=np.float32),
        sender_id="ugv_1"
    )
    node.on_target_detected()
    node.center_and_verify()

    # Simulate /identity/verify service returning match_confirmed = False
    node.handle_verify_result(match_confirmed=False, similarity=0.42, message="Mismatch")

    assert node.state == AcquisitionState.CANCELLED
    assert len(published_cancels) == 1

    cancel: HandoffCancel = published_cancels[0]
    assert cancel.header.transaction_id == "txn_mismatch_test"
    assert cancel.header.ownership_epoch == 4
    assert cancel.header.receiver_id == "ugv_1"
    assert cancel.reason == HandoffCancel.REASON_REID_MISMATCH

    node.destroy_node()


def test_acquisition_timeout_publishes_cancel_track_lost():
    node = TargetAcquisitionNode(node_name="test_acq_timeout_cancel")

    published_cancels = []
    node.pub_cancel.publish = lambda msg: published_cancels.append(msg)

    node.start_acquisition(
        prior_pos_map=(3.0, 1.0, 0.0),
        txn_id="txn_timeout_test",
        epoch=5,
        reference_vector=np.ones(512, dtype=np.float32),
        sender_id="ugv_1"
    )

    # Fire acquisition timeout (700 ms exceeded without lock)
    node._on_acquisition_timeout()

    assert node.state == AcquisitionState.CANCELLED
    assert len(published_cancels) == 1

    cancel: HandoffCancel = published_cancels[0]
    assert cancel.header.transaction_id == "txn_timeout_test"
    assert cancel.reason == HandoffCancel.REASON_TRACK_LOST

    node.destroy_node()


def test_p_controller_proportional_output():
    """Verify Proportional controller calculates angular velocity and handles deadband and saturation."""
    node = TargetAcquisitionNode(node_name="test_acq_p_controller")

    # Optical center is 320.0, tolerance is 15.0 px, kp = 0.005, max_w = 1.5 rad/s
    # 1. Target to right of center (error = +100 px) -> negative angular velocity (rotate right)
    w_z = node.compute_p_control(centroid_x=420.0, optical_center_x=320.0, is_pixel=True)
    assert pytest.approx(w_z, abs=1e-3) == -0.50

    # 2. Target to left of center (error = -100 px) -> positive angular velocity (rotate left)
    w_z = node.compute_p_control(centroid_x=220.0, optical_center_x=320.0, is_pixel=True)
    assert pytest.approx(w_z, abs=1e-3) == 0.50

    # 3. Target inside deadband (|error| <= 15 px) -> exactly 0.0
    w_z = node.compute_p_control(centroid_x=325.0, optical_center_x=320.0, is_pixel=True)
    assert w_z == 0.0

    # 4. Extreme error saturates at max_angular_vel (1.5 rad/s)
    w_z = node.compute_p_control(centroid_x=1000.0, optical_center_x=320.0, is_pixel=True)
    assert pytest.approx(w_z, abs=1e-3) == -1.50

    node.destroy_node()


def test_p_controller_tracking_and_auto_centering():
    """Verify apply_tracking_control rotates during SEARCHING and auto-transitions to VERIFYING when centered."""
    node = TargetAcquisitionNode(node_name="test_acq_tracking_flow")

    published_twists = []
    node.pub_cmd_vel.publish = lambda msg: published_twists.append(msg)

    # Start acquisition in SLEWING
    node.start_acquisition(
        prior_pos_map=(2.0, 0.0, 0.0),
        txn_id="txn_p_ctrl",
        epoch=1,
        sender_id="ugv_1"
    )
    assert node.state == AcquisitionState.SLEWING

    # 1. Tracking control with target offset: transitions SLEWING -> SEARCHING and publishes Twist
    cmd = node.apply_tracking_control(centroid_x=400.0, is_pixel=True)
    assert node.state == AcquisitionState.SEARCHING
    assert cmd.angular.z < 0.0  # Rotating right toward target
    assert len(published_twists) > 0

    # 2. Centroid centered within tolerance: triggers center_and_verify -> VERIFYING and stops motion
    published_twists.clear()
    cmd_centered = node.apply_tracking_control(centroid_x=322.0, is_pixel=True)
    assert node.state == AcquisitionState.VERIFYING
    assert len(published_twists) >= 1
    last_twist = published_twists[-1]
    assert last_twist.angular.z == 0.0
    assert last_twist.linear.x == 0.0

    node.destroy_node()


def test_zero_velocity_published_on_cancelled():
    """Verify zero-velocity Twist is published on CANCELLED (via timeout and mismatch)."""
    node = TargetAcquisitionNode(node_name="test_acq_zero_twist_cancel")

    published_twists = []
    node.pub_cmd_vel.publish = lambda msg: published_twists.append(msg)

    node.start_acquisition(
        prior_pos_map=(2.0, 0.0, 0.0),
        txn_id="txn_cancel_twist",
        epoch=1,
        sender_id="ugv_1"
    )

    published_twists.clear()
    # Trigger timeout -> transitions to CANCELLED
    node._on_acquisition_timeout()
    assert node.state == AcquisitionState.CANCELLED
    assert len(published_twists) >= 1
    assert published_twists[-1].angular.z == 0.0
    assert published_twists[-1].linear.x == 0.0

    node.destroy_node()


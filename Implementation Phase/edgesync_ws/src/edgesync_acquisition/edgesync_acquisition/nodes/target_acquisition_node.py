"""Target Acquisition Node (C-08): Frustum Lock & Search Controller.

SAD v2.3 §3 & Math/Protocol Spec v2.3 §3.2:
- Bounds acquisition window to <= 700 ms (hard cutoff).
- Receives target spatial prior and initiates rotation slew.
- State machine: IDLE -> SLEWING -> SEARCHING -> VERIFYING -> CONFIRMED / CANCELLED.
- Interrogates /identity/verify service once target is centered in FOV.
- Publishes HANDOFF_CONFIRM on confirmed match; publishes HANDOFF_CANCEL (REASON_REID_MISMATCH) on failure.
"""
from enum import Enum, auto
from typing import Optional, List, Tuple
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, PointStamped
from sensor_msgs.msg import Image

from edgesync_msgs.msg import (
    HandoffHeader, HandoffRequest, HandoffConfirm, HandoffCancel, AcqStatus
)
from edgesync_msgs.srv import IdentityVerify


class AcquisitionState(Enum):
    IDLE = auto()
    SLEWING = auto()
    SEARCHING = auto()
    VERIFYING = auto()
    CONFIRMED = auto()
    CANCELLED = auto()


class TargetAcquisitionNode(Node):
    """ROS 2 Node controlling chassis interception slew and candidate acquisition."""

    def __init__(self, node_name: str = 'target_acquisition_node'):
        super().__init__(node_name)

        # 1. Parameter Declarations (Zero Hardcoding)
        self.declare_parameter('robot_id', 'ugv_2')
        self.declare_parameter('peer_id', 'ugv_1')
        self.declare_parameter('protocol_version', 0x0203)
        self.declare_parameter('acquisition_timeout_ms', 700)
        self.declare_parameter('slew_angular_speed_rad_s', 1.2)
        self.declare_parameter('similarity_threshold', 0.70)
        self.declare_parameter('service_name', '/identity/verify')

        # Kinematics & P-Controller Parameters
        self.robot_id = self.get_parameter('robot_id').get_parameter_value().string_value
        default_cmd_vel = f'/{self.robot_id}/edgesync/cmd_vel'
        default_centroid = f'/{self.robot_id}/edgesync/perception/target_centroid_camera'

        self.declare_parameter('cmd_vel_topic', default_cmd_vel)
        self.declare_parameter('centroid_topic', default_centroid)
        self.declare_parameter('kp_angular', 0.005)
        self.declare_parameter('optical_center_x', 320.0)
        self.declare_parameter('max_angular_vel_rad_s', 1.5)
        self.declare_parameter('centering_tolerance_px', 15.0)

        # 2. Extract Parameters
        self.peer_id = self.get_parameter('peer_id').get_parameter_value().string_value
        self.proto_ver = self.get_parameter('protocol_version').get_parameter_value().integer_value
        self.timeout_ms = self.get_parameter('acquisition_timeout_ms').get_parameter_value().integer_value
        self.slew_speed = self.get_parameter('slew_angular_speed_rad_s').get_parameter_value().double_value
        self.sim_thresh = self.get_parameter('similarity_threshold').get_parameter_value().double_value
        self.service_name = self.get_parameter('service_name').get_parameter_value().string_value

        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        self.centroid_topic = self.get_parameter('centroid_topic').get_parameter_value().string_value
        self.kp_angular = self.get_parameter('kp_angular').get_parameter_value().double_value
        self.optical_center_x = self.get_parameter('optical_center_x').get_parameter_value().double_value
        self.max_angular_vel = self.get_parameter('max_angular_vel_rad_s').get_parameter_value().double_value
        self.centering_tolerance = self.get_parameter('centering_tolerance_px').get_parameter_value().double_value

        # 3. State Machine & Active Handoff Data
        self.state = AcquisitionState.IDLE
        self.active_transaction_id: str = ""
        self.active_epoch: int = 1
        self.active_reference_vector: Optional[np.ndarray] = None
        self._target_prior: Optional[Tuple[float, float, float]] = None
        self.last_similarity: float = 0.0
        self._timeout_timer = None

        # 4. QoS Profiles
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 5. Service Client for I-10 Identity Verification
        self.cli_verify = self.create_client(IdentityVerify, self.service_name)

        # 6. Publishers
        self.pub_cmd_vel = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.pub_confirm = self.create_publisher(
            HandoffConfirm, f'/{self.peer_id}/edgesync/handoff_confirm', reliable_qos
        )
        self.pub_cancel = self.create_publisher(
            HandoffCancel, f'/{self.peer_id}/edgesync/handoff_cancel', reliable_qos
        )
        self.pub_acq_status = self.create_publisher(
            AcqStatus, f'/{self.peer_id}/edgesync/acq_status', best_effort_qos
        )

        # 7. Subscribers
        self.sub_request = self.create_subscription(
            HandoffRequest, f'/{self.robot_id}/edgesync/handoff_request', self._on_handoff_request, reliable_qos
        )
        self.sub_centroid = self.create_subscription(
            PointStamped, self.centroid_topic, self._on_target_centroid, best_effort_qos
        )

        self.get_logger().info(
            f"Target Acquisition Node initialized (robot_id={self.robot_id}, cmd_vel={self.cmd_vel_topic}, timeout={self.timeout_ms}ms)."
        )

    def stop_motion(self) -> Twist:
        """Immediately publish zero-velocity Twist to halt chassis rotation."""
        stop_twist = Twist()
        stop_twist.linear.x = 0.0
        stop_twist.linear.y = 0.0
        stop_twist.linear.z = 0.0
        stop_twist.angular.x = 0.0
        stop_twist.angular.y = 0.0
        stop_twist.angular.z = 0.0
        self.pub_cmd_vel.publish(stop_twist)
        return stop_twist

    def compute_p_control(self, centroid_x: float, optical_center_x: Optional[float] = None, is_pixel: bool = True) -> float:
        """Calculate angular z-velocity via Proportional control.

        Args:
            centroid_x: Horizontal pixel or coordinate position.
            optical_center_x: Reference center (default self.optical_center_x for pixel, 0.0 for metric).
            is_pixel: True if in pixels, False if in metric camera frame coordinates.

        Returns:
            Angular z-velocity in rad/s (clamped to [-max_angular_vel, max_angular_vel]).
        """
        if is_pixel:
            center = optical_center_x if optical_center_x is not None else self.optical_center_x
            tol = self.centering_tolerance
            kp = self.kp_angular
        else:
            center = optical_center_x if optical_center_x is not None else 0.0
            tol = 0.05
            kp = self.kp_angular * 300.0 if self.kp_angular < 0.1 else self.kp_angular

        error_x = float(centroid_x - center)
        if abs(error_x) <= tol:
            return 0.0

        # Rotate right (clockwise, negative w_z) if target is right of center (error_x > 0)
        w_z = -float(kp * error_x)
        w_z = float(np.clip(w_z, -self.max_angular_vel, self.max_angular_vel))
        return w_z

    def apply_tracking_control(self, centroid_x: float, is_pixel: bool = True, optical_center_x: Optional[float] = None) -> Twist:
        """Apply P-Controller during SLEWING and SEARCHING states.

        If error is within tolerance, transitions to VERIFYING and stops chassis.
        """
        if self.state == AcquisitionState.SLEWING:
            self.state = AcquisitionState.SEARCHING
            self.get_logger().info("Target detected during SLEWING. Transitioning to SEARCHING.")
            self._publish_acq_status(AcqStatus.STATUS_SEARCHING, confidence=0.60)

        if self.state != AcquisitionState.SEARCHING:
            return self.stop_motion()

        w_z = self.compute_p_control(centroid_x, optical_center_x=optical_center_x, is_pixel=is_pixel)

        if abs(w_z) < 1e-4:
            self.get_logger().info("Target centered within tolerance. Transitioning to VERIFYING.")
            self.center_and_verify()
            return Twist()

        cmd = Twist()
        cmd.angular.z = w_z
        self.pub_cmd_vel.publish(cmd)
        return cmd

    def _on_target_centroid(self, msg: PointStamped):
        """Callback for PointStamped target centroid from perception."""
        if self.state in (AcquisitionState.SLEWING, AcquisitionState.SEARCHING):
            self.apply_tracking_control(centroid_x=msg.point.x, is_pixel=False)

    def _on_handoff_request(self, msg: HandoffRequest):
        """Receive HandoffRequest with target prior and reference Re-ID vector."""
        prior = (float(msg.position[0]), float(msg.position[1]), float(msg.position[2]))
        ref_vec = np.asarray(msg.reid_feature_vector, dtype=np.float32)
        self.start_acquisition(
            prior_pos_map=prior,
            txn_id=msg.header.transaction_id,
            epoch=msg.header.ownership_epoch,
            reference_vector=ref_vec,
            sender_id=msg.header.sender_id
        )

    def start_acquisition(self, prior_pos_map: Tuple[float, float, float],
                          txn_id: str = "txn_0",
                          epoch: int = 1,
                          reference_vector: Optional[np.ndarray] = None,
                          sender_id: Optional[str] = None):
        """Begin intercept acquisition toward target prior."""
        self._target_prior = prior_pos_map
        self.active_transaction_id = txn_id
        self.active_epoch = epoch
        if reference_vector is not None:
            self.active_reference_vector = np.asarray(reference_vector, dtype=np.float32).reshape(-1)
        if sender_id:
            self.peer_id = sender_id

        self.state = AcquisitionState.SLEWING
        self.get_logger().info(f"Acquisition initiated toward prior {prior_pos_map} for txn {txn_id}. State: SLEWING.")

        # Arm acquisition hard cutoff timer (700 ms)
        self._arm_timeout_timer()

        # Command initial chassis rotation towards target
        twist = Twist()
        twist.angular.z = float(self.slew_speed)
        self.pub_cmd_vel.publish(twist)

        # Publish ACQ_STATUS SLEWING
        self._publish_acq_status(AcqStatus.STATUS_SLEWING, confidence=0.30)

    def on_target_detected(self, centroid_x: Optional[float] = None):
        """Target enters receiver FOV; transition from SLEWING to SEARCHING."""
        if self.state == AcquisitionState.SLEWING:
            self.state = AcquisitionState.SEARCHING
            self.get_logger().info("Target entered FOV. State: SEARCHING.")
            self._publish_acq_status(AcqStatus.STATUS_SEARCHING, confidence=0.60)

        if centroid_x is not None:
            self.apply_tracking_control(centroid_x)

    def center_and_verify(self, candidate_crop: Optional[Image] = None,
                          candidate_vector: Optional[np.ndarray] = None):
        """Receiver centers target in FOV (SEARCHING -> VERIFYING) and calls /identity/verify."""
        if self.state in (AcquisitionState.SLEWING, AcquisitionState.SEARCHING):
            self.state = AcquisitionState.VERIFYING
            self.get_logger().info("Target centered in FOV. State: VERIFYING. Halting chassis and invoking Re-ID.")

            # Stop rotation immediately on entering VERIFYING
            self.stop_motion()

            # Publish ACQ_STATUS ACQUIRED
            self._publish_acq_status(AcqStatus.STATUS_ACQUIRED, confidence=0.85)

            # Make service call to /identity/verify
            self._call_identity_verify(candidate_crop=candidate_crop, candidate_vector=candidate_vector)

    def _call_identity_verify(self, candidate_crop: Optional[Image] = None,
                              candidate_vector: Optional[np.ndarray] = None):
        """Build and send IdentityVerify service request."""
        req = IdentityVerify.Request()
        req.transaction_id = self.active_transaction_id

        if self.active_reference_vector is not None:
            req.reference_feature_vector = [float(x) for x in self.active_reference_vector]

        if candidate_vector is not None:
            c_arr = np.asarray(candidate_vector, dtype=np.float32).reshape(-1)
            if hasattr(req, 'candidate_feature_vector'):
                req.candidate_feature_vector = [float(x) for x in c_arr]
        elif candidate_crop is not None and hasattr(req, 'candidate_crop'):
            req.candidate_crop = candidate_crop

        future = self.cli_verify.call_async(req)
        future.add_done_callback(self._on_verify_completed)

    def _on_verify_completed(self, future):
        """Callback when /identity/verify returns response."""
        try:
            res = future.result()
            is_match = bool(getattr(res, 'match_confirmed', getattr(res, 'match', False)))
            sim_score = float(getattr(res, 'similarity_score', getattr(res, 'similarity', 0.0)))
            msg = getattr(res, 'message', '')
            self.handle_verify_result(is_match, sim_score, msg)
        except Exception as e:
            self.get_logger().error(f"Identity verify service call failed: {e}")
            self.handle_verify_result(False, 0.0, str(e))

    def handle_verify_result(self, match_confirmed: bool, similarity: float, message: str = ""):
        """Evaluate Re-ID result: publish HANDOFF_CONFIRM or HANDOFF_CANCEL."""
        self.last_similarity = float(similarity)
        self._disarm_timeout_timer()

        # Halt chassis rotation in all termination paths
        self.stop_motion()

        if match_confirmed:
            self.state = AcquisitionState.CONFIRMED
            self.get_logger().info(
                f"Re-ID MATCH CONFIRMED (sim={similarity:.4f} >= {self.sim_thresh:.2f}). Publishing HANDOFF_CONFIRM."
            )
            confirm_msg = HandoffConfirm()
            confirm_msg.header.protocol_version = self.proto_ver
            confirm_msg.header.message_type = HandoffHeader.MSG_TYPE_CONFIRM
            confirm_msg.header.sender_id = self.robot_id
            confirm_msg.header.receiver_id = self.peer_id
            confirm_msg.header.transaction_id = self.active_transaction_id
            confirm_msg.header.ownership_epoch = self.active_epoch
            confirm_msg.header.stamp_utc_ns = self.get_clock().now().nanoseconds
            confirm_msg.reid_similarity = float(similarity)
            confirm_msg.frame_id = "map"
            if self._target_prior is not None:
                confirm_msg.position = [
                    float(self._target_prior[0]), float(self._target_prior[1]), float(self._target_prior[2])
                ]
            self.pub_confirm.publish(confirm_msg)
        else:
            self.state = AcquisitionState.CANCELLED
            self.get_logger().warn(
                f"Re-ID MISMATCH (sim={similarity:.4f} < {self.sim_thresh:.2f}). Publishing HANDOFF_CANCEL (REASON_REID_MISMATCH)."
            )
            cancel_msg = HandoffCancel()
            cancel_msg.header.protocol_version = self.proto_ver
            cancel_msg.header.message_type = HandoffHeader.MSG_TYPE_CANCEL
            cancel_msg.header.sender_id = self.robot_id
            cancel_msg.header.receiver_id = self.peer_id
            cancel_msg.header.transaction_id = self.active_transaction_id
            cancel_msg.header.ownership_epoch = self.active_epoch
            cancel_msg.header.stamp_utc_ns = self.get_clock().now().nanoseconds
            cancel_msg.reason = HandoffCancel.REASON_REID_MISMATCH
            self.pub_cancel.publish(cancel_msg)

    def _arm_timeout_timer(self):
        """Arm hard acquisition timeout timer (700 ms)."""
        self._disarm_timeout_timer()
        period_s = max(0.01, float(self.timeout_ms) / 1000.0)
        self._timeout_timer = self.create_timer(period_s, self._on_acquisition_timeout)

    def _disarm_timeout_timer(self):
        """Disarm acquisition timeout timer."""
        if self._timeout_timer is not None:
            self._timeout_timer.cancel()
            self.destroy_timer(self._timeout_timer)
            self._timeout_timer = None

    def _on_acquisition_timeout(self):
        """Acquisition deadline exceeded (700ms cutoff) -> Abort with HANDOFF_CANCEL."""
        self._disarm_timeout_timer()
        if self.state not in (AcquisitionState.CONFIRMED, AcquisitionState.CANCELLED, AcquisitionState.IDLE):
            self.state = AcquisitionState.CANCELLED
            self.stop_motion()
            self.get_logger().error(f"Acquisition HARD TIMEOUT ({self.timeout_ms}ms) exceeded. Aborting with HANDOFF_CANCEL.")
            cancel_msg = HandoffCancel()
            cancel_msg.header.protocol_version = self.proto_ver
            cancel_msg.header.message_type = HandoffHeader.MSG_TYPE_CANCEL
            cancel_msg.header.sender_id = self.robot_id
            cancel_msg.header.receiver_id = self.peer_id
            cancel_msg.header.transaction_id = self.active_transaction_id
            cancel_msg.header.ownership_epoch = self.active_epoch
            cancel_msg.header.stamp_utc_ns = self.get_clock().now().nanoseconds
            cancel_msg.reason = HandoffCancel.REASON_TRACK_LOST
            self.pub_cancel.publish(cancel_msg)

    def _publish_acq_status(self, status: int, confidence: float):
        """Publish AcqStatus heartbeat."""
        msg = AcqStatus()
        msg.header.protocol_version = self.proto_ver
        msg.header.message_type = HandoffHeader.MSG_TYPE_ACQ_STATUS
        msg.header.sender_id = self.robot_id
        msg.header.receiver_id = self.peer_id
        msg.header.transaction_id = self.active_transaction_id
        msg.header.ownership_epoch = self.active_epoch
        msg.header.stamp_utc_ns = self.get_clock().now().nanoseconds
        msg.status = status
        msg.confidence = float(confidence)
        self.pub_acq_status.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TargetAcquisitionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

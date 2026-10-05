#!/usr/bin/env python3
"""Comprehensive end-to-end diagnostic for EdgeSync handoff.

This script:
1. Subscribes to ALL handoff topics on /edgesync_swarm/... to watch for negotiation traffic.
2. Subscribes to /robot_a/edgesync/tto/estimate to echo what coordination_node actually receives.
3. Publishes a properly-formed TTO message with guaranteed delivery (timer-based, multiple attempts).
4. Reports every message seen on every topic.
"""
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from edgesync_msgs.msg import (
    HandoffRequest, HandoffResponse, HandoffConfirm,
    HandoffConfirmAck, HandoffCancel, OwnershipEvent,
    StateHeartbeat, TTOEstimate
)
from geometry_msgs.msg import Point


class HandoffDiagnostic(Node):
    def __init__(self):
        super().__init__('handoff_diagnostic')

        best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=10
        )
        reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=10
        )

        # === WATCHERS: Subscribe to ALL swarm topics ===
        self.create_subscription(
            HandoffRequest, '/edgesync_swarm/handoff_request',
            lambda m: self.get_logger().info(
                f"[SWARM] HandoffRequest from={m.header.sender_id} to={m.header.receiver_id} txn={m.header.transaction_id} tto_ms={m.tto_remaining_ms_at_send}"
            ), reliable)

        self.create_subscription(
            HandoffResponse, '/edgesync_swarm/handoff_response',
            lambda m: self.get_logger().info(
                f"[SWARM] HandoffResponse from={m.header.sender_id} to={m.header.receiver_id} decision={m.decision} utility={m.utility:.4f}"
            ), reliable)

        self.create_subscription(
            HandoffConfirm, '/edgesync_swarm/handoff_confirm',
            lambda m: self.get_logger().info(
                f"[SWARM] HandoffConfirm from={m.header.sender_id} to={m.header.receiver_id}"
            ), reliable)

        self.create_subscription(
            HandoffConfirmAck, '/edgesync_swarm/handoff_confirm_ack',
            lambda m: self.get_logger().info(
                f"[SWARM] HandoffConfirmAck from={m.header.sender_id} to={m.header.receiver_id}"
            ), reliable)

        self.create_subscription(
            HandoffCancel, '/edgesync_swarm/handoff_cancel',
            lambda m: self.get_logger().info(
                f"[SWARM] HandoffCancel from={m.header.sender_id}"
            ), reliable)

        self.create_subscription(
            OwnershipEvent, '/edgesync_swarm/ownership_event',
            lambda m: self.get_logger().info(
                f"[SWARM] OwnershipEvent from={m.header.sender_id} role={m.role} epoch={m.header.ownership_epoch}"
            ), reliable)

        self.create_subscription(
            StateHeartbeat, '/edgesync_swarm/state_heartbeat',
            lambda m: self.get_logger().info(
                f"[SWARM] StateHeartbeat from={m.robot_id} cpu={m.cpu_load:.0f}% bat={m.battery_pct:.0f}%"
            ), best_effort)

        # === Echo TTO arriving at robot_a ===
        self.create_subscription(
            TTOEstimate, '/robot_a/edgesync/tto/estimate',
            lambda m: self.get_logger().info(
                f"[ECHO] TTO arrived on robot_a topic: tto_safe={m.tto_safe_s:.2f}s tto_hat={m.tto_hat_s:.2f}s"
            ), best_effort)

        # === Publisher for TTO injection ===
        self.tto_pub = self.create_publisher(
            TTOEstimate, '/robot_a/edgesync/tto/estimate', best_effort
        )

        self._injection_count = 0
        self._max_injections = 10

        # Timer: inject TTO every 1 second until we see results or hit max
        self.inject_timer = self.create_timer(1.0, self._inject_tto)

        self.get_logger().info("="*60)
        self.get_logger().info("EdgeSync Handoff Diagnostic Node started")
        self.get_logger().info(f"Waiting for subscribers on /robot_a/edgesync/tto/estimate...")
        self.get_logger().info(f"Subscription count: {self.tto_pub.get_subscription_count()}")
        self.get_logger().info("Listening on ALL /edgesync_swarm/* topics")
        self.get_logger().info("="*60)

    def _inject_tto(self):
        sub_count = self.tto_pub.get_subscription_count()

        if sub_count == 0:
            self.get_logger().warn(f"Injection #{self._injection_count}: No subscribers yet on TTO topic. Waiting...")
            return

        msg = TTOEstimate()
        msg.target_id = 'target_1'
        msg.frame_id = 'map'
        msg.tto_safe_s = 1.0        # well below 2.5s threshold
        msg.tto_hat_s = 1.2
        msg.sigma_tto_s = 0.1
        msg.exit_boundary = 'horizontal_right'
        msg.stamp_utc_ns = self.get_clock().now().nanoseconds

        target_pt = Point()
        target_pt.x = 1.0
        target_pt.y = 2.0
        target_pt.z = 0.0
        msg.predicted_trajectory.append(target_pt)

        self.tto_pub.publish(msg)
        self._injection_count += 1
        self.get_logger().info(
            f"TTO Injection #{self._injection_count} sent (tto_safe=1.0s, subs={sub_count})"
        )

        if self._injection_count >= self._max_injections:
            self.get_logger().warn("Max injections reached. Stopping TTO injection.")
            self.inject_timer.cancel()


def main():
    rclpy.init()
    node = HandoffDiagnostic()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Diagnostic shutting down.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

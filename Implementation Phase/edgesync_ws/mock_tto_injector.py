#!/usr/bin/env python3
"""Standalone script to inject low TTO to Robot A and trigger autonomous handoff."""
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from edgesync_msgs.msg import TTOEstimate
from geometry_msgs.msg import Point


def main():
    rclpy.init()
    node = Node('mock_tto_injector')

    qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=10
    )

    topic_name = '/robot_a/edgesync/tto/estimate'
    publisher = node.create_publisher(TTOEstimate, topic_name, qos)

    node.get_logger().info(f"Mock TTO Injector started. Target topic: {topic_name}")

    # Wait until Robot A's coordination_node has connected
    while publisher.get_subscription_count() == 0:
        node.get_logger().info("Waiting for Robot A to connect...")
        rclpy.spin_once(node, timeout_sec=0.1)
        time.sleep(0.5)

    node.get_logger().info(f"Robot A connected! (Subscriptions: {publisher.get_subscription_count()})")

    # Build TTOEstimate triggering autonomous handoff (tto_safe_s < 2.5s)
    msg = TTOEstimate()
    msg.target_id = 'target_1'
    msg.frame_id = 'map'
    msg.tto_safe_s = 1.0
    msg.tto_hat_s = 1.2
    msg.sigma_tto_s = 0.1
    msg.exit_boundary = 'horizontal_right'
    msg.stamp_utc_ns = node.get_clock().now().nanoseconds

    target_pt = Point()
    target_pt.x = 1.0
    target_pt.y = 2.0
    target_pt.z = 0.0
    msg.predicted_trajectory.append(target_pt)

    # Publish message
    publisher.publish(msg)
    node.get_logger().info("Injection successful!")

    # Allow time for transmission before teardown
    time.sleep(1.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

"""Navigation Bridge Node (C-10): Nav2 Integration & Intercept Pre-positioning.

Wraps Nav2 Python interface and translates advisory intercept poses into navigation goals.
"""
from typing import Optional
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist

from ..core.acquisition_controller import AcquisitionController


class NavigationBridgeNode(Node):
    """ROS 2 Node interfacing with Nav2 stack and handling acquisition slew."""

    def __init__(self):
        super().__init__('navigation_bridge_node')

        self.declare_parameter('max_angular_vel_rad_s', 1.5)
        self.declare_parameter('angular_accel_limit', 2.0)

        max_w = self.get_parameter('max_angular_vel_rad_s').get_parameter_value().double_value
        accel = self.get_parameter('angular_accel_limit').get_parameter_value().double_value

        self.slew_controller = AcquisitionController(
            max_angular_vel_rad_s=max_w, angular_accel_limit=accel
        )

        # Publisher to safety authority input
        self.pub_cmd_vel = self.create_publisher(Twist, 'cmd_vel_input', 10)

        self.get_logger().info("EdgeSync Navigation Bridge Node initialized.")


def main(args=None):
    rclpy.init(args=args)
    node = NavigationBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

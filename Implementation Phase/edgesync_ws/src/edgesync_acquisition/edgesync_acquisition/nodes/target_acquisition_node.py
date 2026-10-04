"""Target Acquisition Node (C-08): Frustum Lock & Search Controller.

SAD v2.3 §3 & Math/Protocol Spec v2.3 §3.2:
- Bounds acquisition window to <= 700 ms (hard cutoff).
- Receives target spatial prior and initiates rotation slew.
- Interrogates IdentityVerify.srv (I-10) to verify visual signature against transferred embedding.
"""
from typing import Optional
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Twist

from edgesync_msgs.msg import HandoffRequest, AcqStatus
from edgesync_msgs.srv import IdentityVerify


class TargetAcquisitionNode(Node):
    """ROS 2 Node controlling chassis interception slew and candidate acquisition."""

    def __init__(self):
        super().__init__('target_acquisition_node')

        self.declare_parameter('acquisition_timeout_ms', 700)
        self.declare_parameter('slew_angular_speed_rad_s', 1.2)
        self.declare_parameter('similarity_threshold', 0.70)

        self.timeout_ms = self.get_parameter('acquisition_timeout_ms').get_parameter_value().integer_value
        self.slew_speed = self.get_parameter('slew_angular_speed_rad_s').get_parameter_value().double_value
        self.sim_thresh = self.get_parameter('similarity_threshold').get_parameter_value().double_value

        # Client for I-10 Identity Verification
        self.cli_verify = self.create_client(IdentityVerify, 'identity/verify')

        # Publisher for advisory motion / orientation slew
        self.pub_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 10)

        self._target_prior: Optional[tuple[float, float, float]] = None
        self._is_acquiring = False

        self.get_logger().info(f"Target Acquisition Node initialized (Hard timeout: {self.timeout_ms} ms).")

    def start_acquisition(self, prior_pos_map: tuple[float, float, float]):
        """Begin intercept acquisition toward target prior."""
        self._target_prior = prior_pos_map
        self._is_acquiring = True
        self.get_logger().info(f"Acquisition initiated toward prior: {prior_pos_map}")


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

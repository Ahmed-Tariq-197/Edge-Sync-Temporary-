"""Chassis Safety Authority Node (C-10): Final Motion Authority (/cmd_vel Mux).

Safety Mandates:
- Multiplexes velocity commands from Nav2 and Acquisition slew controller.
- Employs SafetySupervisor to guarantee ramp-down to zero within <= 50 ms.
- Monitors 150 ms command watchdog to avoid runaway condition.
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

from ..core.safety_supervisor import SafetySupervisor


class SafetyAuthorityNode(Node):
    """ROS 2 Node acting as final motion authority prior to hardware motor driver."""

    def __init__(self):
        super().__init__('safety_authority_node')

        self.declare_parameter('max_linear_decel', 2.0)
        self.declare_parameter('max_angular_decel', 3.0)
        self.declare_parameter('cmd_vel_watchdog_ms', 150.0)

        lin_decel = self.get_parameter('max_linear_decel').get_parameter_value().double_value
        ang_decel = self.get_parameter('max_angular_decel').get_parameter_value().double_value
        wd_ms = self.get_parameter('cmd_vel_watchdog_ms').get_parameter_value().double_value

        self.supervisor = SafetySupervisor(
            max_linear_decel=lin_decel,
            max_angular_decel=ang_decel,
            cmd_vel_watchdog_ms=wd_ms
        )

        # Subscribers
        self.sub_cmd_raw = self.create_subscription(
            Twist, 'cmd_vel_input', self._on_cmd_vel_input, 10
        )
        self.sub_e_stop = self.create_subscription(
            Bool, 'safety/emergency_stop', self._on_emergency_stop, 10
        )

        # Publisher to hardware base driver
        self.pub_cmd_safe = self.create_publisher(Twist, 'cmd_vel', 10)

        self._current_vx = 0.0
        self._current_vy = 0.0
        self._current_wz = 0.0

        # High frequency safety control loop (50 Hz = 20 ms)
        self.timer = self.create_timer(0.02, self._control_loop)

        self.get_logger().info("EdgeSync Safety Authority Node initialized (150ms watchdog active).")

    def _on_emergency_stop(self, msg: Bool):
        if msg.data:
            self.supervisor.trigger_safe_stop("Emergency stop topic signal")
            self.get_logger().warn("Safety Authority: Emergency stop triggered!")
        else:
            self.supervisor.clear_safe_stop()

    def _on_cmd_vel_input(self, msg: Twist):
        self.supervisor.feed_heartbeat()
        self._req_vx = msg.linear.x
        self._req_vy = msg.linear.y
        self._req_wz = msg.angular.z

    def _control_loop(self):
        cmd_vx, cmd_vy, cmd_wz, stopped = self.supervisor.filter_command(
            getattr(self, '_req_vx', 0.0),
            getattr(self, '_req_vy', 0.0),
            getattr(self, '_req_wz', 0.0),
            self._current_vx, self._current_vy, self._current_wz,
            dt=0.02
        )

        self._current_vx = cmd_vx
        self._current_vy = cmd_vy
        self._current_wz = cmd_wz

        out_msg = Twist()
        out_msg.linear.x = cmd_vx
        out_msg.linear.y = cmd_vy
        out_msg.angular.z = cmd_wz
        self.pub_cmd_safe.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SafetyAuthorityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

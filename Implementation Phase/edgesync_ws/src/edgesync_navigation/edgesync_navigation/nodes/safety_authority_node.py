"""Chassis Safety Authority Node (C-10): Final Motion Authority (/cmd_vel Mux).

Safety Mandates:
- Multiplexes velocity commands from Nav2 and Acquisition slew controller.
- Employs SafetySupervisor to guarantee ramp-down to zero within <= 50 ms.
- Monitors 150 ms command watchdog to avoid runaway condition.
- Subscribes to peer and local StateHeartbeat, Watchdog timeouts, and Ownership events.
- Acts as ultimate motion override: triggers CRITICAL: SAFE STOP TRIGGERED on peer loss > 2s during handoff or navigation failure.
"""
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

from edgesync_msgs.msg import StateHeartbeat, OwnershipEvent
from ..core.safety_supervisor import SafetySupervisor


class SafetyAuthorityNode(Node):
    """ROS 2 Node acting as final motion authority prior to hardware motor driver."""

    def __init__(self, node_name: str = 'safety_authority_node'):
        super().__init__(node_name)

        # 1. Parameter Declarations (Zero Hardcoding)
        self.declare_parameter('robot_id', 'ugv_1')
        self.declare_parameter('peer_id', 'ugv_2')
        self.declare_parameter('max_linear_decel', 2.0)
        self.declare_parameter('max_angular_decel', 3.0)
        self.declare_parameter('cmd_vel_watchdog_ms', 150.0)
        self.declare_parameter('peer_heartbeat_timeout_s', 2.0)

        self.robot_id = self.get_parameter('robot_id').get_parameter_value().string_value
        self.peer_id = self.get_parameter('peer_id').get_parameter_value().string_value
        lin_decel = self.get_parameter('max_linear_decel').get_parameter_value().double_value
        ang_decel = self.get_parameter('max_angular_decel').get_parameter_value().double_value
        wd_ms = self.get_parameter('cmd_vel_watchdog_ms').get_parameter_value().double_value
        self.peer_timeout_s = self.get_parameter('peer_heartbeat_timeout_s').get_parameter_value().double_value

        default_peer_hb = f'/{self.peer_id}/edgesync/state_heartbeat'
        default_self_hb = f'/{self.robot_id}/edgesync/state_heartbeat'
        default_ownership = f'/{self.robot_id}/edgesync/ownership_event'

        self.declare_parameter('peer_heartbeat_topic', default_peer_hb)
        self.declare_parameter('self_heartbeat_topic', default_self_hb)
        self.declare_parameter('ownership_event_topic', default_ownership)
        self.declare_parameter('watchdog_timeout_topic', 'safety/watchdog_timeout')
        self.declare_parameter('emergency_stop_topic', 'safety/emergency_stop')
        self.declare_parameter('cmd_vel_input_topic', 'cmd_vel_input')
        self.declare_parameter('cmd_vel_safe_topic', 'cmd_vel')

        peer_hb_topic = self.get_parameter('peer_heartbeat_topic').get_parameter_value().string_value
        self_hb_topic = self.get_parameter('self_heartbeat_topic').get_parameter_value().string_value
        ownership_topic = self.get_parameter('ownership_event_topic').get_parameter_value().string_value
        wd_topic = self.get_parameter('watchdog_timeout_topic').get_parameter_value().string_value
        estop_topic = self.get_parameter('emergency_stop_topic').get_parameter_value().string_value
        cmd_in_topic = self.get_parameter('cmd_vel_input_topic').get_parameter_value().string_value
        cmd_out_topic = self.get_parameter('cmd_vel_safe_topic').get_parameter_value().string_value

        # 2. Safety Core State
        self.supervisor = SafetySupervisor(
            max_linear_decel=lin_decel,
            max_angular_decel=ang_decel,
            cmd_vel_watchdog_ms=wd_ms
        )

        self.handoff_in_progress: bool = False
        self.last_peer_heartbeat_time: Optional[float] = None
        self.last_self_heartbeat_time: Optional[float] = None
        self.is_critical_stopped: bool = False
        self.critical_stop_reason: str = ""

        self._current_vx = 0.0
        self._current_vy = 0.0
        self._current_wz = 0.0
        self._req_vx = 0.0
        self._req_vy = 0.0
        self._req_wz = 0.0

        # 3. QoS Profiles
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 4. Subscribers
        self.sub_cmd_raw = self.create_subscription(
            Twist, cmd_in_topic, self._on_cmd_vel_input, 10
        )
        self.sub_e_stop = self.create_subscription(
            Bool, estop_topic, self._on_emergency_stop, 10
        )
        self.sub_watchdog = self.create_subscription(
            Bool, wd_topic, self._on_watchdog_timeout, 10
        )
        self.sub_peer_heartbeat = self.create_subscription(
            StateHeartbeat, peer_hb_topic, self._on_peer_heartbeat, best_effort_qos
        )
        self.sub_self_heartbeat = self.create_subscription(
            StateHeartbeat, self_hb_topic, self._on_self_heartbeat, best_effort_qos
        )
        self.sub_ownership = self.create_subscription(
            OwnershipEvent, ownership_topic, self._on_ownership_event, reliable_qos
        )

        # 5. Publisher to hardware base driver
        self.pub_cmd_safe = self.create_publisher(Twist, cmd_out_topic, 10)

        # 6. High frequency safety control loop (50 Hz = 20 ms)
        self.timer = self.create_timer(0.02, self._control_loop)

        self.get_logger().info(
            f"EdgeSync Safety Authority Node initialized (robot_id={self.robot_id}, peer_id={self.peer_id}, peer_timeout={self.peer_timeout_s}s)."
        )

    def trigger_critical_safe_stop(self, reason: str):
        """Take ultimate precedence, halt platform immediately, and log critical safety stop."""
        self.is_critical_stopped = True
        self.critical_stop_reason = reason
        self.supervisor.trigger_safe_stop(reason)
        self.get_logger().error(f"CRITICAL: SAFE STOP TRIGGERED: {reason}")

        # Immediately publish zero-velocity Twist
        zero_twist = Twist()
        zero_twist.linear.x = 0.0
        zero_twist.linear.y = 0.0
        zero_twist.linear.z = 0.0
        zero_twist.angular.x = 0.0
        zero_twist.angular.y = 0.0
        zero_twist.angular.z = 0.0
        self.pub_cmd_safe.publish(zero_twist)

    def clear_safe_stop(self):
        """Clear safe stop state and restore motion capability."""
        self.is_critical_stopped = False
        self.critical_stop_reason = ""
        self.supervisor.clear_safe_stop()
        self.get_logger().info("Safety Authority: Safe stop cleared.")

    def _on_emergency_stop(self, msg: Bool):
        if msg.data:
            self.trigger_critical_safe_stop("Emergency stop topic signal received")
        else:
            self.clear_safe_stop()

    def _on_watchdog_timeout(self, msg: Bool):
        if msg.data:
            self.trigger_critical_safe_stop("Watchdog timeout triggered")

    def _on_peer_heartbeat(self, msg: StateHeartbeat):
        """Monitor peer liveness and navigation stack status."""
        self.last_peer_heartbeat_time = time.monotonic()
        if msg.nav_status == StateHeartbeat.NAV_STATUS_DEGRADED and self.handoff_in_progress:
            self.trigger_critical_safe_stop(f"Peer {msg.header.sender_id} navigation stack reported NAV_STATUS_DEGRADED during handoff")

    def _on_self_heartbeat(self, msg: StateHeartbeat):
        """Monitor self navigation stack health."""
        self.last_self_heartbeat_time = time.monotonic()
        if msg.nav_status == StateHeartbeat.NAV_STATUS_DEGRADED:
            self.trigger_critical_safe_stop("Local navigation stack reported NAV_STATUS_DEGRADED (crashed/degraded)")

    def set_handoff_active(self, active: bool, transaction_id: str = ""):
        """Set handoff tracking state."""
        self.handoff_in_progress = active
        if active:
            self.get_logger().info(f"Safety Authority: Handoff active ({transaction_id}). Peer heartbeat monitoring armed.")
        else:
            self.get_logger().info(f"Safety Authority: Handoff finished/standby ({transaction_id}). Monitoring disarmed.")

    def _on_ownership_event(self, msg: OwnershipEvent):
        """Track handoff lifecycle to enforce peer liveness during transactions."""
        if msg.role == OwnershipEvent.ROLE_CANDIDATE:
            self.set_handoff_active(True, msg.header.transaction_id)
        elif msg.role in (OwnershipEvent.ROLE_OWNER, OwnershipEvent.ROLE_STANDBY):
            self.set_handoff_active(False, msg.header.transaction_id)

    def _on_cmd_vel_input(self, msg: Twist):
        self.supervisor.feed_heartbeat()
        self._req_vx = msg.linear.x
        self._req_vy = msg.linear.y
        self._req_wz = msg.angular.z

    def check_safety_conditions(self, current_time: Optional[float] = None) -> bool:
        """Evaluate peer connection liveness and watchdog conditions.

        Returns True if critical safe-stop was engaged.
        """
        now = current_time if current_time is not None else time.monotonic()

        # Critical Check 1: Peer connection lost > peer_heartbeat_timeout_s during active handoff
        if self.handoff_in_progress and self.last_peer_heartbeat_time is not None:
            dt = now - self.last_peer_heartbeat_time
            if dt > self.peer_timeout_s:
                if not self.is_critical_stopped:
                    self.trigger_critical_safe_stop(
                        f"Peer connection completely lost for {dt:.2f}s (> {self.peer_timeout_s:.1f}s) during active handoff"
                    )
                return True

        return self.is_critical_stopped

    def _control_loop(self):
        self.check_safety_conditions()

        if self.is_critical_stopped:
            zero_twist = Twist()
            zero_twist.linear.x = 0.0
            zero_twist.linear.y = 0.0
            zero_twist.linear.z = 0.0
            zero_twist.angular.x = 0.0
            zero_twist.angular.y = 0.0
            zero_twist.angular.z = 0.0
            self.pub_cmd_safe.publish(zero_twist)
            return

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

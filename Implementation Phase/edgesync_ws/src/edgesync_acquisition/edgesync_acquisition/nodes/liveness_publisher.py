"""Liveness Publisher (C-08): 10 Hz ACQ_STATUS in Dedicated Callback Group.

Math/Protocol Spec v2.3 §3.3 & §5.2:
- Publishes AcqStatus.msg at strictly 10 Hz (100 ms period).
- Runs in an isolated MutuallyExclusiveCallbackGroup to prevent timer starvation from vision/EKF tasks.
- QoS: BEST_EFFORT, deadline 150 ms, liveliness lease 300 ms.
"""
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, Duration

from edgesync_msgs.msg import HandoffHeader, AcqStatus


class LivenessPublisher(Node):
    """Publishes continuous 10 Hz acquisition liveness heartbeats."""

    def __init__(self):
        super().__init__('liveness_publisher')

        # Parameters
        self.declare_parameter('robot_id', 'ugv_2')
        self.declare_parameter('peer_id', 'ugv_1')
        self.declare_parameter('publish_rate_hz', 10.0)

        self.robot_id = self.get_parameter('robot_id').get_parameter_value().string_value
        self.peer_id = self.get_parameter('peer_id').get_parameter_value().string_value
        rate_hz = self.get_parameter('publish_rate_hz').get_parameter_value().double_value

        # Dedicated Callback Group (Non-blocking isolated timer)
        self.cb_group = MutuallyExclusiveCallbackGroup()

        # Strict Liveness QoS Profile
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            deadline=Duration(seconds=0, nanoseconds=150000000),       # 150 ms deadline
            lifespan=Duration(seconds=0, nanoseconds=300000000)        # 300 ms lease
        )

        self.pub_status = self.create_publisher(
            AcqStatus, f'/{self.peer_id}/edgesync/acq_status', qos, callback_group=self.cb_group
        )

        self._current_status = AcqStatus.STATUS_SLEWING
        self._confidence = 0.50
        self._is_active = False
        self._txn_id = ""
        self._epoch = 1

        # 10 Hz Timer
        period_s = 1.0 / rate_hz
        self.timer = self.create_timer(
            period_s, self._on_timer_tick, callback_group=self.cb_group
        )

        self.get_logger().info(f"EdgeSync Liveness Publisher initialized at {rate_hz:.1f} Hz.")

    def set_active(self, active: bool, txn_id: str = "", epoch: int = 1):
        """Enable/disable heartbeat streaming during handoff."""
        self._is_active = active
        self._txn_id = txn_id
        self._epoch = epoch

    def update_status(self, status: int, confidence: float):
        """Update current acquisition status (SLEWING, SEARCHING, ACQUIRED)."""
        self._current_status = status
        self._confidence = float(confidence)

    def _on_timer_tick(self):
        """Publish 10 Hz heartbeat."""
        if not self._is_active:
            return

        msg = AcqStatus()
        msg.header.protocol_version = 0x0203
        msg.header.message_type = HandoffHeader.MSG_TYPE_ACQ_STATUS
        msg.header.sender_id = self.robot_id
        msg.header.receiver_id = self.peer_id
        msg.header.transaction_id = self._txn_id
        msg.header.ownership_epoch = self._epoch
        msg.header.stamp_utc_ns = self.get_clock().now().nanoseconds
        msg.status = self._current_status
        msg.confidence = self._confidence

        self.pub_status.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LivenessPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

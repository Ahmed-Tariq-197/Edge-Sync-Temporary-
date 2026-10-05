"""Monitoring Node (C-11): Platform Resource Telemetry & Thesis Black Box Logger.

Monitors & Evaluates:
- Host CPU load percentage and battery status (1 Hz StateHeartbeat).
- Live TTO estimation telemetry (/{robot_id}/edgesync/tto/estimate).
- End-to-end handoff transactions:
  * Negotiation latency (Request -> Response)
  * Receiver Re-ID verification latency (Response -> Confirm)
  * Total transaction execution latency (Request -> ConfirmAck)
  * Dual-ownership overlap duration (Yield -> Claim <= 200ms bound)
- Logs structured records to CSV and SQLite database for thesis Tier 3 experimental validation.
"""
from typing import Optional, Dict, Any
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from edgesync_msgs.msg import (
    HandoffHeader, StateHeartbeat, TTOEstimate,
    HandoffRequest, HandoffResponse, HandoffConfirm,
    HandoffConfirmAck, HandoffCancel, OwnershipEvent, AcqStatus
)
from ..logging.event_logger import ThesisMetricsLogger


class MonitoringNode(Node):
    """ROS 2 Node publishing periodic health telemetry and acting as Black Box evaluation recorder."""

    def __init__(self, node_name: str = 'monitoring_node'):
        super().__init__(node_name)

        # 1. Parameter Declarations (Zero Hardcoding)
        self.declare_parameter('robot_id', 'ugv_1')
        self.declare_parameter('peer_id', 'ugv_2')
        self.declare_parameter('publish_rate_hz', 1.0)
        self.declare_parameter('log_dir', '/tmp/edgesync_logs')
        self.declare_parameter('session_id', '')
        self.declare_parameter('protocol_version', 0x0203)
        self.declare_parameter('enable_thesis_logging', True)

        self.robot_id = self.get_parameter('robot_id').get_parameter_value().string_value
        self.peer_id = self.get_parameter('peer_id').get_parameter_value().string_value
        rate = self.get_parameter('publish_rate_hz').get_parameter_value().double_value
        self.log_dir = self.get_parameter('log_dir').get_parameter_value().string_value
        session = self.get_parameter('session_id').get_parameter_value().string_value
        self.session_id = session if session else f"thesis_run_{int(time.time())}"
        self.proto_ver = self.get_parameter('protocol_version').get_parameter_value().integer_value
        self.enable_logging = self.get_parameter('enable_thesis_logging').get_parameter_value().bool_value

        # 2. Black Box Metrics Logger
        self.metrics_logger = ThesisMetricsLogger(
            log_dir=self.log_dir,
            session_id=self.session_id
        )

        # In-memory transaction tracking
        self.active_transactions: Dict[str, Dict[str, Any]] = {}
        self.latest_tto_safe_s: float = 0.0

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

        # 4. Publishers
        self.pub_heartbeat = self.create_publisher(
            StateHeartbeat, f'/{self.robot_id}/edgesync/state_heartbeat', best_effort_qos
        )

        # 5. Subscribers for Live Protocol Exchanges & Black Box Recording
        # TTO Estimation
        self.sub_tto = self.create_subscription(
            TTOEstimate, f'/{self.robot_id}/edgesync/tto/estimate', self._on_tto_estimate, best_effort_qos
        )

        # Handoff Handshake Exchanges (Listen to self and peer channels)
        for r_id in (self.robot_id, self.peer_id):
            self.create_subscription(
                HandoffRequest, f'/{r_id}/edgesync/handoff_request', self._on_handoff_request, reliable_qos
            )
            self.create_subscription(
                HandoffResponse, f'/{r_id}/edgesync/handoff_response', self._on_handoff_response, reliable_qos
            )
            self.create_subscription(
                HandoffConfirm, f'/{r_id}/edgesync/handoff_confirm', self._on_handoff_confirm, reliable_qos
            )
            self.create_subscription(
                HandoffConfirmAck, f'/{r_id}/edgesync/handoff_confirm_ack', self._on_handoff_confirm_ack, reliable_qos
            )
            self.create_subscription(
                HandoffCancel, f'/{r_id}/edgesync/handoff_cancel', self._on_handoff_cancel, reliable_qos
            )
            self.create_subscription(
                OwnershipEvent, f'/{r_id}/edgesync/ownership_event', self._on_ownership_event, reliable_qos
            )
            self.create_subscription(
                AcqStatus, f'/{r_id}/edgesync/acq_status', self._on_acq_status, best_effort_qos
            )

        # 6. Periodic 1 Hz Telemetry Timer
        self.timer = self.create_timer(1.0 / rate, self._publish_telemetry)
        self.get_logger().info(
            f"EdgeSync Monitoring & Black Box Node initialized ({self.robot_id}, log_dir={self.metrics_logger.log_dir})."
        )

    def _read_cpu_and_battery(self) -> tuple[float, float]:
        """Read system CPU percentage and simulated/real battery status."""
        try:
            import psutil
            cpu = float(psutil.cpu_percent(interval=None))
            batt = psutil.sensors_battery()
            battery_pct = float(batt.percent) if batt else 95.0
        except Exception:
            cpu = 25.0
            battery_pct = 95.0
        return cpu, battery_pct

    def _publish_telemetry(self):
        """Broadcast 1 Hz platform health telemetry."""
        cpu, batt = self._read_cpu_and_battery()

        msg = StateHeartbeat()
        msg.header.protocol_version = self.proto_ver
        msg.header.message_type = HandoffHeader.MSG_TYPE_STATE_HEARTBEAT
        msg.header.sender_id = self.robot_id
        msg.header.receiver_id = "*"
        msg.header.stamp_utc_ns = self.get_clock().now().nanoseconds
        msg.role = StateHeartbeat.ROLE_OWNER
        msg.cpu_load = cpu
        msg.battery_pct = batt
        msg.nav_status = StateHeartbeat.NAV_STATUS_AVAILABLE

        self.pub_heartbeat.publish(msg)

    def _on_tto_estimate(self, msg: TTOEstimate):
        """Track live TTO prediction telemetry."""
        self.latest_tto_safe_s = float(msg.tto_safe_s)
        if self.enable_logging:
            target_id = getattr(msg, 'target_id', 'target_human')
            stamp = getattr(msg, 'stamp_utc_ns', self.get_clock().now().nanoseconds)
            tto_hat = getattr(msg, 'tto_hat_s', float(msg.tto_safe_s))
            self.metrics_logger.log_tto_telemetry(
                robot_id=self.robot_id,
                target_id=target_id,
                tto_safe_s=float(msg.tto_safe_s),
                tto_mean_s=float(tto_hat),
                distance_m=0.0,
                timestamp_utc_ns=stamp
            )

    def _on_handoff_request(self, msg: HandoffRequest):
        txn_id = msg.header.transaction_id
        if not txn_id:
            return
        rec = self.active_transactions.setdefault(txn_id, {})
        rec['txn_id'] = txn_id
        rec['sender_id'] = msg.header.sender_id
        rec['receiver_id'] = msg.header.receiver_id
        rec['t_request_ns'] = msg.header.stamp_utc_ns or self.get_clock().now().nanoseconds
        rec['tto_safe_s'] = self.latest_tto_safe_s
        rec['epoch'] = msg.header.ownership_epoch

    def _on_handoff_response(self, msg: HandoffResponse):
        txn_id = msg.header.transaction_id
        if not txn_id:
            return
        rec = self.active_transactions.setdefault(txn_id, {})
        rec['t_response_ns'] = msg.header.stamp_utc_ns or self.get_clock().now().nanoseconds

    def _on_acq_status(self, msg: AcqStatus):
        txn_id = msg.header.transaction_id
        if not txn_id:
            return
        rec = self.active_transactions.setdefault(txn_id, {})
        if msg.status == AcqStatus.STATUS_ACQUIRED:
            rec['t_acquired_ns'] = msg.header.stamp_utc_ns or self.get_clock().now().nanoseconds

    def _on_handoff_confirm(self, msg: HandoffConfirm):
        txn_id = msg.header.transaction_id
        if not txn_id:
            return
        rec = self.active_transactions.setdefault(txn_id, {})
        rec['t_confirm_ns'] = msg.header.stamp_utc_ns or self.get_clock().now().nanoseconds
        rec['reid_similarity'] = float(msg.reid_similarity)

    def _on_ownership_event(self, msg: OwnershipEvent):
        txn_id = msg.header.transaction_id
        if not txn_id:
            return
        rec = self.active_transactions.setdefault(txn_id, {})
        ts = msg.header.stamp_utc_ns or self.get_clock().now().nanoseconds
        if msg.role == OwnershipEvent.ROLE_CANDIDATE:
            rec['t_yield_ns'] = ts
        elif msg.role == OwnershipEvent.ROLE_OWNER:
            rec['t_claim_ns'] = ts

    def _on_handoff_confirm_ack(self, msg: HandoffConfirmAck):
        txn_id = msg.header.transaction_id
        if not txn_id:
            return
        rec = self.active_transactions.setdefault(txn_id, {})
        rec['t_ack_ns'] = self.get_clock().now().nanoseconds
        rec['status'] = "SUCCESS"
        self._finalize_and_log_transaction(txn_id)

    def _on_handoff_cancel(self, msg: HandoffCancel):
        txn_id = msg.header.transaction_id
        if not txn_id:
            return
        rec = self.active_transactions.setdefault(txn_id, {})
        rec['t_cancel_ns'] = msg.header.stamp_utc_ns or self.get_clock().now().nanoseconds
        rec['status'] = f"CANCELLED_{msg.reason}"
        self._finalize_and_log_transaction(txn_id)

    def _finalize_and_log_transaction(self, txn_id: str):
        rec = self.active_transactions.get(txn_id)
        if not rec or rec.get('logged', False):
            return

        rec['logged'] = True
        t_req = rec.get('t_request_ns')
        t_resp = rec.get('t_response_ns')
        t_conf = rec.get('t_confirm_ns')
        t_ack = rec.get('t_ack_ns')
        t_cancel = rec.get('t_cancel_ns')
        t_yield = rec.get('t_yield_ns')
        t_claim = rec.get('t_claim_ns')

        # 1. Negotiation Latency (Request -> Response)
        neg_latency_ms = (t_resp - t_req) / 1e6 if (t_resp and t_req and t_resp >= t_req) else 0.0

        # 2. Re-ID Latency (Response -> Confirm)
        reid_latency_ms = (t_conf - t_resp) / 1e6 if (t_conf and t_resp and t_conf >= t_resp) else 0.0

        # 3. Total Transaction Latency (Request -> Ack / Cancel)
        t_end = t_ack or t_cancel or t_conf
        total_latency_ms = (t_end - t_req) / 1e6 if (t_end and t_req and t_end >= t_req) else 0.0

        # 4. Dual-Ownership Overlap Duration (Yield -> Claim)
        dual_overlap_ms = max(0.0, (t_claim - t_yield) / 1e6) if (t_claim and t_yield and t_claim >= t_yield) else 0.0

        status = rec.get('status', 'SUCCESS')
        sim = rec.get('reid_similarity', 0.0)
        tto = rec.get('tto_safe_s', 0.0)
        robot_id = rec.get('sender_id', self.robot_id)
        peer_id = rec.get('receiver_id', self.peer_id)

        if self.enable_logging:
            self.metrics_logger.log_handoff_metric(
                txn_id=txn_id,
                robot_id=robot_id,
                peer_id=peer_id,
                status=status,
                negotiation_latency_ms=neg_latency_ms,
                reid_latency_ms=reid_latency_ms,
                total_transaction_latency_ms=total_latency_ms,
                dual_ownership_overlap_ms=dual_overlap_ms,
                tto_safe_s=tto,
                reid_similarity=sim,
                protocol_version=self.proto_ver,
                timestamp_utc_ns=t_end or self.get_clock().now().nanoseconds
            )

        self.get_logger().info(
            f"THESIS RECORD COMMITTED: txn={txn_id} status={status} total={total_latency_ms:.2f}ms "
            f"neg={neg_latency_ms:.2f}ms reid={reid_latency_ms:.2f}ms dual_overlap={dual_overlap_ms:.2f}ms"
        )


def main(args=None):
    rclpy.init(args=args)
    node = MonitoringNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

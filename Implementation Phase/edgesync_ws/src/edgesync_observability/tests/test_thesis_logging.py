"""Unit tests for Black Box Thesis Evaluation Logging (edgesync_observability).

Verifies:
1. ThesisMetricsLogger writes structured records to both CSV and SQLite.
2. Querying SQLite summary statistics produces accurate aggregations.
3. MonitoringNode tracks transaction lifecycle from HandoffRequest -> ConfirmAck.
4. Latencies (negotiation, Re-ID, total) and dual-ownership overlaps are computed and committed.
5. Handoff cancellations are logged with cancel reasons.
"""
import os
import tempfile
import sqlite3
import pytest
import rclpy

from edgesync_msgs.msg import (
    HandoffHeader, HandoffRequest, HandoffResponse,
    HandoffConfirm, HandoffConfirmAck, HandoffCancel,
    OwnershipEvent, TTOEstimate
)
from edgesync_observability.logging.event_logger import ThesisMetricsLogger
from edgesync_observability.nodes.monitoring_node import MonitoringNode


@pytest.fixture(scope="module", autouse=True)
def rclpy_init():
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def test_thesis_metrics_logger_csv_and_sqlite():
    """Verify ThesisMetricsLogger creates CSV and SQLite records with proper schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ThesisMetricsLogger(log_dir=tmpdir, session_id="test_session_1")

        assert os.path.exists(logger.csv_path)
        assert os.path.exists(logger.db_path)

        # Log sample handoff
        logger.log_handoff_metric(
            txn_id="txn_test_01",
            robot_id="ugv_1",
            peer_id="ugv_2",
            status="SUCCESS",
            negotiation_latency_ms=45.2,
            reid_latency_ms=120.5,
            total_transaction_latency_ms=280.0,
            dual_ownership_overlap_ms=35.0,
            tto_safe_s=2.35,
            reid_similarity=0.885,
            protocol_version=0x0203
        )

        # Log sample TTO telemetry
        logger.log_tto_telemetry(
            robot_id="ugv_1",
            target_id="target_human",
            tto_safe_s=2.35,
            tto_mean_s=2.80,
            distance_m=1.20
        )

        # 1. Verify CSV contents
        with open(logger.csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 2  # Header + 1 record
            assert "txn_test_01" in lines[1]
            assert "SUCCESS" in lines[1]
            assert "45.2" in lines[1]

        # 2. Verify SQLite contents
        summary = logger.query_metrics_summary()
        assert summary["total_count"] == 1
        assert summary["success_count"] == 1
        assert pytest.approx(summary["avg_negotiation_latency_ms"], abs=0.1) == 45.2
        assert pytest.approx(summary["avg_reid_latency_ms"], abs=0.1) == 120.5
        assert pytest.approx(summary["avg_total_latency_ms"], abs=0.1) == 280.0
        assert pytest.approx(summary["max_dual_overlap_ms"], abs=0.1) == 35.0


def test_monitoring_node_end_to_end_transaction_logging():
    """Verify MonitoringNode aggregates protocol events and commits transaction metrics to disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        node = MonitoringNode(node_name="test_monitoring_blackbox")
        # Direct logger to temporary test directory
        node.metrics_logger = ThesisMetricsLogger(log_dir=tmpdir, session_id="test_m_run")

        txn_id = "txn_eval_999"
        t0_ns = 1_000_000_000  # 1.0s
        t1_ns = 1_050_000_000  # 1.05s (+50ms)
        t2_ns = 1_180_000_000  # 1.18s (+130ms Re-ID)
        t3_ns = 1_220_000_000  # 1.22s (yield)
        t4_ns = 1_250_000_000  # 1.25s (claim: overlap = 30ms)
        t5_ns = 1_270_000_000  # 1.27s (commit ack, total = 270ms)

        # 1. TTO estimate
        tto = TTOEstimate()
        tto.stamp_utc_ns = t0_ns
        tto.target_id = "target_human"
        tto.tto_safe_s = 2.15
        tto.tto_hat_s = 2.50
        node._on_tto_estimate(tto)
        assert pytest.approx(node.latest_tto_safe_s, abs=1e-3) == 2.15

        # 2. HandoffRequest
        req = HandoffRequest()
        req.header.transaction_id = txn_id
        req.header.sender_id = "ugv_1"
        req.header.receiver_id = "ugv_2"
        req.header.stamp_utc_ns = t0_ns
        node._on_handoff_request(req)

        # 3. HandoffResponse
        resp = HandoffResponse()
        resp.header.transaction_id = txn_id
        resp.header.stamp_utc_ns = t1_ns
        node._on_handoff_response(resp)

        # 4. HandoffConfirm
        conf = HandoffConfirm()
        conf.header.transaction_id = txn_id
        conf.header.stamp_utc_ns = t2_ns
        conf.reid_similarity = 0.912
        node._on_handoff_confirm(conf)

        # 5. OwnershipEvents (yield & claim)
        ev_yield = OwnershipEvent()
        ev_yield.header.transaction_id = txn_id
        ev_yield.role = OwnershipEvent.ROLE_CANDIDATE
        ev_yield.header.stamp_utc_ns = t3_ns
        node._on_ownership_event(ev_yield)

        ev_claim = OwnershipEvent()
        ev_claim.header.transaction_id = txn_id
        ev_claim.role = OwnershipEvent.ROLE_OWNER
        ev_claim.header.stamp_utc_ns = t4_ns
        node._on_ownership_event(ev_claim)

        # 6. HandoffConfirmAck (commits transaction)
        ack = HandoffConfirmAck()
        ack.header.transaction_id = txn_id
        node._on_handoff_confirm_ack(ack)

        # Verify transaction logged
        summary = node.metrics_logger.query_metrics_summary()
        assert summary["total_count"] == 1
        assert summary["success_count"] == 1
        assert pytest.approx(summary["avg_negotiation_latency_ms"], abs=1.0) == 50.0
        assert pytest.approx(summary["avg_reid_latency_ms"], abs=1.0) == 130.0
        assert pytest.approx(summary["max_dual_overlap_ms"], abs=1.0) == 30.0

        node.destroy_node()


def test_monitoring_node_cancelled_transaction_logging():
    """Verify MonitoringNode records cancelled transactions with reason."""
    with tempfile.TemporaryDirectory() as tmpdir:
        node = MonitoringNode(node_name="test_monitoring_cancelled")
        node.metrics_logger = ThesisMetricsLogger(log_dir=tmpdir, session_id="test_cancel_run")

        txn_id = "txn_cancel_404"
        t0 = 2_000_000_000
        t1 = 2_350_000_000

        req = HandoffRequest()
        req.header.transaction_id = txn_id
        req.header.sender_id = "ugv_1"
        req.header.receiver_id = "ugv_2"
        req.header.stamp_utc_ns = t0
        node._on_handoff_request(req)

        cancel = HandoffCancel()
        cancel.header.transaction_id = txn_id
        cancel.header.stamp_utc_ns = t1
        cancel.reason = HandoffCancel.REASON_REID_MISMATCH
        node._on_handoff_cancel(cancel)

        # Verify logged as cancelled
        with sqlite3.connect(node.metrics_logger.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status, total_transaction_latency_ms FROM handoff_metrics WHERE txn_id = ?", (txn_id,))
            row = cursor.fetchone()
            assert row is not None
            assert "CANCELLED" in row[0]
            assert pytest.approx(row[1], abs=1.0) == 350.0

        node.destroy_node()

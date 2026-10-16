"""Structured Event Logger (C-11) for offline evaluation and latency profiling.

Logs timestamped state transitions, protocol latencies, and resource telemetry
into structured JSON Lines (JSONL) and CSV records.
"""
import os
import json
import csv
import time
from typing import Dict, Any, Optional


class EventLogger:
    """Thread-safe structured event recorder."""

    def __init__(self, log_dir: str = "/tmp/edgesync_logs", session_id: Optional[str] = None):
        self.session_id = session_id if session_id else f"session_{int(time.time())}"
        self.log_dir = os.path.join(log_dir, self.session_id)
        os.makedirs(self.log_dir, exist_ok=True)

        self.jsonl_path = os.path.join(self.log_dir, "events.jsonl")
        self.csv_path = os.path.join(self.log_dir, "transitions.csv")

        # Initialize CSV with header
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp_utc_ns", "robot_id", "txn_id", "event_type", "state_from", "state_to", "epoch", "latency_ms"])

    def log_event(self, event_type: str, txn_id: str, data: Dict[str, Any], robot_id: str = ""):
        """Write structured event to JSONL file."""
        record = {
            "timestamp_utc_ns": time.time_ns(),
            "robot_id": robot_id,
            "txn_id": txn_id,
            "event_type": event_type,
            "data": data
        }
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def log_transition(self, robot_id: str, txn_id: str,
                       state_from: str, state_to: str,
                       epoch: int, latency_ms: float = 0.0):
        """Append state transition row to CSV."""
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.time_ns(), robot_id, txn_id, "STATE_TRANSITION",
                state_from, state_to, epoch, round(latency_ms, 2)
            ])


class ThesisMetricsLogger:
    """Black Box evaluation logger storing transaction metrics into CSV and SQLite."""

    def __init__(self, log_dir: str = "/tmp/edgesync_logs", session_id: Optional[str] = None):
        import sqlite3
        self._sqlite3 = sqlite3
        self.session_id = session_id if session_id else f"session_{int(time.time())}"
        self.log_dir = os.path.join(log_dir, self.session_id)
        os.makedirs(self.log_dir, exist_ok=True)

        self.csv_path = os.path.join(self.log_dir, "handoff_metrics.csv")
        self.db_path = os.path.join(self.log_dir, "thesis_evaluation.db")

        self._init_csv()
        self._init_sqlite()

    def _init_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp_utc_ns",
                    "txn_id",
                    "robot_id",
                    "peer_id",
                    "status",
                    "negotiation_latency_ms",
                    "reid_latency_ms",
                    "total_transaction_latency_ms",
                    "dual_ownership_overlap_ms",
                    "tto_safe_s",
                    "reid_similarity",
                    "protocol_version"
                ])

    def _init_sqlite(self):
        with self._sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS handoff_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc_ns INTEGER,
                    txn_id TEXT,
                    robot_id TEXT,
                    peer_id TEXT,
                    status TEXT,
                    negotiation_latency_ms REAL,
                    reid_latency_ms REAL,
                    total_transaction_latency_ms REAL,
                    dual_ownership_overlap_ms REAL,
                    tto_safe_s REAL,
                    reid_similarity REAL,
                    protocol_version INTEGER
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tto_telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc_ns INTEGER,
                    robot_id TEXT,
                    target_id TEXT,
                    tto_safe_s REAL,
                    tto_mean_s REAL,
                    distance_m REAL
                )
            """)
            conn.commit()

    def log_handoff_metric(self,
                           txn_id: str,
                           robot_id: str,
                           peer_id: str,
                           status: str,
                           negotiation_latency_ms: float,
                           reid_latency_ms: float,
                           total_transaction_latency_ms: float,
                           dual_ownership_overlap_ms: float,
                           tto_safe_s: float,
                           reid_similarity: float,
                           protocol_version: int = 0x0203,
                           timestamp_utc_ns: Optional[int] = None):
        """Append handoff evaluation record to both CSV and SQLite."""
        ts = timestamp_utc_ns if timestamp_utc_ns is not None else time.time_ns()

        # 1. Write CSV
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                ts, txn_id, robot_id, peer_id, status,
                round(negotiation_latency_ms, 2),
                round(reid_latency_ms, 2),
                round(total_transaction_latency_ms, 2),
                round(dual_ownership_overlap_ms, 2),
                round(tto_safe_s, 3),
                round(reid_similarity, 4),
                protocol_version
            ])

        # 2. Write SQLite
        with self._sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO handoff_metrics (
                    timestamp_utc_ns, txn_id, robot_id, peer_id, status,
                    negotiation_latency_ms, reid_latency_ms, total_transaction_latency_ms,
                    dual_ownership_overlap_ms, tto_safe_s, reid_similarity, protocol_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ts, txn_id, robot_id, peer_id, status,
                float(negotiation_latency_ms),
                float(reid_latency_ms),
                float(total_transaction_latency_ms),
                float(dual_ownership_overlap_ms),
                float(tto_safe_s),
                float(reid_similarity),
                int(protocol_version)
            ))
            conn.commit()

    def log_tto_telemetry(self,
                          robot_id: str,
                          target_id: str,
                          tto_safe_s: float,
                          tto_mean_s: float,
                          distance_m: float,
                          timestamp_utc_ns: Optional[int] = None):
        """Record live TTO estimation point to SQLite."""
        ts = timestamp_utc_ns if timestamp_utc_ns is not None else time.time_ns()
        with self._sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO tto_telemetry (
                    timestamp_utc_ns, robot_id, target_id, tto_safe_s, tto_mean_s, distance_m
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (ts, robot_id, target_id, float(tto_safe_s), float(tto_mean_s), float(distance_m)))
            conn.commit()

    def query_metrics_summary(self) -> Dict[str, Any]:
        """Query summary statistics from SQLite database."""
        with self._sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_count,
                    SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) as success_count,
                    AVG(negotiation_latency_ms) as avg_neg,
                    AVG(reid_latency_ms) as avg_reid,
                    AVG(total_transaction_latency_ms) as avg_total,
                    MAX(dual_ownership_overlap_ms) as max_dual_overlap
                FROM handoff_metrics
            """)
            row = cursor.fetchone()
            if row and row[0] > 0:
                return {
                    "total_count": row[0],
                    "success_count": row[1] or 0,
                    "avg_negotiation_latency_ms": round(row[2] or 0.0, 2),
                    "avg_reid_latency_ms": round(row[3] or 0.0, 2),
                    "avg_total_latency_ms": round(row[4] or 0.0, 2),
                    "max_dual_overlap_ms": round(row[5] or 0.0, 2),
                }
            return {
                "total_count": 0,
                "success_count": 0,
                "avg_negotiation_latency_ms": 0.0,
                "avg_reid_latency_ms": 0.0,
                "avg_total_latency_ms": 0.0,
                "max_dual_overlap_ms": 0.0,
            }

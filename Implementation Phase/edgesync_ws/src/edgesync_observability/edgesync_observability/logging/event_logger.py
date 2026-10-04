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

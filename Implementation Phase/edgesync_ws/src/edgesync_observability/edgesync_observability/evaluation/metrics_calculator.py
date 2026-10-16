"""Offline Metrics Calculator (C-14 & Experimental Design v1.1 §8).

Computes:
1. Handoff Success Rate (HSR >= 90%).
2. Zero Unbounded Dual-Ownership Violations (overlap <= T_claim = 200 ms).
3. TTO Prediction Error and Lower Bound Coverage.
4. Execution Latency Percentiles (p50, p90, p99) and Calibrated Safety Margin.
"""
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np


@dataclass
class EvaluationReport:
    total_handoff_attempts: int
    successful_handoffs: int
    failed_handoffs: int
    handoff_success_rate_pct: float
    unbounded_dual_ownership_count: int
    p50_latency_ms: float
    p90_latency_ms: float
    p99_latency_ms: float
    calibrated_safety_margin_ms: float
    tto_mean_absolute_error_s: float
    tto_conservative_coverage_pct: float


class MetricsCalculator:
    """Computes experimental evaluation metrics from log records."""

    @staticmethod
    def calculate_hsr(attempts: int, successes: int) -> float:
        """Handoff Success Rate (HSR >= 90%)."""
        if attempts <= 0:
            return 0.0
        return round((successes / attempts) * 100.0, 2)

    @staticmethod
    def evaluate_dual_ownership(overlap_durations_ms: List[float], t_claim_max_ms: float = 200.0) -> int:
        """Count unbounded dual-ownership occurrences (> T_claim). Zero permitted."""
        unbounded_violations = sum(1 for d in overlap_durations_ms if d > t_claim_max_ms)
        return unbounded_violations

    @staticmethod
    def calculate_calibrated_safety_margin(latencies_ms: List[float], delta_sync_ms: float = 2.0) -> float:
        """Compute Delta_t_safety = 3 * sigma_total + 2 * delta_sync (Math Spec §3.2)."""
        if len(latencies_ms) < 2:
            return 364.0  # Planning baseline fallback

        sigma_total = float(np.std(latencies_ms))
        calibrated_margin = 3.0 * sigma_total + 2.0 * delta_sync_ms
        return round(calibrated_margin, 2)

    @classmethod
    def generate_full_report(cls,
                             latencies_ms: List[float],
                             successes: int,
                             failures: int,
                             overlap_durations_ms: List[float],
                             tto_errors_s: List[float],
                             tto_underestimates_count: int) -> EvaluationReport:
        """Generate comprehensive statistical validation report."""
        attempts = successes + failures
        hsr = cls.calculate_hsr(attempts, successes)
        unbounded_violations = cls.evaluate_dual_ownership(overlap_durations_ms)

        if latencies_ms:
            p50 = float(np.percentile(latencies_ms, 50))
            p90 = float(np.percentile(latencies_ms, 90))
            p99 = float(np.percentile(latencies_ms, 99))
            margin = cls.calculate_calibrated_safety_margin(latencies_ms)
        else:
            p50, p90, p99, margin = 0.0, 0.0, 0.0, 364.0

        mae_tto = float(np.mean(np.abs(tto_errors_s))) if tto_errors_s else 0.0
        coverage = round((tto_underestimates_count / max(1, len(tto_errors_s))) * 100.0, 2)

        return EvaluationReport(
            total_handoff_attempts=attempts,
            successful_handoffs=successes,
            failed_handoffs=failures,
            handoff_success_rate_pct=hsr,
            unbounded_dual_ownership_count=unbounded_violations,
            p50_latency_ms=round(p50, 1),
            p90_latency_ms=round(p90, 1),
            p99_latency_ms=round(p99, 1),
            calibrated_safety_margin_ms=margin,
            tto_mean_absolute_error_s=round(mae_tto, 3),
            tto_conservative_coverage_pct=coverage
        )

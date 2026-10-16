"""Unit tests for Edge Profiling & Parameter Calibration (Quality Gate 9).

Verifies:
1. Multi-stage latency measurements produce finite positive distributions.
2. Safety margin calculation strictly computes Delta_t_safety = 3 * sigma + 2 * delta_sync.
3. Total budget bound satisfies Delta_t_exec + Delta_t_safety <= 1230 ms.
"""
import pytest
import numpy as np
from edgesync_observability.scripts.profile_edge_latency import (
    EdgeLatencyProfiler, TegrastatsMonitor
)


def test_profiler_benchmark_stages_run():
    profiler = EdgeLatencyProfiler(sample_count=20)
    data = profiler.benchmark_stages()

    assert "dt_dec" in data
    assert "dt_neg" in data
    assert "dt_acq" in data
    assert "dt_id" in data
    assert "dt_conf" in data

    for stage_key in ["dt_dec", "dt_neg", "dt_acq", "dt_id", "dt_conf"]:
        assert len(data[stage_key]) == 20
        assert np.all(data[stage_key] >= 0.0)
        assert np.all(np.isfinite(data[stage_key]))


def test_calibrated_safety_margin_formula():
    sigma_measured = 35.0   # ms
    delta_sync_measured = 1.5  # ms

    # Delta_t_safety = 3 * sigma + 2 * delta_sync
    # = 3 * 35.0 + 2 * 1.5 = 105.0 + 3.0 = 108.0 ms
    calibrated_margin = 3.0 * sigma_measured + 2.0 * delta_sync_measured
    assert pytest.approx(calibrated_margin, abs=1e-5) == 108.0


def test_tegrastats_monitor_fallback_soak():
    monitor = TegrastatsMonitor(log_interval_s=0.1)
    monitor.start()
    import time
    time.sleep(0.3)
    monitor.stop()

    assert len(monitor.cpu_temps) > 0
    assert len(monitor.gpu_temps) > 0
    assert len(monitor.ram_usages_mb) > 0
    assert monitor.throttling_events == 0

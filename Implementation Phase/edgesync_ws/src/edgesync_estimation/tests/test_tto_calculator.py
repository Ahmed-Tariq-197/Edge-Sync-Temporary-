"""Unit tests for Analytical TTO Calculator (Quality Gate 4).

Verifies:
1. FOV frustum geometry boundaries (horizontal, vertical, range).
2. Analytical TTO prediction accuracy against known linear trajectories.
3. Safe conservative bound T_safe = max(0, T_hat - k_tau * sigma) with k_tau=2.0.
"""
import pytest
import numpy as np
from edgesync_estimation.core.tto_calculator import AnalyticalTTOCalculator


@pytest.fixture
def tto_calc():
    """Calculator with standard 60 deg HFOV, 49.5 deg VFOV, 5.0m max range."""
    return AnalyticalTTOCalculator(
        hfov_deg=60.0,
        vfov_deg=49.5,
        min_range_m=0.40,
        max_range_m=5.00,
        k_tau_safety=2.0,
        prediction_horizon_s=4.0,
        dt_step_s=0.02
    )


def test_stationary_target_inside_frustum(tto_calc):
    """Stationary target right in front of camera should remain inside up to horizon."""
    # Target at X=0, Y=0, Z=2.0m with zero velocity
    state = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    cov = np.eye(6) * 0.01

    cam_pos = np.array([0.0, 0.0, 0.0])
    cam_rot = np.eye(3)  # Camera aligned with map

    result = tto_calc.compute_tto(state, cov, cam_pos, cam_rot)

    assert result.tto_hat_s == 4.0
    assert result.exit_boundary == "horizon_limit"
    assert result.tto_safe_s > 0.0


def test_lateral_horizontal_exit_predictable_time(tto_calc):
    """Target moving laterally to the right should exit through horizontal_right."""
    # Initial: 2.0m forward on optical axis (Z=2.0, X=0.0).
    # HFOV half angle is 30 degrees.
    # At Z=2.0m, maximum allowed X_c is: 2.0 * tan(30 deg) = 2.0 * 0.57735 = 1.1547m.
    # If target moves right at vx = 1.0 m/s:
    # Exit time should be approximately t = 1.155 seconds.
    state = np.array([0.0, 0.0, 2.0, 1.0, 0.0, 0.0])  # vx = 1.0 m/s
    cov = np.eye(6) * 0.01

    cam_pos = np.array([0.0, 0.0, 0.0])
    cam_rot = np.eye(3)

    result = tto_calc.compute_tto(state, cov, cam_pos, cam_rot)

    assert result.exit_boundary == "horizontal_right"
    assert pytest.approx(result.tto_hat_s, abs=0.05) == 1.155
    # T_safe must be strictly smaller than T_hat (conservative guarantee)
    assert result.tto_safe_s < result.tto_hat_s
    assert result.tto_safe_s >= 0.0


def test_forward_range_max_exit(tto_calc):
    """Target moving straight forward along Z exits range_max at 5.0m."""
    # Start at Z=2.0m, moving forward at vz = 1.5 m/s.
    # Distance to max range (5.0m) is 3.0m.
    # Expected time t = 3.0 / 1.5 = 2.0 seconds.
    state = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 1.5])
    cov = np.eye(6) * 0.01

    cam_pos = np.array([0.0, 0.0, 0.0])
    cam_rot = np.eye(3)

    result = tto_calc.compute_tto(state, cov, cam_pos, cam_rot)

    assert result.exit_boundary == "range_max"
    assert pytest.approx(result.tto_hat_s, abs=0.05) == 2.00
    assert result.tto_safe_s <= result.tto_hat_s

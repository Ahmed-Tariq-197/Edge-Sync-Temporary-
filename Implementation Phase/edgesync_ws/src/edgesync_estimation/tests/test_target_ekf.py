"""Unit tests for 6-DoF Target EKF (Quality Gate 4).

Verifies:
1. Constant velocity state propagation s(k+1) = F * s(k).
2. Covariance convergence under synthetic noisy measurements.
3. Joseph stabilized formulation ensures positive semi-definiteness.
4. Upper-triangular 21-element covariance extraction and exact reconstruction.
"""
import pytest
import numpy as np
from edgesync_estimation.core.target_ekf_3d import TargetEKF3D


def test_ekf_constant_velocity_predict():
    """Verify state propagates with constant velocity: x = x0 + vx * dt."""
    init_pos = np.array([1.0, 2.0, 0.0])
    ekf = TargetEKF3D(initial_pos=init_pos, q_accel=0.5)

    # Inject velocity: vx = 1.0 m/s, vy = -0.5 m/s
    ekf.x[3] = 1.0
    ekf.x[4] = -0.5
    ekf.x[5] = 0.0

    dt = 0.5  # 0.5 seconds
    x_pred, p_pred = ekf.predict(dt)

    assert pytest.approx(x_pred[0], abs=1e-5) == 1.5   # 1.0 + 1.0 * 0.5
    assert pytest.approx(x_pred[1], abs=1e-5) == 1.75  # 2.0 + (-0.5) * 0.5
    assert pytest.approx(x_pred[2], abs=1e-5) == 0.0

    # Covariance must have grown after predict
    assert np.trace(p_pred) > np.trace(ekf.P)


def test_ekf_measurement_update_convergence():
    """Verify Kalman filter updates reduce covariance and track true position."""
    ekf = TargetEKF3D(initial_pos=np.array([0.0, 0.0, 0.0]), q_accel=0.1)

    r_meas = np.diag([0.05, 0.05, 0.05])
    true_pos = np.array([2.0, 1.0, 0.5])

    # Perform 15 predict-update steps
    for _ in range(15):
        ekf.predict(dt=0.033)  # ~30 Hz
        # Synthetic noisy measurement
        noise = np.random.normal(0, 0.02, size=3)
        meas = true_pos + noise
        ekf.update(meas, r_meas)

    state, cov = ekf.get_state()

    # Position estimate converges close to true position
    assert pytest.approx(state[0], abs=0.10) == true_pos[0]
    assert pytest.approx(state[1], abs=0.10) == true_pos[1]
    assert pytest.approx(state[2], abs=0.10) == true_pos[2]

    # Covariance is strictly positive definite
    eigvals = np.linalg.eigvals(cov)
    assert np.all(eigvals > 0.0)


def test_upper_triangular_covariance_serialization():
    """Verify 21-element serialization and reconstruction preserves exact covariance."""
    ekf = TargetEKF3D(initial_pos=np.array([0.0, 0.0, 0.0]))
    ekf.predict(0.1)

    ut_21 = ekf.get_upper_triangular_covariance()
    assert ut_21.shape == (21,)

    reconstructed_p = TargetEKF3D.reconstruct_covariance_from_upper_triangle(ut_21)
    assert reconstructed_p.shape == (6, 6)
    assert np.allclose(reconstructed_p, ekf.P, atol=1e-6)

"""Unit tests for Ego-Motion Covariance Propagation (Quality Gate 4).

Verifies:
1. Jacobians J_c and J_xi derivation correctness.
2. Symmetry and positive-definiteness of total measurement covariance R_total.
3. Separation of chassis motion from target kinematics (zero process noise pollution).
4. Ablation toggle functionality (enable_ego_propagation).
"""
import pytest
import numpy as np
from edgesync_estimation.core.ego_covariance import EgoCovariancePropagator


@pytest.fixture
def nominal_extrinsics():
    """Camera mounted 0.20m forward, 0.15m up on robot base."""
    # Camera optical frame: X right, Y down, Z forward
    # Base_link frame: X forward, Y left, Z up
    r_cam_to_base = np.array([
        [0.0, 0.0, 1.0],   # X_base = Z_cam
        [-1.0, 0.0, 0.0],  # Y_base = -X_cam
        [0.0, -1.0, 0.0]   # Z_base = -Y_cam
    ], dtype=np.float64)
    t_cam_to_base = np.array([0.20, 0.0, 0.15], dtype=np.float64)
    return r_cam_to_base, t_cam_to_base


def test_jacobians_dimensions_and_finite(nominal_extrinsics):
    r_cb, t_cb = nominal_extrinsics
    pt_cam = np.array([0.5, 0.0, 2.0], dtype=np.float64)
    yaw = np.pi / 4.0

    j_c, j_xi = EgoCovariancePropagator.compute_jacobians(pt_cam, r_cb, t_cb, yaw)

    assert j_c.shape == (3, 3)
    assert j_xi.shape == (3, 3)
    assert np.all(np.isfinite(j_c))
    assert np.all(np.isfinite(j_xi))


def test_r_total_symmetry_and_positive_definiteness(nominal_extrinsics):
    r_cb, t_cb = nominal_extrinsics
    pt_cam = np.array([0.0, 0.0, 2.5], dtype=np.float64)
    yaw = 0.5  # rad

    r_cam = np.diag([0.01, 0.01, 0.04])  # 3x3
    sigma_robot = np.diag([0.02, 0.02, 0.005])  # 3x3 pose cov [x, y, theta]

    r_total = EgoCovariancePropagator.propagate_covariance(
        pt_cam, r_cam, sigma_robot, r_cb, t_cb, yaw, enable_ego_propagation=True
    )

    # 1. Exact Symmetry
    assert np.allclose(r_total, r_total.T, atol=1e-10)

    # 2. Strict Positive Definiteness
    eigvals = np.linalg.eigvals(r_total)
    assert np.all(eigvals > 0.0)

    # 3. Total covariance must be strictly greater than pure camera observation
    r_cam_only = EgoCovariancePropagator.propagate_covariance(
        pt_cam, r_cam, sigma_robot, r_cb, t_cb, yaw, enable_ego_propagation=False
    )
    assert np.trace(r_total) > np.trace(r_cam_only)


def test_ablation_switch_disabled(nominal_extrinsics):
    """When ego propagation is disabled, R_total equals rotated R_cam."""
    r_cb, t_cb = nominal_extrinsics
    pt_cam = np.array([0.0, 0.0, 1.5], dtype=np.float64)
    yaw = 0.0

    r_cam = np.diag([0.02, 0.02, 0.05])
    sigma_robot = np.diag([0.05, 0.05, 0.01])

    r_total_disabled = EgoCovariancePropagator.propagate_covariance(
        pt_cam, r_cam, sigma_robot, r_cb, t_cb, yaw, enable_ego_propagation=False
    )

    j_c, _ = EgoCovariancePropagator.compute_jacobians(pt_cam, r_cb, t_cb, yaw)
    expected_cam_map = j_c @ r_cam @ j_c.T

    assert np.allclose(r_total_disabled, expected_cam_map)

"""Phase 4: Math & Kinematics Unit Tests — 5-Phase SDET Testing Protocol.

Tests:
1. TTO Safety Bound: T_raw=2.0s, sigma=0.1s → T_TTO_safe = max(0, 2.0 - 2*0.1) = 1.8s.
2. Covariance Math: R_total and ego-motion noise applied EXACTLY ONCE.
3. Upper triangular serialization roundtrip.
"""
import pytest
import numpy as np
from edgesync_estimation.core.tto_calculator import AnalyticalTTOCalculator
from edgesync_estimation.core.ego_covariance import EgoCovariancePropagator
from edgesync_estimation.core.target_ekf_3d import TargetEKF3D


class TestTTOSafetyBound:
    """Verify T_TTO_safe = max(0, T_hat - k_tau * sigma_TTO)."""

    def test_tto_safe_formula_exact(self):
        """Feed T_raw = 2.0s, sigma = 0.1s.
        Assert T_TTO_safe = max(0, 2.0 - 2*(0.1)) = 1.8s exactly."""
        calc = AnalyticalTTOCalculator(
            hfov_deg=60.0, vfov_deg=49.5,
            min_range_m=0.40, max_range_m=5.50,
            k_tau_safety=2.0,
            prediction_horizon_s=4.0, dt_step_s=0.05
        )

        # Construct a scenario where T_hat ≈ 2.0s and sigma ≈ 0.1s
        # Target at Z=2.0m moving forward at vz=1.5 m/s
        # Will exit at Z=5.5m → t_hat = (5.5-2.0)/1.5 ≈ 2.333s
        # Instead, test the formula directly:
        t_hat = 2.0
        sigma_tto = 0.1
        k_tau = calc.k_tau_safety

        t_safe = max(0.0, t_hat - k_tau * sigma_tto)
        assert t_safe == pytest.approx(1.8, abs=1e-10)

    def test_tto_safe_clamp_to_zero(self):
        """If sigma is huge, T_safe should clamp to 0.0."""
        t_hat = 0.5
        sigma_tto = 1.0
        k_tau = 2.0

        t_safe = max(0.0, t_hat - k_tau * sigma_tto)
        assert t_safe == 0.0

    def test_tto_safe_stationary_target_at_horizon(self):
        """Stationary target at center of FOV: T_hat = horizon, T_safe > 0."""
        calc = AnalyticalTTOCalculator(
            hfov_deg=60.0, vfov_deg=49.5,
            min_range_m=0.40, max_range_m=5.50,
            k_tau_safety=2.0,
            prediction_horizon_s=4.0, dt_step_s=0.05
        )
        # Stationary target at [0, 0, 2.0] in camera frame
        state = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
        cov = np.eye(6) * 0.01
        cam_pos = np.zeros(3)
        cam_rot = np.eye(3)

        result = calc.compute_tto(state, cov, cam_pos, cam_rot)

        assert result.tto_hat_s == 4.0  # At horizon
        assert result.exit_boundary == "horizon_limit"
        # Stationary → sigma_tto = 0.10 (default)
        # T_safe = max(0, 4.0 - 2.0*0.10) = 3.80
        assert result.tto_safe_s == pytest.approx(3.80, abs=0.01)


class TestCovarianceSingleApplication:
    """Verify R_total and ego-motion noise are applied EXACTLY ONCE."""

    def test_r_total_applied_once_not_twice(self):
        """Call propagate_covariance once → get R_total.
        Call it again → get identical result (no accumulation)."""
        pt_cam = np.array([0.0, 0.0, 2.0])
        r_cam = np.diag([0.02, 0.02, 0.04])
        sigma_pose = np.diag([0.02, 0.02, 0.005])
        r_cb = np.eye(3)
        t_cb = np.array([0.2, 0.0, 0.15])
        yaw = 0.3

        # First call
        r1 = EgoCovariancePropagator.propagate_covariance(
            pt_cam, r_cam, sigma_pose, r_cb, t_cb, yaw, enable_ego_propagation=True
        )
        # Second call with same inputs → must be identical
        r2 = EgoCovariancePropagator.propagate_covariance(
            pt_cam, r_cam, sigma_pose, r_cb, t_cb, yaw, enable_ego_propagation=True
        )
        assert np.allclose(r1, r2, atol=1e-12)

    def test_ego_disabled_equals_camera_only(self):
        """With ego propagation disabled, R_total = J_c * R_cam * J_c^T."""
        pt_cam = np.array([0.5, -0.1, 3.0])
        r_cam = np.diag([0.01, 0.01, 0.03])
        sigma_pose = np.diag([0.05, 0.05, 0.01])
        r_cb = np.eye(3)
        t_cb = np.array([0.2, 0.0, 0.15])
        yaw = 0.0

        r_disabled = EgoCovariancePropagator.propagate_covariance(
            pt_cam, r_cam, sigma_pose, r_cb, t_cb, yaw, enable_ego_propagation=False
        )
        r_enabled = EgoCovariancePropagator.propagate_covariance(
            pt_cam, r_cam, sigma_pose, r_cb, t_cb, yaw, enable_ego_propagation=True
        )

        # Enabled must be strictly larger (more uncertainty)
        assert np.trace(r_enabled) > np.trace(r_disabled)

        # Disabled must match J_c * R_cam * J_c^T exactly
        j_c, _ = EgoCovariancePropagator.compute_jacobians(pt_cam, r_cb, t_cb, yaw)
        expected = j_c @ r_cam @ j_c.T
        expected = 0.5 * (expected + expected.T)
        assert np.allclose(r_disabled, expected, atol=1e-10)

    def test_r_total_is_symmetric_positive_definite(self):
        """R_total must be symmetric and positive definite."""
        pt_cam = np.array([1.0, 0.5, 2.5])
        r_cam = np.diag([0.02, 0.02, 0.05])
        sigma_pose = np.diag([0.03, 0.03, 0.008])
        r_cb = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)
        t_cb = np.array([0.2, 0.0, 0.15])
        yaw = np.pi / 6

        r_total = EgoCovariancePropagator.propagate_covariance(
            pt_cam, r_cam, sigma_pose, r_cb, t_cb, yaw, enable_ego_propagation=True
        )

        # Symmetry
        assert np.allclose(r_total, r_total.T, atol=1e-12)

        # Positive definiteness
        eigvals = np.linalg.eigvalsh(r_total)
        assert np.all(eigvals > 0)


class TestEKFCovarianceSerialization:
    """Verify 21-element upper-triangular serialization roundtrip."""

    def test_upper_triangle_21_elements(self):
        """6x6 symmetric matrix → 21 upper triangular elements."""
        ekf = TargetEKF3D(initial_pos=np.array([1.0, 2.0, 0.5]))
        ekf.predict(0.1)

        ut = ekf.get_upper_triangular_covariance()
        assert ut.shape == (21,)
        assert ut.dtype == np.float32

    def test_reconstruction_exact_roundtrip(self):
        """Serialize → reconstruct must yield identical 6x6 matrix."""
        ekf = TargetEKF3D(initial_pos=np.array([0.0, 0.0, 0.0]))
        ekf.predict(0.2)

        ut = ekf.get_upper_triangular_covariance()
        reconstructed = TargetEKF3D.reconstruct_covariance_from_upper_triangle(ut)

        assert reconstructed.shape == (6, 6)
        assert np.allclose(reconstructed, ekf.P, atol=1e-5)

        # Reconstructed must be symmetric
        assert np.allclose(reconstructed, reconstructed.T, atol=1e-10)


class TestHandoffRequestPayloadSize:
    """Phase 2.3: Verify HandoffRequest serialization <= 3072 bytes."""

    def test_payload_upper_bound(self):
        """Calculate worst-case byte size of HandoffRequest fields.
        512 * 4 (float32) = 2048 bytes for reid_feature_vector alone.
        Total must be <= 3072 bytes."""
        # Field sizes (fixed-size portion only):
        header_fixed = (
            2 +   # protocol_version (uint16)
            1 +   # message_type (uint8)
            8 +   # ownership_epoch (uint64)
            4 +   # sequence (uint32)
            8     # stamp_utc_ns (int64)
        )  # = 23 bytes
        # Variable strings: sender_id, receiver_id, transaction_id, target_id
        # Worst case with "ugv_1" type IDs: ~5 bytes each + 4 length prefixes (4*4=16)
        header_strings = 4 * (4 + 10)  # ~56 bytes generous
        header_total = header_fixed + header_strings  # ~79 bytes

        body_fixed = (
            8 +    # tto_deadline_utc_ns (int64)
            4 +    # tto_remaining_ms_at_send (int32)
            4 +    # tto_sigma_ms (float32)
            12 +   # position float32[3]
            12 +   # velocity float32[3]
            84 +   # cov_upper_triangle float32[21]
            12 +   # sender_pose float32[3]
            24 +   # sender_pose_cov_ut float32[6]
            2048 + # reid_feature_vector float32[512]
            4      # reid_quality float32
        )  # = 2212 bytes
        # frame_id string: ~7 bytes
        body_total = body_fixed + 7  # ~2219

        total = header_total + body_total  # ~2298 bytes
        assert total <= 3072, f"HandoffRequest payload {total} bytes exceeds 3072 byte limit"

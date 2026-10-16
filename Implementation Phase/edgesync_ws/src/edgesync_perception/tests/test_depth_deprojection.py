"""Unit tests for DepthDeprojector (TDD Quality Gate 3).

Verifies:
1. Median depth extraction across central 50% ROI with background noise bleed.
2. Pinhole deprojection against known ground-truth 3D coordinates.
3. Robust handling of NaN, zero depth, inf, and out-of-range depths.
4. Camera covariance matrix properties (symmetry, positive-definiteness, range scaling).
"""
import pytest
import numpy as np
from edgesync_perception.math.depth_deprojector import (
    CameraIntrinsics,
    DepthDeprojector,
    DeprojectionResult
)


@pytest.fixture
def astra_intrinsics():
    """Fixture providing Astra Pro Plus RGB-D camera intrinsics."""
    return CameraIntrinsics(
        fx=554.25469,
        fy=554.25469,
        cx=320.5,
        cy=240.5,
        width=640,
        height=480,
        min_depth_m=0.40,
        max_depth_m=6.00,
        depth_scale=0.001
    )


@pytest.fixture
def deprojector(astra_intrinsics):
    return DepthDeprojector(astra_intrinsics)


def test_pinhole_deprojection_center(deprojector, astra_intrinsics):
    """Test optical center ray projects directly along the forward Z axis (X=0, Y=0)."""
    u_center = astra_intrinsics.cx
    v_center = astra_intrinsics.cy
    z_test = 2.0  # 2 meters forward

    point_3d = deprojector.deproject_pixel_to_3d(u_center, v_center, z_test)

    assert isinstance(point_3d, np.ndarray)
    assert point_3d.shape == (3,)
    assert pytest.approx(point_3d[0], abs=1e-5) == 0.0  # X_c
    assert pytest.approx(point_3d[1], abs=1e-5) == 0.0  # Y_c
    assert pytest.approx(point_3d[2], abs=1e-5) == 2.0  # Z_c


def test_pinhole_deprojection_known_quadrants(deprojector, astra_intrinsics):
    """Test top-left, top-right, bottom-left, bottom-right quadrants follow camera frame."""
    z = 1.0
    # Top-Left: u < cx -> X < 0; v < cy -> Y < 0
    tl = deprojector.deproject_pixel_to_3d(100.0, 50.0, z)
    assert tl[0] < 0.0
    assert tl[1] < 0.0
    assert tl[2] == 1.0

    # Bottom-Right: u > cx -> X > 0; v > cy -> Y > 0
    br = deprojector.deproject_pixel_to_3d(500.0, 400.0, z)
    assert br[0] > 0.0
    assert br[1] > 0.0
    assert br[2] == 1.0


def test_central_50_percent_roi_rejection_of_background_bleed(deprojector):
    """Test that background depth outside the central 50% does not corrupt target depth."""
    # Synthetic depth image: Background is 5.0m, Person is 1.8m in the center
    depth_map = np.full((480, 640), 5.0, dtype=np.float32)

    bbox = (200.0, 100.0, 400.0, 300.0)  # w = 200, h = 200
    # Person is located in central 50% (x: 250..350, y: 150..250)
    depth_map[150:250, 250:350] = 1.80

    median_depth, count = deprojector.extract_central_depth(depth_map, bbox)

    assert median_depth is not None
    assert count > 0
    # Should cleanly extract 1.80m despite surrounding 5.0m background bleed
    assert pytest.approx(median_depth, abs=1e-3) == 1.80


def test_handling_of_nan_zero_and_out_of_range(deprojector):
    """Test filter robustly handles invalid sensor readings."""
    depth_map = np.zeros((480, 640), dtype=np.float32)
    bbox = (100.0, 100.0, 200.0, 200.0)

    # 1. All zeros
    median_depth, count = deprojector.extract_central_depth(depth_map, bbox)
    assert median_depth is None
    assert count == 0

    # 2. All NaNs and Infs
    depth_map.fill(np.nan)
    depth_map[120:130, 120:130] = np.inf
    median_depth, count = deprojector.extract_central_depth(depth_map, bbox)
    assert median_depth is None
    assert count == 0

    # 3. Below min range (< 0.40m)
    depth_map.fill(0.25)
    median_depth, count = deprojector.extract_central_depth(depth_map, bbox)
    assert median_depth is None

    # 4. Above max range (> 6.0m)
    depth_map.fill(8.5)
    median_depth, count = deprojector.extract_central_depth(depth_map, bbox)
    assert median_depth is None


def test_uint16_depth_scale_conversion(deprojector):
    """Test raw mm depth maps (uint16) are correctly scaled to metric meters."""
    depth_map_mm = np.full((480, 640), 2500, dtype=np.uint16)  # 2500 mm = 2.5 meters
    bbox = (100.0, 100.0, 300.0, 300.0)

    median_depth, count = deprojector.extract_central_depth(depth_map_mm, bbox)
    assert median_depth is not None
    assert pytest.approx(median_depth, abs=1e-3) == 2.50


def test_covariance_properties_and_range_growth(deprojector):
    """Test R_cam is symmetric, positive definite, and variance grows with range."""
    pt_near = np.array([0.0, 0.0, 1.0])
    pt_far = np.array([0.0, 0.0, 4.0])

    r_near = deprojector.compute_camera_covariance(pt_near)
    r_far = deprojector.compute_camera_covariance(pt_far)

    # Symmetry
    assert np.allclose(r_near, r_near.T)
    assert np.allclose(r_far, r_far.T)

    # Positive Definite (all eigenvalues > 0)
    eig_near = np.linalg.eigvals(r_near)
    eig_far = np.linalg.eigvals(r_far)
    assert np.all(eig_near > 0)
    assert np.all(eig_far > 0)

    # Far measurement has strictly greater uncertainty than near
    assert r_far[0, 0] > r_near[0, 0]  # Var(X)
    assert r_far[1, 1] > r_near[1, 1]  # Var(Y)
    assert r_far[2, 2] > r_near[2, 2]  # Var(Z)


def test_deproject_bbox_end_to_end(deprojector):
    """Test full end-to-end deproject_bbox returns valid DeprojectionResult."""
    depth_map = np.full((480, 640), 2.20, dtype=np.float32)
    bbox = (250.0, 180.0, 390.0, 320.0)

    result = deprojector.deproject_bbox(depth_map, bbox)
    assert isinstance(result, DeprojectionResult)
    assert pytest.approx(result.depth_median, abs=1e-3) == 2.20
    assert result.point_camera.shape == (3,)
    assert result.r_cam.shape == (3, 3)
    assert result.valid_pixel_count > 0

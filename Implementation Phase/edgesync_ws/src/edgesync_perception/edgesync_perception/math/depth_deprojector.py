"""Pure mathematical module for 2D-to-3D depth deprojection and camera covariance.

Aligned with Math/Protocol Spec v2.3 §6 & Master Implementation Plan v1.0 (WP-3).
Zero ROS 2 dependencies in this module to ensure 100% testability via pytest.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 640
    height: int = 480
    min_depth_m: float = 0.40
    max_depth_m: float = 6.00
    depth_scale: float = 0.001  # Convert raw uint16 mm to meters if needed


@dataclass
class DeprojectionResult:
    """Represents a 3D centroid observation in camera optical frame."""
    point_camera: np.ndarray        # [X_c, Y_c, Z_c] in meters (shape: (3,))
    r_cam: np.ndarray               # 3x3 diagonal measurement covariance matrix
    depth_median: float             # Extracted median depth (meters)
    valid_pixel_count: int          # Number of valid depth pixels in central ROI


class DepthDeprojector:
    """Performs depth extraction over central 50% ROI and pinhole deprojection."""

    def __init__(self, intrinsics: CameraIntrinsics,
                 pixel_noise_sigma: float = 1.5,
                 depth_noise_factor: float = 0.015):
        """Initialize deprojector with camera parameters.

        Args:
            intrinsics: CameraIntrinsics with fx, fy, cx, cy and depth limits.
            pixel_noise_sigma: Standard deviation of 2D bounding box localization noise (pixels).
            depth_noise_factor: Empirical depth variance factor where sigma_z = depth_noise_factor * Z^2.
        """
        self.intrinsics = intrinsics
        self.pixel_noise_sigma = pixel_noise_sigma
        self.depth_noise_factor = depth_noise_factor

    def extract_central_depth(self, depth_map: np.ndarray,
                              bbox: Tuple[float, float, float, float]) -> Tuple[Optional[float], int]:
        """Extract median depth from the central 50% bounding box ROI.

        Prevents background bleed by cropping out the boundary 25% on all four sides.

        Args:
            depth_map: HxW float or uint16 depth image.
            bbox: (xmin, ymin, xmax, ymax) in pixel coordinates.

        Returns:
            Tuple of (median_depth_in_meters, valid_pixel_count). If no valid pixels, returns (None, 0).
        """
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin

        if w <= 0 or h <= 0:
            return None, 0

        # Central 50% ROI boundaries
        roi_xmin = int(np.clip(np.floor(xmin + 0.25 * w), 0, depth_map.shape[1] - 1))
        roi_xmax = int(np.clip(np.ceil(xmax - 0.25 * w), 0, depth_map.shape[1]))
        roi_ymin = int(np.clip(np.floor(ymin + 0.25 * h), 0, depth_map.shape[0] - 1))
        roi_ymax = int(np.clip(np.ceil(ymax - 0.25 * h), 0, depth_map.shape[0]))

        if roi_xmax <= roi_xmin or roi_ymax <= roi_ymin:
            return None, 0

        roi_depth = depth_map[roi_ymin:roi_ymax, roi_xmin:roi_xmax].astype(np.float64)

        # If raw uint16 mm depth, convert to meters
        if depth_map.dtype == np.uint16 or np.nanmax(roi_depth) > 50.0:
            roi_depth = roi_depth * self.intrinsics.depth_scale

        # Filter out NaN, Inf, zeros, and out-of-range sensor readings
        valid_mask = (
            np.isfinite(roi_depth) &
            (roi_depth >= self.intrinsics.min_depth_m) &
            (roi_depth <= self.intrinsics.max_depth_m)
        )

        valid_depths = roi_depth[valid_mask]
        valid_count = int(valid_depths.size)

        if valid_count == 0:
            return None, 0

        median_depth = float(np.median(valid_depths))
        return median_depth, valid_count

    def deproject_pixel_to_3d(self, u: float, v: float, z: float) -> np.ndarray:
        """Pinhole deprojection from pixel coordinates (u, v) and metric depth Z.

        Camera Optical Frame Convention:
            X: Right
            Y: Down
            Z: Forward

        Args:
            u: Horizontal pixel coordinate.
            v: Vertical pixel coordinate.
            z: Metric depth along optical axis (meters).

        Returns:
            np.ndarray: [X_c, Y_c, Z_c] 3D coordinates in meters.
        """
        x_c = (u - self.intrinsics.cx) * z / self.intrinsics.fx
        y_c = (v - self.intrinsics.cy) * z / self.intrinsics.fy
        z_c = z
        return np.array([x_c, y_c, z_c], dtype=np.float64)

    def compute_camera_covariance(self, point_camera: np.ndarray) -> np.ndarray:
        """Compute 3x3 measurement covariance R_cam in camera optical frame.

        Uncertainty grows with range Z:
            sigma_x = (Z / fx) * pixel_noise_sigma
            sigma_y = (Z / fy) * pixel_noise_sigma
            sigma_z = depth_noise_factor * Z^2 + 0.01

        Args:
            point_camera: [X_c, Y_c, Z_c]

        Returns:
            np.ndarray: 3x3 diagonal covariance matrix R_cam.
        """
        z = float(max(point_camera[2], self.intrinsics.min_depth_m))

        sigma_x = (z / self.intrinsics.fx) * self.pixel_noise_sigma
        sigma_y = (z / self.intrinsics.fy) * self.pixel_noise_sigma
        sigma_z = self.depth_noise_factor * (z ** 2) + 0.01

        var_x = float(sigma_x ** 2)
        var_y = float(sigma_y ** 2)
        var_z = float(sigma_z ** 2)

        return np.diag([var_x, var_y, var_z])

    def deproject_bbox(self, depth_map: np.ndarray,
                       bbox: Tuple[float, float, float, float]) -> Optional[DeprojectionResult]:
        """Deproject a 2D bounding box and depth map to a 3D centroid observation.

        Args:
            depth_map: Registered depth map (HxW).
            bbox: (xmin, ymin, xmax, ymax) in pixel coordinates.

        Returns:
            DeprojectionResult or None if depth extraction fails.
        """
        median_depth, valid_count = self.extract_central_depth(depth_map, bbox)
        if median_depth is None:
            return None

        # Use bbox center as centroid projection ray
        u_center = (bbox[0] + bbox[2]) / 2.0
        v_center = (bbox[1] + bbox[3]) / 2.0

        point_c = self.deproject_pixel_to_3d(u_center, v_center, median_depth)
        r_cam = self.compute_camera_covariance(point_c)

        return DeprojectionResult(
            point_camera=point_c,
            r_cam=r_cam,
            depth_median=median_depth,
            valid_pixel_count=valid_count
        )

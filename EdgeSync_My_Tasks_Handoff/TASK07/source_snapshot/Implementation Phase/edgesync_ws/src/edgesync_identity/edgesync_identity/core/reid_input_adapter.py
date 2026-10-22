"""TransReID input and asset-path adapters for the identity node (ES-EXP07 Task 7).

These helpers exist so that the sensor-input, camera-conditioning and asset-resolution
behavior required by the TransReID arm is testable without a ROS 2 runtime. They are pure
functions over duck-typed inputs, so a test can pass a plain object with the same fields as
``sensor_msgs/msg/Image``.

Scope: TransReID integration only (charter section 3A items 2, 3, 7). Nothing here changes
OSNet behavior, the ``IdentityVerify.srv`` interface, or the frozen extractor wire contract
in ``interfaces/base_identity_extractor.py``.

Traces: TR-03 (camera conditioning unreachable), TR-04 (image encoding ignored),
TR-05 (asset paths resolved against the process CWD).
"""
from typing import Optional, Tuple
import os

import numpy as np

SUPPORTED_ENCODINGS_BGR = ('bgr8', '8uc3')
SUPPORTED_ENCODINGS_RGB = ('rgb8',)

# TransReID ViT-Base/16 Side Information Embedding table has one entry per Market-1501 camera.
MAX_CAMERA_LABELS = 6


def image_to_rgb(image_msg) -> np.ndarray:
    """Convert a ROS image message into an HxWx3 uint8 RGB array in extractor convention.

    Args:
        image_msg: object exposing ``data``, ``height``, ``width``, ``encoding`` and
                   optionally ``step`` (duck-typed ``sensor_msgs/msg/Image``).

    Returns:
        np.ndarray: (height, width, 3) uint8, RGB channel order.

    Raises:
        ValueError: unsupported encoding, wrong dtype/size, or a non-3-channel shape. The
            caller is expected to record this as an explicit invalid-frame outcome rather
            than substituting an embedding.
    """
    encoding = str(getattr(image_msg, 'encoding', '') or '').lower().split(';')[0].strip()
    height = int(getattr(image_msg, 'height', 0) or 0)
    width = int(getattr(image_msg, 'width', 0) or 0)
    step = int(getattr(image_msg, 'step', 0) or 0)

    if height <= 0 or width <= 0:
        raise ValueError(f"invalid image geometry: {height}x{width}")

    raw = bytes(image_msg.data)
    row_bytes = width * 3
    if step and step != row_bytes:
        if len(raw) < step * height:
            raise ValueError(
                f"image buffer too small: got {len(raw)} bytes, need >= {step * height} "
                f"(step={step}, {height}x{width})"
            )
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, step)[:, :row_bytes]
        arr = np.ascontiguousarray(arr).reshape(height, width, 3)
    else:
        expected = row_bytes * height
        if len(raw) != expected:
            raise ValueError(
                f"image buffer size mismatch for {encoding}: got {len(raw)} bytes, "
                f"expected {expected} for {height}x{width}x3"
            )
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)

    if encoding in SUPPORTED_ENCODINGS_RGB:
        return np.ascontiguousarray(arr)

    if encoding in SUPPORTED_ENCODINGS_BGR:
        return np.ascontiguousarray(arr[:, :, ::-1])

    raise ValueError(
        f"unsupported image encoding '{encoding}': the TransReID extractor consumes "
        f"3-channel 8-bit images; supported: {', '.join(SUPPORTED_ENCODINGS_BGR + SUPPORTED_ENCODINGS_RGB)}"
    )


def resolve_camera_label(reid_camera_id: int) -> Tuple[Optional[int], str]:
    """Map a 1-based Market-1501 camera id to the extractor's 0-based SIE label.

    Args:
        reid_camera_id: Camera id as configured on the robot. 0 or a negative value means
            "no camera information available", which keeps inference camera-agnostic
            (the behavior of the pre-TR-03 node) instead of inventing a label.

    Returns:
        (cam_label, reason): ``cam_label`` is None when SIE must not be applied.
    """
    try:
        cam = int(reid_camera_id)
    except (TypeError, ValueError):
        return None, "reid_camera_id is not an integer - running camera-agnostic"

    if cam <= 0:
        return None, "reid_camera_id unset - running camera-agnostic (no SIE)"
    if cam > MAX_CAMERA_LABELS:
        return None, (f"reid_camera_id={cam} outside 1..{MAX_CAMERA_LABELS} - "
                      f"running camera-agnostic (no SIE)")
    return cam - 1, f"SIE applied with cam_label={cam - 1} (reid_camera_id={cam})"


def resolve_asset_path(path: str, package: str = 'edgesync_bringup') -> Tuple[str, Tuple[str, ...]]:
    """Resolve a model asset path portably, without assuming the process CWD.

    Absolute paths and paths that already resolve from the CWD are returned unchanged so
    existing launch behavior keeps working. Anything else is attempted against the installed
    package share directory. The caller still receives the fail-fast error from the factory;
    this only widens the set of places the asset is looked for, and reports them.

    Returns:
        (resolved_path, attempted_paths)
    """
    attempted: list = []
    if not path:
        return path, tuple(attempted)

    if os.path.isabs(path):
        attempted.append(path)
        return path, tuple(attempted)

    attempted.append(path)
    if os.path.isfile(path):
        return path, tuple(attempted)

    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory(package)
        candidate = os.path.join(share, path)
        attempted.append(candidate)
        if os.path.isfile(candidate):
            return candidate, tuple(attempted)
    except Exception as exc:  # package not installed / not indexed on this host
        attempted.append(f"<{package} share directory unavailable: {type(exc).__name__}>")

    return path, tuple(attempted)


def is_invalid_embedding(vector: np.ndarray, expected_dim: int = 512) -> Optional[str]:
    """Return a reason string when a produced embedding must not be scored, else None.

    The frozen extractor contract returns ``zeros(512)`` for failures; scoring that vector
    would look like a legitimate low-similarity mismatch. This distinguishes the two.
    """
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    if arr.shape[0] != expected_dim:
        return f"embedding dimension {arr.shape[0]} != {expected_dim}"
    if not np.all(np.isfinite(arr)):
        return "embedding contains non-finite values"
    if float(np.linalg.norm(arr)) < 1e-6:
        return "embedding is the invalid-crop sentinel (zero norm) - not scored"
    return None

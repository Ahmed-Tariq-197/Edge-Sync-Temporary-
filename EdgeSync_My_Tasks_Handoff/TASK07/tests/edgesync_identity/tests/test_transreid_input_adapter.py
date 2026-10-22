"""TR-03 / TR-04 / TR-05 regression tests — TransReID input and asset adapters.

These are unit tests over the pure adapter functions; they do not require a ROS 2 runtime
(``rclpy`` is absent on this host) and they do not perform model inference. Channel ordering is
checked against the offline preprocessing convention used by ``scripts/evaluate_offline.py``
(BGR image read by OpenCV, converted to RGB before the extractor).

Traces: TR-03 (SIE camera label plumbing), TR-04 (image encoding), TR-05 (portable asset paths).
"""
import os
import sys

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

import numpy as np
import pytest

from edgesync_identity.core.reid_input_adapter import (
    image_to_rgb, is_invalid_embedding, resolve_asset_path, resolve_camera_label)


class _Image:
    """Duck-typed stand-in for sensor_msgs/msg/Image."""

    def __init__(self, data, height, width, encoding, step=None):
        self.data = data
        self.height = height
        self.width = width
        self.encoding = encoding
        self.step = step if step is not None else width * 3


def _bgr_tensor(h=4, w=3):
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = 10  # B
    arr[:, :, 1] = 20  # G
    arr[:, :, 2] = 30  # R
    return arr


class TestImageEncoding:
    def test_bgr8_is_converted_to_rgb(self):
        arr = _bgr_tensor()
        msg = _Image(arr.tobytes(), arr.shape[0], arr.shape[1], 'bgr8')
        out = image_to_rgb(msg)
        assert out.shape == arr.shape and out.dtype == np.uint8
        assert out[0, 0].tolist() == [30, 20, 10], "bgr8 must be swapped to RGB"

    def test_rgb8_is_not_swapped_again(self):
        arr = _bgr_tensor()[:, :, ::-1].copy()  # already RGB
        msg = _Image(arr.tobytes(), arr.shape[0], arr.shape[1], 'rgb8')
        out = image_to_rgb(msg)
        assert out[0, 0].tolist() == [30, 20, 10]

    def test_uppercase_and_8uc3_encodings_accepted(self):
        arr = _bgr_tensor()
        assert image_to_rgb(_Image(arr.tobytes(), 4, 3, 'BGR8'))[0, 0].tolist() == [30, 20, 10]
        assert image_to_rgb(_Image(arr.tobytes(), 4, 3, '8UC3'))[0, 0].tolist() == [30, 20, 10]

    def test_big_endian_suffix_is_stripped(self):
        arr = _bgr_tensor()
        msg = _Image(arr.tobytes(), 4, 3, 'bgr8; bson_msgs::Image')
        assert image_to_rgb(msg)[0, 0].tolist() == [30, 20, 10]

    def test_row_stride_padded_buffer_is_trimmed(self):
        arr = _bgr_tensor(h=2, w=3)
        step = 3 * 3 + 5  # 5 bytes of per-row padding
        padded = np.zeros((2, step), dtype=np.uint8)
        padded[:, :9] = arr.reshape(2, 9)
        msg = _Image(padded.tobytes(), 2, 3, 'bgr8', step=step)
        out = image_to_rgb(msg)
        assert out.shape == (2, 3, 3)
        assert out[1, 2].tolist() == [30, 20, 10]

    @pytest.mark.parametrize('encoding', ['mono8', 'bgra8', '16UC1', ''])
    def test_unsupported_encoding_rejected_with_diagnostic(self, encoding):
        arr = _bgr_tensor()
        msg = _Image(arr.tobytes(), 4, 3, encoding, step=arr.shape[1] * 3)
        with pytest.raises(ValueError, match='unsupported image encoding'):
            image_to_rgb(msg)

    def test_buffer_length_mismatch_rejected(self):
        msg = _Image(np.zeros(4 * 3 * 2, dtype=np.uint8).tobytes(), 4, 3, 'bgr8')
        with pytest.raises(ValueError, match='buffer size mismatch'):
            image_to_rgb(msg)

    def test_zero_geometry_rejected(self):
        with pytest.raises(ValueError, match='invalid image geometry'):
            image_to_rgb(_Image(b'', 0, 0, 'bgr8'))

    def test_roundtrip_matches_offline_bgr_to_rgb(self):
        """Same pixels, same channel order as evaluate_offline.py's cv2.cvtColor path."""
        arr = (np.arange(4 * 3 * 3, dtype=np.uint8)).reshape(4, 3, 3)
        msg = _Image(arr.tobytes(), 4, 3, 'bgr8')
        assert np.array_equal(image_to_rgb(msg), arr[:, :, ::-1].copy())


class TestCameraLabel:
    @pytest.mark.parametrize('cam_id,expected', [(1, 0), (2, 1), (6, 5)])
    def test_valid_ids_map_to_zero_based_labels(self, cam_id, expected):
        label, reason = resolve_camera_label(cam_id)
        assert label == expected
        assert 'SIE applied' in reason

    @pytest.mark.parametrize('cam_id', [0, -1, -6])
    def test_unset_stays_camera_agnostic(self, cam_id):
        label, reason = resolve_camera_label(cam_id)
        assert label is None and 'camera-agnostic' in reason

    @pytest.mark.parametrize('cam_id', [7, 100])
    def test_out_of_range_does_not_invent_a_label(self, cam_id):
        label, reason = resolve_camera_label(cam_id)
        assert label is None and 'outside 1..6' in reason

    def test_non_integer_is_rejected_not_guessed(self):
        label, reason = resolve_camera_label('c1')
        assert label is None and 'not an integer' in reason

    def test_none_is_rejected_not_guessed(self):
        label, _ = resolve_camera_label(None)
        assert label is None


class TestAssetPathResolution:
    def test_empty_path_passes_through(self):
        assert resolve_asset_path('') == ('', ())

    def test_existing_absolute_path_returned_unchanged(self, tmp_path):
        f = tmp_path / 'model.pth'
        f.write_bytes(b'x')
        resolved, attempts = resolve_asset_path(str(f))
        assert resolved == str(f)
        assert attempts == (str(f),)

    def test_relative_path_resolving_from_cwd_is_kept(self, tmp_path, monkeypatch):
        (tmp_path / 'models').mkdir()
        (tmp_path / 'models' / 'a.pth').write_bytes(b'x')
        monkeypatch.chdir(tmp_path)
        resolved, attempts = resolve_asset_path(os.path.join('models', 'a.pth'))
        assert resolved == os.path.join('models', 'a.pth')
        assert len(attempts) == 1

    def test_missing_relative_path_reports_every_attempt(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        resolved, attempts = resolve_asset_path('models/transreid/nope.pth')
        assert resolved == 'models/transreid/nope.pth', "caller must still fail fast"
        assert len(attempts) >= 2, "attempt list must show the share-directory try"
        assert any('share directory unavailable' in a or 'nope.pth' in a for a in attempts)

    def test_share_directory_used_when_package_is_indexed(self, tmp_path, monkeypatch):
        share = tmp_path / 'install' / 'edgesync_bringup'
        (share / 'models').mkdir(parents=True)
        (share / 'models' / 'inst.pth').write_bytes(b'x')
        cwd = tmp_path / 'elsewhere'
        cwd.mkdir()
        monkeypatch.chdir(cwd)  # relative path must NOT resolve from the CWD here

        import sys as _sys
        import types
        mod = types.ModuleType('ament_index_python')
        pkgs = types.ModuleType('ament_index_python.packages')
        pkgs.get_package_share_directory = lambda name: str(share)
        mod.packages = pkgs
        _sys.modules['ament_index_python'] = mod
        _sys.modules['ament_index_python.packages'] = pkgs
        try:
            resolved, attempts = resolve_asset_path(os.path.join('models', 'inst.pth'))
            assert resolved == os.path.join(str(share), 'models', 'inst.pth')
            assert len(attempts) == 2
        finally:
            for key in ('ament_index_python', 'ament_index_python.packages'):
                _sys.modules.pop(key, None)


class TestInvalidEmbeddingGuard:
    def test_zero_sentinel_is_flagged(self):
        reason = is_invalid_embedding(np.zeros(512, dtype=np.float32))
        assert reason and 'sentinel' in reason

    def test_non_finite_is_flagged(self):
        v = np.ones(512, dtype=np.float32) / np.sqrt(512)
        v[3] = np.nan
        assert 'non-finite' in is_invalid_embedding(v)

    def test_wrong_dimension_is_flagged(self):
        assert 'dimension' in is_invalid_embedding(np.ones(768, dtype=np.float32))

    def test_unit_norm_embedding_passes(self):
        rng = np.random.default_rng(7)
        v = rng.standard_normal(512).astype(np.float32)
        v /= np.linalg.norm(v)
        assert is_invalid_embedding(v) is None

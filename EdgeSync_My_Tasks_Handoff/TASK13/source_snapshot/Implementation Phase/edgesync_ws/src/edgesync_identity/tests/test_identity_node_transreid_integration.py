"""TR-02/TR-03/TR-04/TR-05 regression tests for IdentityNode candidate handling.

Test class: UNIT. ROS 2 middleware (rclpy, edgesync_msgs, sensor_msgs) is not installed
on this host, so the stubs installed below replace ONLY the middleware scaffolding. These
tests exercise IdentityNode's own decision logic (encoding decode, camera-label routing,
asset-path resolution, refusal of invalid embeddings). They are NOT ROS 2 integration
tests: no DDS, no live service client, no real executor spin. Integration status is
recorded separately as BLOCKED — Pending ROS 2 runtime.

Fake extractors are used deliberately: a real extractor's inference accuracy is verified in
the offline/evaluation tests, not here. Everything asserted below is about what the node
does with whatever vector the extractor returns.
"""
import sys
import os
import types
import logging
import pytest
import numpy as np

# ── Stub ROS scaffolding (must be installed before importing identity_node) ──


class _StubParameterValue:
    def __init__(self, value):
        self._value = value

    @property
    def string_value(self):
        return str(self._value)

    @property
    def double_value(self):
        return float(self._value)

    @property
    def bool_value(self):
        return bool(self._value)

    @property
    def integer_value(self):
        return int(self._value)


class _StubParameter:
    def __init__(self, name, value):
        self.name = name
        self._value = value

    def get_parameter_value(self):
        return _StubParameterValue(self._value)


class _StubNode:
    """Minimal rclpy.node.Node stand-in. Overrides come from StubNodeOverrides."""

    param_overrides = {}

    def __init__(self, name):
        self._name = name
        self._params = {}

    def declare_parameter(self, name, default, *args, **kwargs):
        self._params[name] = _StubParameter(name, self.param_overrides.get(name, default))

    def get_parameter(self, name):
        return self._params[name]

    def create_service(self, srv_type, name, callback, *args, **kwargs):
        return ('service', name, callback)

    def create_subscription(self, msg_type, topic, callback, qos, *args, **kwargs):
        return ('subscription', topic, callback)

    def get_logger(self):
        logger = logging.getLogger(f'stub.{self._name}')
        return logger


def _build_stub_package(name, module_names=()):
    pkg = types.ModuleType(name)
    pkg.__path__ = []
    sys.modules[name] = pkg
    return pkg


class _StubImage:
    """Duck-typed sensor_msgs/Image with the fields image_to_rgb() consumes."""

    def __init__(self, data, height, width, encoding, step=None):
        self.data = bytes(bytearray(data))
        self.height = height
        self.width = width
        self.encoding = encoding
        self.step = step if step is not None else width * 3


class _StubRequest:
    def __init__(self, transaction_id='', reference=None, candidate=None, crop=None):
        self.transaction_id = transaction_id
        self.reference_feature_vector = (np.zeros(512, dtype=np.float32)
                                        if reference is None else list(np.asarray(reference).reshape(-1)))
        self.candidate_feature_vector = ([] if candidate is None
                                        else list(np.asarray(candidate).reshape(-1)))
        self.candidate_crop = _StubImage(b'', 0, 0, 'bgr8') if crop is None else crop


class _StubResponse:
    def __init__(self):
        self.match = None
        self.similarity = None
        self.message = ''
        self.match_confirmed = None
        self.similarity_score = None


class _IdentityVerify:
    Request = _StubRequest
    Response = _StubResponse


class _StubHandoffMsg:
    def __init__(self):
        self.transaction_id = ''


def _install_ros_stubs():
    rclpy = _build_stub_package('rclpy')
    rclpy.node = types.ModuleType('rclpy.node')
    rclpy.node.Node = _StubNode
    rclpy.Node = _StubNode

    def _no_middleware(*args, **kwargs):
        raise RuntimeError('rclpy stub only — no ROS 2 middleware on this host')

    rclpy.init = _no_middleware
    rclpy.shutdown = _no_middleware
    rclpy.ok = lambda: False
    rclpy.spin = _no_middleware
    sys.modules['rclpy.node'] = rclpy.node

    msgs = _build_stub_package('edgesync_msgs')
    srv = types.ModuleType('edgesync_msgs.srv')
    srv.IdentityVerify = _IdentityVerify
    msgs.srv = srv
    sys.modules['edgesync_msgs.srv'] = srv
    msg = types.ModuleType('edgesync_msgs.msg')
    msg.HandoffConfirm = _StubHandoffMsg
    msg.HandoffCancel = _StubHandoffMsg
    msgs.msg = msg
    sys.modules['edgesync_msgs.msg'] = msg


_STUB_KEYS = ('rclpy', 'rclpy.node', 'edgesync_msgs', 'edgesync_msgs.srv', 'edgesync_msgs.msg')


def _import_identity_node_under_stubs():
    """Import identity_node with the ROS stubs installed, then uninstall the stubs again.

    Leaving the stubs in sys.modules would let another suite's ``import rclpy`` succeed against
    a fake middleware — including this file's deliberate ``rclpy.init()`` guard — so a real ROS
    test would report a confusing stub RuntimeError (or, worse, pass against fakes) instead of
    its own genuine ModuleNotFoundError on a host without ROS 2.
    """
    saved = {key: sys.modules.get(key) for key in _STUB_KEYS}
    _install_ros_stubs()
    try:
        from edgesync_identity.nodes.identity_node import IdentityNode
        return IdentityNode
    finally:
        for key, module in saved.items():
            if module is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = module


IdentityNode = _import_identity_node_under_stubs()


def _unit_vector(idx, dim=512):
    v = np.zeros(dim, dtype=np.float32)
    v[idx] = 1.0
    return v


class _RecordingExtractor:
    """Returns a scripted embedding and records every call it receives."""
    embedding_dim = 512

    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        self.calls = []

    def extract_embedding(self, image_crop):
        self.calls.append({'image_crop': image_crop})
        return self.vector.copy()


class _CamAwareExtractor(_RecordingExtractor):
    def extract_embedding(self, image_crop, cam_label=None):
        self.calls.append({'image_crop': image_crop, 'cam_label': cam_label})
        return self.vector.copy()


def make_node(overrides=None, extractor=None):
    _StubNode.param_overrides = dict(overrides or {})
    try:
        return IdentityNode(extractor=extractor or _RecordingExtractor(_unit_vector(0)))
    finally:
        _StubNode.param_overrides = {}


def run_verify(node, request):
    return node._handle_identity_verify(request, _IdentityVerify.Response())


class TestValidPathStillScores:
    def setup_method(self):
        self.ref = _unit_vector(0)
        self.node = make_node(extractor=_RecordingExtractor(self.ref))

    def test_matching_crop_is_confirmed(self):
        crop = _StubImage(self._rgb_bytes(), 1, 2, 'rgb8')
        resp = run_verify(self.node, _StubRequest('txn-1', reference=self.ref, crop=crop))
        assert resp.match is True
        assert pytest.approx(resp.similarity, abs=1e-6) == 1.0
        assert not resp.message.startswith(('INVALID_INPUT', 'INVALID_EMBEDDING'))
        assert self.node._invalid_frame_count == 0

    @staticmethod
    def _rgb_bytes():
        return bytes([255, 0, 0, 0, 255, 0])

    def test_bgr8_crop_is_converted_to_rgb_before_extraction(self):
        # bgr8 buffer (B,G,R): first pixel blue -> RGB (0,0,255)
        crop = _StubImage(bytes([255, 0, 0, 0, 255, 0]), 1, 2, 'bgr8')
        run_verify(self.node, _StubRequest('txn-bgr', reference=self.ref, crop=crop))
        seen = self.node.extractor.calls[0]['image_crop']
        assert seen.shape == (1, 2, 3)
        assert list(seen[0, 0]) == [0, 0, 255]

    def test_rgb8_crop_is_passed_through_unreversed(self):
        crop = _StubImage(bytes([255, 0, 0, 0, 255, 0]), 1, 2, 'rgb8')
        run_verify(self.node, _StubRequest('txn-rgb', reference=self.ref, crop=crop))
        seen = self.node.extractor.calls[0]['image_crop']
        assert list(seen[0, 0]) == [255, 0, 0]

    def test_request_vector_takes_precedence_over_crop(self):
        crop = _StubImage(bytes(6), 1, 2, 'rgb8')
        resp = run_verify(self.node, _StubRequest('txn-vec', reference=self.ref,
                                                 candidate=self.ref, crop=crop))
        assert resp.match is True
        assert self.node.extractor.calls == []


class TestTR04InvalidEncodingRefused:
    def test_unsupported_encoding_refuses_without_running_inference(self):
        extractor = _RecordingExtractor(_unit_vector(0))
        node = make_node(extractor=extractor)
        # 1x2 bgra8: step 8 bytes/row, so the buffer itself is consistent and the
        # unsupported-encoding branch is what must reject it.
        crop = _StubImage(bytes(8), 1, 2, 'bgra8', step=8)
        resp = run_verify(node, _StubRequest('txn-enc', reference=_unit_vector(0), crop=crop))
        assert resp.match is False
        assert resp.similarity == 0.0
        assert resp.message.startswith('INVALID_INPUT')
        assert 'unsupported image encoding' in resp.message
        assert extractor.calls == []
        assert node._invalid_frame_count == 1

    def test_geometry_mismatch_refused(self):
        node = make_node()
        crop = _StubImage(bytes(9), 2, 2, 'rgb8')  # 2x2x3 needs 12 bytes
        resp = run_verify(node, _StubRequest('txn-geo', reference=_unit_vector(0), crop=crop))
        assert resp.message.startswith('INVALID_INPUT')
        assert 'source=none' in resp.message

    def test_no_candidate_at_all_is_not_scored_as_mismatch(self):
        node = make_node()
        resp = run_verify(node, _StubRequest('txn-empty', reference=_unit_vector(0)))
        assert resp.match is False
        assert 'no candidate image or vector supplied' in resp.message
        assert node._invalid_frame_count == 1


class TestTR02ZeroEmbeddingRefused:
    def test_zero_embedding_from_extractor_is_not_scored_as_low_similarity(self):
        node = make_node(extractor=_RecordingExtractor(np.zeros(512, dtype=np.float32)))
        crop = _StubImage(bytes(6), 1, 2, 'rgb8')
        resp = run_verify(node, _StubRequest('txn-zero', reference=_unit_vector(0), crop=crop))
        assert resp.message.startswith('INVALID_EMBEDDING')
        assert 'candidate' in resp.message
        assert 'below threshold' not in resp.message
        assert resp.similarity == 0.0

    def test_non_finite_embedding_refused(self):
        bad = _unit_vector(0)
        bad[7] = np.nan
        node = make_node(extractor=_RecordingExtractor(bad))
        crop = _StubImage(bytes(6), 1, 2, 'rgb8')
        resp = run_verify(node, _StubRequest('txn-nan', reference=_unit_vector(0), crop=crop))
        assert resp.message.startswith('INVALID_EMBEDDING')

    def test_wrong_dimension_embedding_refused(self):
        node = make_node(extractor=_RecordingExtractor(_unit_vector(0, dim=768)))
        crop = _StubImage(bytes(6), 1, 2, 'rgb8')
        resp = run_verify(node, _StubRequest('txn-dim', reference=_unit_vector(0), crop=crop))
        assert resp.message.startswith('INVALID_EMBEDDING')

    def test_missing_reference_refused_rather_than_falsely_rejected(self):
        node = make_node(extractor=_RecordingExtractor(_unit_vector(0)))
        crop = _StubImage(bytes(6), 1, 2, 'rgb8')
        resp = run_verify(node, _StubRequest('txn-noref', reference=None, crop=crop))
        assert resp.message.startswith('INVALID_EMBEDDING')
        assert 'reference' in resp.message
        assert 'ref_source=none' in resp.message

    def test_invalid_counter_accumulates(self):
        node = make_node(extractor=_RecordingExtractor(np.zeros(512, dtype=np.float32)))
        crop = _StubImage(bytes(6), 1, 2, 'rgb8')
        for i in range(3):
            run_verify(node, _StubRequest(f'txn-{i}', reference=_unit_vector(0), crop=crop))
        assert node._invalid_frame_count == 3

    def test_cached_reference_satisfies_request_with_empty_reference(self):
        ref = _unit_vector(3)
        node = make_node(extractor=_RecordingExtractor(ref))
        node.store_candidate_vector(ref, 'txn-cache')
        crop = _StubImage(bytes(6), 1, 2, 'rgb8')
        resp = run_verify(node, _StubRequest('txn-cache', reference=None, crop=crop))
        # A successful match is itself the proof the cache fed the verifier: with no reference
        # available TR-02 refuses to score, so match could not have become True.
        assert resp.match is True
        assert 'INVALID' not in resp.message


class TestTR03CameraLabelRouting:
    def test_camera_id_is_converted_to_zero_based_label(self):
        extractor = _CamAwareExtractor(_unit_vector(0))
        node = make_node(overrides={'reid_camera_id': 3}, extractor=extractor)
        crop = _StubImage(bytes(6), 1, 2, 'rgb8')
        run_verify(node, _StubRequest('txn-cam', reference=_unit_vector(0), crop=crop))
        assert extractor.calls[0]['cam_label'] == 2

    def test_unset_camera_id_stays_camera_agnostic(self):
        extractor = _CamAwareExtractor(_unit_vector(0))
        node = make_node(extractor=extractor)
        crop = _StubImage(bytes(6), 1, 2, 'rgb8')
        run_verify(node, _StubRequest('txn-nocam', reference=_unit_vector(0), crop=crop))
        assert extractor.calls[0]['cam_label'] is None

    def test_out_of_range_camera_id_falls_back_to_camera_agnostic(self):
        extractor = _CamAwareExtractor(_unit_vector(0))
        node = make_node(overrides={'reid_camera_id': 9}, extractor=extractor)
        assert node._cam_label is None
        crop = _StubImage(bytes(6), 1, 2, 'rgb8')
        resp = run_verify(node, _StubRequest('txn-oob', reference=_unit_vector(0), crop=crop))
        assert extractor.calls[0]['cam_label'] is None
        assert resp.match is True

    def test_extractor_without_cam_label_kwarg_is_called_with_image_only(self):
        extractor = _RecordingExtractor(_unit_vector(0))
        node = make_node(overrides={'reid_camera_id': 1}, extractor=extractor)
        assert node._extractor_accepts_cam_label is False
        crop = _StubImage(bytes(6), 1, 2, 'rgb8')
        resp = run_verify(node, _StubRequest('txn-nosie', reference=_unit_vector(0), crop=crop))
        assert list(extractor.calls[0].keys()) == ['image_crop']
        assert resp.match is True


class TestTR05AssetPathResolution:
    @staticmethod
    def _fake_share_dir(share_root):
        """Install a stubbed ament_index_python pointing at share_root."""
        ament = types.ModuleType('ament_index_python')
        ament.__path__ = []
        packages = types.ModuleType('ament_index_python.packages')
        packages.get_package_share_directory = lambda name: str(share_root)
        ament.packages = packages
        keys = ('ament_index_python', 'ament_index_python.packages')
        saved = {k: sys.modules.get(k) for k in keys}
        sys.modules['ament_index_python'] = ament
        sys.modules['ament_index_python.packages'] = packages
        return saved, keys

    def test_relative_asset_path_is_resolved_against_package_share_dir(self, tmp_path, monkeypatch):
        share = tmp_path / 'install' / 'edgesync_bringup'
        models = share / 'models' / 'transreid'
        models.mkdir(parents=True)
        ckpt = models / 'transformer_120.pth'
        ckpt.write_bytes(b'x')
        cwd = tmp_path / 'elsewhere'
        cwd.mkdir()
        monkeypatch.chdir(cwd)

        saved, keys = self._fake_share_dir(share)
        try:
            node = make_node(overrides={
                'reid_checkpoint_path': 'models/transreid/transformer_120.pth',
                'reid_projection_path': 'models/transreid/missing.npz'})
            assert os.path.normcase(node._asset_paths['checkpoint']) == \
                os.path.normcase(os.path.normpath(str(ckpt)))
            assert any(os.path.normcase(os.path.normpath(a)) ==
                       os.path.normcase(os.path.normpath(str(ckpt)))
                       for a in node._asset_attempts['checkpoint'])
            # An unresolvable projection is passed through untouched so the factory still fails fast.
            assert node._asset_paths['projection'] == 'models/transreid/missing.npz'
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    def test_absolute_asset_path_is_returned_unchanged(self, tmp_path):
        from edgesync_identity.core.reid_input_adapter import resolve_asset_path
        target = tmp_path / 'engine.trt'
        target.write_bytes(b'x')
        resolved, attempts = resolve_asset_path(str(target))
        assert resolved == str(target)
        assert attempts == (str(target),)

    def test_empty_asset_path_needs_no_resolution(self):
        from edgesync_identity.core.reid_input_adapter import resolve_asset_path
        resolved, attempts = resolve_asset_path('')
        assert resolved == ''
        assert attempts == ()

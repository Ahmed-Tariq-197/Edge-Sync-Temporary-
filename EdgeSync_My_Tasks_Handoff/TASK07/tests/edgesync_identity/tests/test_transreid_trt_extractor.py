"""TR-10 — focused regression tests for models/transreid_trt_extractor.py.

UNIT tests with a STUBBED TensorRT/pycuda layer. There is no TensorRT, no engine and no Jetson on
this host, so nothing here is real inference or integration testing (charter §9): the engine build
(TR-11) and every on-device number stay BLOCKED. What these tests do prove is the parts that are
host-independent and that would otherwise only be discovered on the target:

  * no silent mock — a missing engine or missing tensorrt fails loudly instead of returning a
    plausible pseudo-embedding (the behaviour inspected in osnet_trt_extractor.py, not copied);
  * the frozen wire contract: float32[512], unit norm, or zeros + an attributed failure stage;
  * the preprocessing contract matches the deployed TransReID normalisation, not ImageNet;
  * the engine-shape guards reject a 512-output engine (which would mean a double projection);
  * the TRT 8.x / TRT 10.x binding-API branches and the binding-address ordering.
"""
import os
import sys

import numpy as np
import pytest

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from edgesync_identity.models.transreid_trt_extractor import (  # noqa: E402
    CAM_BINDING, FEATURE_BINDING, IMAGE_BINDING, TransReIdTrtExtractor)

FEATURE_DIM = 768


def _projection(tmp_path, mean=None, projection=None, name='proj.npz'):
    mean = np.zeros(FEATURE_DIM, dtype=np.float32) if mean is None else mean
    projection = (np.eye(FEATURE_DIM, 512, dtype=np.float32) if projection is None
                  else projection)
    path = os.path.join(str(tmp_path), name)
    np.savez(path, projection=projection, mean=mean)
    return path


def _bare(tmp_path, projection_path, input_shape=(256, 128)):
    """An instance without running __init__ — the load path needs a GPU, the methods do not.

    A real (dummy) engine file is created because the metadata reader stats it, exactly as the
    production path would after the fail-fast check in __init__.
    """
    engine = os.path.join(str(tmp_path), 'dummy.engine')
    if not os.path.exists(engine):
        with open(engine, 'wb') as handle:
            handle.write(b'\x00' * 32)
    obj = TransReIdTrtExtractor.__new__(TransReIdTrtExtractor)
    obj.engine_path = engine
    obj.projection_path = projection_path
    obj.input_height, obj.input_width = input_shape
    obj.last_failure = None
    obj.failure_counts = {}
    obj.embedding_count = 0
    obj.engine_metadata = None
    obj.use_unified_memory = True
    data = np.load(projection_path)
    obj._svd_projection = data['projection'].astype(np.float32)
    obj._svd_mean = data['mean'].astype(np.float32)
    return obj


def _crop(h=64, w=32, value=100):
    return np.full((h, w, 3), value, dtype=np.uint8)


class TestFailFastNoMock:
    def test_missing_engine_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match='TensorRT engine not found'):
            TransReIdTrtExtractor(str(tmp_path / 'absent.engine'), _projection(tmp_path))

    def test_missing_projection_raises_file_not_found(self, tmp_path):
        engine = tmp_path / 'e.engine'
        engine.write_bytes(b'\x00')
        with pytest.raises(FileNotFoundError, match='SVD projection not found'):
            TransReIdTrtExtractor(str(engine), str(tmp_path / 'absent.npz'))

    def test_empty_engine_path_is_not_treated_as_valid(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            TransReIdTrtExtractor('', _projection(tmp_path))

    def test_host_without_tensorrt_raises_instead_of_mocking(self, tmp_path):
        """The OsnetTrtExtractor fallback would return RGB-mean pseudo-embeddings here.

        Charter §5 forbids presenting a synthetic vector as inference output, so this arm must
        raise. This test documents the divergence and proves it holds on this host.
        """
        pytest.importorskip('numpy')
        if 'tensorrt' in sys.modules:
            pytest.skip('a real tensorrt is importable on this host; the no-mock branch is not '
                        'the one being exercised')
        engine = tmp_path / 'e.engine'
        engine.write_bytes(b'\x00' * 16)
        with pytest.raises(RuntimeError, match='tensorrt is not importable'):
            TransReIdTrtExtractor(str(engine), _projection(tmp_path))

    def test_projection_shape_is_validated_at_load(self, tmp_path):
        obj = _bare(tmp_path, _projection(tmp_path))
        obj.projection_path = _projection(tmp_path, projection=np.ones((768, 256), np.float32),
                                          name='bad.npz')
        with pytest.raises(ValueError, match=r'SVD projection shape mismatch'):
            obj._load_svd_projection()

    def test_mean_shape_is_validated_at_load(self, tmp_path):
        obj = _bare(tmp_path, _projection(tmp_path))
        obj.projection_path = _projection(tmp_path, mean=np.zeros(512, np.float32), name='m.npz')
        with pytest.raises(ValueError, match='SVD mean shape mismatch'):
            obj._load_svd_projection()


class TestPreprocessContract:
    def test_matches_the_documented_normalisation_arithmetic(self, tmp_path):
        obj = _bare(tmp_path, _projection(tmp_path))
        crop = np.zeros((40, 20, 3), dtype=np.uint8)
        crop[..., 0], crop[..., 1], crop[..., 2] = 0, 128, 255
        out = obj.preprocess(crop)
        assert out.shape == (1, 3, 256, 128)
        assert out.dtype == np.float32
        assert out.flags['C_CONTIGUOUS']
        # each channel is constant, so resize keeps it constant: (v/255 - .5)/.5 = v/127.5 - 1
        assert np.allclose(out[0, 0], -1.0, atol=1e-6)
        assert np.allclose(out[0, 1], 128 / 127.5 - 1, atol=1e-6)
        assert np.allclose(out[0, 2], 1.0, atol=1e-6)

    def test_is_the_transreid_stats_not_imagenet(self, tmp_path):
        """The OSNet arm normalises with ImageNet stats; reusing that here would shift features."""
        obj = _bare(tmp_path, _projection(tmp_path))
        grey = np.full((32, 16, 3), 128, dtype=np.uint8)
        value = float(obj.preprocess(grey)[0, 0, 0, 0])
        imagenet_value = (128 / 255.0 - 0.485) / 0.229
        assert value == pytest.approx(128 / 127.5 - 1, abs=1e-6)
        assert abs(value - imagenet_value) > 0.05

    def test_output_range_is_minus_one_to_one(self, tmp_path):
        obj = _bare(tmp_path, _projection(tmp_path))
        rng = np.random.default_rng(0)
        crop = rng.integers(0, 256, size=(64, 32, 3), dtype=np.uint8)
        out = obj.preprocess(crop)
        assert out.min() >= -1.0 - 1e-6 and out.max() <= 1.0 + 1e-6


class TestPreprocessParityWithTorchArm:
    """ES07_HEAVY=1 only: loads the real 414 MB checkpoint to prove both arms agree on pixels."""

    def test_trt_preprocess_equals_the_deployed_torch_extractor(self, tmp_path):
        if not os.environ.get('ES07_HEAVY'):
            pytest.skip('set ES07_HEAVY=1 to load transformer_120.pth for the parity check')
        bringup = os.path.abspath(os.path.join(pkg_dir, '..', 'edgesync_bringup', 'models',
                                               'transreid'))
        checkpoint = os.path.join(bringup, 'transformer_120.pth')
        projection = os.path.join(bringup, 'svd_projection_768_to_512_real_v2.npz')
        if not (os.path.isfile(checkpoint) and os.path.isfile(projection)):
            pytest.skip(f'TransReID assets not present under {bringup}')
        from edgesync_identity.models.transreid_extractor import TransReIdExtractor

        torch_arm = TransReIdExtractor(checkpoint_path=checkpoint, projection_path=projection)
        trt_arm = _bare(tmp_path, projection)
        crop = np.random.default_rng(1).integers(0, 256, size=(200, 90, 3), dtype=np.uint8)
        np.testing.assert_array_equal(trt_arm.preprocess(crop),
                                      torch_arm.preprocess(crop).cpu().numpy())


class TestWireContractSentinels:
    def test_invalid_inputs_return_zeros_and_record_the_stage(self, tmp_path):
        obj = _bare(tmp_path, _projection(tmp_path))
        cases = [(None, 'input_empty'),
                 (np.zeros((0, 0, 3), np.uint8), 'input_empty'),
                 (np.zeros((32, 16), np.uint8), 'input_shape'),
                 (np.zeros((32, 16, 1), np.uint8), 'input_shape'),
                 (np.zeros((8, 40, 3), np.uint8), 'input_too_small'),
                 (np.zeros((40, 8, 3), np.uint8), 'input_too_small')]
        for crop, stage in cases:
            out = obj.extract_embedding(crop)
            assert out.shape == (512,) and out.dtype == np.float32, stage
            assert np.all(out == 0.0), stage
            assert obj.last_failure['stage'] == stage, stage
        assert obj.embedding_count == 0
        assert sum(obj.failure_counts.values()) == len(cases)

    def test_cam_label_out_of_range_is_rejected_before_inference(self, tmp_path):
        obj = _bare(tmp_path, _projection(tmp_path))
        obj.has_cam_input = True
        calls = []
        obj._execute = lambda crop, cam: calls.append(cam) or np.ones(FEATURE_DIM, np.float32)
        for bad in (-1, 6, 99):
            assert np.all(obj.extract_embedding(_crop(), cam_label=bad) == 0.0)
            assert obj.last_failure['stage'] == 'cam_label_out_of_range'
        assert calls == []

    def test_a_raising_engine_becomes_a_sentinel_not_an_exception(self, tmp_path):
        obj = _bare(tmp_path, _projection(tmp_path))
        obj.has_cam_input = False

        def boom(crop, cam):
            raise RuntimeError('engine exploded')

        obj._execute = boom
        out = obj.extract_embedding(_crop())
        assert np.all(out == 0.0)
        assert obj.last_failure['stage'] == 'inference_exception'
        assert 'engine exploded' in obj.last_failure['reason']


class TestProjectionPath:
    """The post-engine contract: centre -> SVD 768->512 -> L2, with failure attribution."""

    def _stub(self, tmp_path, feature, mean=None, projection=None):
        obj = _bare(tmp_path, _projection(tmp_path, mean=mean, projection=projection))
        obj.has_cam_input = False
        obj._execute = lambda crop, cam: np.asarray(feature, dtype=np.float32)
        return obj

    def test_matches_the_documented_contract_exactly(self, tmp_path):
        mean = np.full(FEATURE_DIM, 0.5, np.float32)
        projection = np.eye(FEATURE_DIM, 512, dtype=np.float32)
        feature = np.arange(FEATURE_DIM, dtype=np.float32) / 100.0
        obj = self._stub(tmp_path, feature, mean=mean, projection=projection)
        out = obj.extract_embedding(_crop())
        expected = (feature - mean) @ projection
        expected = expected / np.linalg.norm(expected)
        assert np.allclose(out, expected, atol=1e-6)
        assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-4)
        assert obj.embedding_count == 1 and obj.last_failure is None

    def test_wrong_feature_width_is_a_shape_mismatch_not_a_truncation(self, tmp_path):
        obj = self._stub(tmp_path, np.ones(512, np.float32))
        assert np.all(obj.extract_embedding(_crop()) == 0.0)
        assert obj.last_failure['stage'] == 'engine_shape_mismatch'
        assert '512' in obj.last_failure['reason']

    def test_nan_from_the_engine_is_rejected(self, tmp_path):
        feature = np.ones(FEATURE_DIM, np.float32)
        feature[7] = np.nan
        obj = self._stub(tmp_path, feature)
        assert np.all(obj.extract_embedding(_crop()) == 0.0)
        assert obj.last_failure['stage'] == 'backbone_non_finite'

    def test_infinite_from_the_engine_is_rejected(self, tmp_path):
        feature = np.ones(FEATURE_DIM, np.float32)
        feature[3] = -np.inf
        obj = self._stub(tmp_path, feature)
        assert np.all(obj.extract_embedding(_crop()) == 0.0)
        assert obj.last_failure['stage'] == 'backbone_non_finite'

    def test_projection_of_the_mean_is_a_degenerate_sentinel(self, tmp_path):
        mean = np.full(FEATURE_DIM, 2.0, np.float32)
        obj = self._stub(tmp_path, mean.copy(), mean=mean)
        assert np.all(obj.extract_embedding(_crop()) == 0.0)
        assert obj.last_failure['stage'] == 'projection_degenerate'

    def test_non_finite_after_projection_is_attributed_separately(self, tmp_path):
        obj = self._stub(tmp_path, np.ones(FEATURE_DIM, np.float32))
        obj._svd_projection = np.full((FEATURE_DIM, 512), np.inf, np.float32)
        assert np.all(obj.extract_embedding(_crop()) == 0.0)
        assert obj.last_failure['stage'] == 'projection_non_finite'

    def test_success_clears_the_previous_failure_record(self, tmp_path):
        obj = self._stub(tmp_path, np.ones(FEATURE_DIM, np.float32))
        obj.extract_embedding(None)
        assert obj.last_failure is not None
        obj._execute = lambda crop, cam: np.arange(FEATURE_DIM, dtype=np.float32)
        out = obj.extract_embedding(_crop())
        assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-4)
        assert obj.last_failure is None


def _bindings(inputs, outputs):
    return {'inputs': inputs, 'outputs': outputs}


class TestEngineContractGuards:
    def _stub(self, tmp_path, inputs, outputs):
        obj = _bare(tmp_path, _projection(tmp_path))
        obj._bindings = _bindings(inputs, outputs)
        return obj

    def test_the_valid_agnostic_engine_passes(self, tmp_path):
        obj = self._stub(tmp_path,
                         {IMAGE_BINDING: {'dtype': 'float32', 'shape': [1, 3, 256, 128]}},
                         {FEATURE_BINDING: {'dtype': 'float32', 'shape': [1, 768]}})
        obj._validate_engine_contract()

    def test_the_valid_sie_engine_passes(self, tmp_path):
        obj = self._stub(tmp_path,
                         {IMAGE_BINDING: {'dtype': 'float32', 'shape': [1, 3, 256, 128]},
                          CAM_BINDING: {'dtype': 'int64', 'shape': [1]}},
                         {FEATURE_BINDING: {'dtype': 'float32', 'shape': [1, 768]}})
        obj._validate_engine_contract()

    def test_a_512_output_engine_is_refused_as_a_double_projection(self, tmp_path):
        obj = self._stub(tmp_path,
                         {IMAGE_BINDING: {'dtype': 'float32', 'shape': [1, 3, 256, 128]}},
                         {'embedding_512': {'dtype': 'float32', 'shape': [1, 512]}})
        with pytest.raises(RuntimeError, match=r'expected exactly \[feature_768\]'):
            obj._validate_engine_contract()

    def test_a_wide_output_that_is_not_768_is_refused(self, tmp_path):
        obj = self._stub(tmp_path,
                         {IMAGE_BINDING: {'dtype': 'float32', 'shape': [1, 3, 256, 128]}},
                         {FEATURE_BINDING: {'dtype': 'float32', 'shape': [1, 384]}})
        with pytest.raises(RuntimeError, match='double|applied twice|!= 768'):
            obj._validate_engine_contract()

    def test_wrong_resolution_engine_is_refused(self, tmp_path):
        obj = self._stub(tmp_path,
                         {IMAGE_BINDING: {'dtype': 'float32', 'shape': [1, 3, 128, 64]}},
                         {FEATURE_BINDING: {'dtype': 'float32', 'shape': [1, 768]}})
        with pytest.raises(RuntimeError, match='disagrees with the declared contract'):
            obj._validate_engine_contract()

    def test_missing_image_binding_is_refused(self, tmp_path):
        obj = self._stub(tmp_path, {'pixels': {'dtype': 'float32', 'shape': [1, 3, 256, 128]}},
                         {FEATURE_BINDING: {'dtype': 'float32', 'shape': [1, 768]}})
        with pytest.raises(RuntimeError, match='image binding'):
            obj._validate_engine_contract()

    def test_an_extra_unexpected_input_is_refused(self, tmp_path):
        obj = self._stub(tmp_path,
                         {IMAGE_BINDING: {'dtype': 'float32', 'shape': [1, 3, 256, 128]},
                          'cam_weights': {'dtype': 'float32', 'shape': [6, 768]}},
                         {FEATURE_BINDING: {'dtype': 'float32', 'shape': [1, 768]}})
        with pytest.raises(RuntimeError, match='unexpected inputs'):
            obj._validate_engine_contract()


class _FakeAlloc:
    def __init__(self, address):
        self.address = address
        self.copied_in = []
        self.copied_out = []

    def __int__(self):
        return self.address

    def copy_from_host(self, buf):
        self.copied_in.append(np.asarray(buf).copy())

    def copy_to_host(self, buf):
        self.copied_out.append(np.asarray(buf).copy())


class _FakeCuda:
    def __init__(self):
        self.counter = 1000

    def mem_alloc(self, nbytes):
        self.counter += 1
        return _FakeAlloc(self.counter)

    def Stream(self):
        return _FakeStream()


class _FakeStream:
    handle = 77

    def synchronize(self):
        pass


class _FakeContext8:
    """TRT 8.x: only execute_v2, addresses in binding order."""

    def __init__(self, ok=True):
        self.calls = []
        self.ok = ok

    def execute_v2(self, addresses):
        self.calls.append(list(addresses))
        return self.ok


class _FakeContext10:
    """TRT >= 10: set_tensor_address per name, then execute_async_v3(stream handle)."""

    def __init__(self):
        self.addresses = {}
        self.stream_handles = []

    def set_tensor_address(self, name, address):
        self.addresses[name] = address

    def execute_async_v3(self, handle):
        self.stream_handles.append(handle)
        return True


class TestEngineExecution:
    """Stubbed buffers/context — verifies the plumbing, NOT real inference."""

    def _loaded(self, tmp_path, binding_order, context, has_cam):
        obj = _bare(tmp_path, _projection(tmp_path))
        obj._cuda = _FakeCuda()
        obj._context = context
        obj._binding_order = binding_order
        obj.has_cam_input = has_cam
        inputs = {IMAGE_BINDING: {'dtype': 'float32', 'shape': [1, 3, 256, 128]}}
        if has_cam:
            inputs[CAM_BINDING] = {'dtype': 'int64', 'shape': [1]}
        obj._bindings = _bindings(inputs,
                                  {FEATURE_BINDING: {'dtype': 'float32', 'shape': [1, 768]}})
        obj._allocate_buffers()
        return obj

    def test_trt8_addresses_follow_engine_binding_order(self, tmp_path):
        ctx = _FakeContext8()
        obj = self._loaded(tmp_path, [IMAGE_BINDING, FEATURE_BINDING], ctx, has_cam=False)
        obj._host[FEATURE_BINDING][0, :] = 0.0
        out = obj._run_engine()
        assert ctx.calls == [[int(obj._addresses[IMAGE_BINDING]),
                              int(obj._addresses[FEATURE_BINDING])]]
        assert out.shape == (768,)

    def test_trt8_sie_engine_stages_the_camera_index(self, tmp_path):
        ctx = _FakeContext8()
        obj = self._loaded(tmp_path, [IMAGE_BINDING, CAM_BINDING, FEATURE_BINDING], ctx,
                           has_cam=True)
        obj._execute(np.zeros((1, 3, 256, 128), np.float32), cam_label=4)
        assert int(obj._addresses[CAM_BINDING]) in ctx.calls[0]
        assert obj._addresses[CAM_BINDING].copied_in[-1][0] == 4

    def test_trt10_sets_tensor_addresses_by_name(self, tmp_path):
        ctx = _FakeContext10()
        obj = self._loaded(tmp_path, [IMAGE_BINDING, FEATURE_BINDING], ctx, has_cam=False)
        obj._run_engine()
        assert ctx.addresses == {IMAGE_BINDING: int(obj._addresses[IMAGE_BINDING]),
                                 FEATURE_BINDING: int(obj._addresses[FEATURE_BINDING])}
        assert ctx.stream_handles == [77]

    def test_failed_execution_raises_rather_than_returning_stale_features(self, tmp_path):
        ctx = _FakeContext8()
        ctx.ok = False
        obj = self._loaded(tmp_path, [IMAGE_BINDING, FEATURE_BINDING], ctx, has_cam=False)
        obj._host[FEATURE_BINDING][0, :] = 42.0
        with pytest.raises(RuntimeError, match='TensorRT execution returned False'):
            obj._run_engine()

    def test_cam_label_direction_is_checked_both_ways(self, tmp_path):
        obj = self._loaded(tmp_path, [IMAGE_BINDING, FEATURE_BINDING], _FakeContext8(),
                           has_cam=False)
        with pytest.raises(ValueError, match='camera-agnostic engine'):
            obj._execute(np.zeros((1, 3, 256, 128), np.float32), cam_label=2)
        obj2 = self._loaded(tmp_path, [IMAGE_BINDING, CAM_BINDING, FEATURE_BINDING],
                            _FakeContext8(), has_cam=True)
        with pytest.raises(ValueError, match='takes a cam_label input'):
            obj2._execute(np.zeros((1, 3, 256, 128), np.float32), cam_label=None)

    def test_staged_feature_becomes_the_sentinel_after_a_failed_run(self, tmp_path):
        """A stale buffer must never masquerade as a fresh embedding."""
        obj = self._loaded(tmp_path, [IMAGE_BINDING, FEATURE_BINDING], _FakeContext8(ok=False),
                           has_cam=False)
        obj.has_cam_input = False
        out = obj.extract_embedding(_crop())
        assert np.all(out == 0.0)
        assert obj.last_failure['stage'] == 'inference_exception'
        assert obj.embedding_count == 0


class _Trt:
    """Module-level stand-in for the tensorrt module's bits that _read_bindings touches."""

    __version__ = 'stub'
    float32, float16, int8, int32, bool, int64 = range(6)
    WARNING = 2

    class TensorIOMode:
        INPUT, OUTPUT = 'input', 'output'

    @staticmethod
    def Logger(level):
        return 'logger'


class TestBindingApiReading:
    """_read_bindings must cope with both TensorRT IO APIs; a Jetson run depends on it."""

    class _Engine10:
        name = 'transreid_orin.engine'

        num_io_tensors = 2

        def get_tensor_name(self, i):
            return [IMAGE_BINDING, FEATURE_BINDING][i]

        def get_tensor_mode(self, name):
            return _Trt.TensorIOMode.INPUT if name == IMAGE_BINDING else \
                _Trt.TensorIOMode.OUTPUT

        def get_tensor_dtype(self, name):
            return _Trt.float32

        def get_tensor_shape(self, name):
            return [-1, 3, 256, 128] if name == IMAGE_BINDING else [-1, 768]

    class _Context10:
        def __init__(self):
            self.set_shapes = {}

        def get_tensor_shape(self, name):
            if name == IMAGE_BINDING:
                return [-1, 3, 256, 128]
            # TensorRT resolves the output's dynamic batch from the pinned input shape.
            batch = self.set_shapes.get(IMAGE_BINDING, (-1,))[0]
            return [batch, 768]

        def set_input_shape(self, name, shape):
            self.set_shapes[name] = tuple(shape)

    def test_trt10_dynamic_batch_is_pinned_to_one_for_the_metadata(self, tmp_path):
        obj = _bare(tmp_path, _projection(tmp_path))
        obj._trt = _Trt
        obj._engine = self._Engine10()
        obj._context = self._Context10()
        bindings = obj._read_bindings()
        assert bindings['inputs'][IMAGE_BINDING]['shape'] == [1, 3, 256, 128]
        assert bindings['outputs'][FEATURE_BINDING]['shape'] == [1, 768]
        assert obj._context.set_shapes[IMAGE_BINDING] == (1, 3, 256, 128)
        assert obj.engine_metadata['api'].startswith('num_io_tensors')
        assert obj.engine_metadata['binding_order'] == [IMAGE_BINDING, FEATURE_BINDING]
        assert obj.engine_metadata['tensorrt_version'] == 'stub'

    class _Engine8:
        name = 'transreid_orin.engine'
        num_bindings = 2

        def get_binding_name(self, i):
            return [IMAGE_BINDING, FEATURE_BINDING][i]

        def binding_is_input(self, i):
            return i == 0

        def get_binding_dtype(self, i):
            return _Trt.float32

        def get_binding_shape(self, i):
            return [1, 3, 256, 128] if i == 0 else [1, 768]

    class _Context8:
        def get_binding_shape(self, i):
            return [1, 3, 256, 128] if i == 0 else [1, 768]

    def test_trt8_bindings_are_read_through_the_legacy_api(self, tmp_path):
        obj = _bare(tmp_path, _projection(tmp_path))
        obj._trt = _Trt
        obj._engine = self._Engine8()
        obj._context = self._Context8()
        bindings = obj._read_bindings()
        assert bindings['inputs'][IMAGE_BINDING]['dtype'] == 'float32'
        assert bindings['outputs'][FEATURE_BINDING]['shape'] == [1, 768]
        assert obj.engine_metadata['api'] == 'num_bindings (TRT 8.x)'

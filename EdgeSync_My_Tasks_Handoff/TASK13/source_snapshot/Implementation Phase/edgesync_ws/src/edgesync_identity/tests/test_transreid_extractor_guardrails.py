"""TR-06 (checkpoint coverage) and TR-02 (attributable sentinel) tests for TransReIdExtractor.

Test class: UNIT. These exercise the extractor's own validation and failure bookkeeping. The
real 415 MB checkpoint is loaded in exactly one test (marked below); the fail-fast branches use
a monkeypatched tiny model so a damaged-checkpoint case does not cost 400 MB of RAM each run.
Nothing here measures accuracy or latency — see the offline evaluation for that.
"""
import os
import numpy as np
import pytest

T = pytest.importorskip('torch')
import torch.nn as nn

from edgesync_identity.models.transreid_extractor import TransReIdExtractor

REAL_CKPT = os.path.join(
    os.path.dirname(__file__), '..', '..', 'edgesync_bringup', 'models', 'transreid',
    'transformer_120.pth')
PROJECTION = os.path.join(
    os.path.dirname(__file__), '..', '..', 'edgesync_bringup', 'models', 'transreid',
    'svd_projection_768_to_512_real_v2.npz')


def write_checkpoint(tmp_path, name, payload):
    path = tmp_path / name
    T.save(payload, str(path))
    return str(path)


class _Tiny(nn.Module):
    """Stand-in for the ViT feature model: two parameters, known key names."""

    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(T.zeros(4))
        self.w2 = nn.Parameter(T.zeros(4))


class TestAssetFailFast:
    def test_missing_checkpoint_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            TransReIdExtractor(checkpoint_path=str(tmp_path / 'nope.pth'),
                               projection_path=str(tmp_path / 'also_nope.npz'))

    def test_missing_projection_raises_file_not_found(self, tmp_path):
        ckpt = write_checkpoint(tmp_path, 'ok.pth', {'w1': T.zeros(4), 'w2': T.zeros(4)})
        with pytest.raises(FileNotFoundError):
            TransReIdExtractor(checkpoint_path=ckpt, projection_path=str(tmp_path / 'nope.npz'))


class TestTR06CheckpointCoverageValidation:
    def _build(self, tmp_path, payload, monkeypatch, projection):
        monkeypatch.setattr('edgesync_identity.models.transreid_backbone.build_transreid_model',
                            lambda **kwargs: _Tiny())
        ckpt = write_checkpoint(tmp_path, 'ckpt.pth', payload)
        return TransReIdExtractor(checkpoint_path=ckpt, projection_path=projection)

    @staticmethod
    def _projection(tmp_path):
        path = tmp_path / 'proj.npz'
        np.savez(path, projection=np.ones((768, 512), dtype=np.float32),
                 mean=np.zeros(768, dtype=np.float32))
        return str(path)

    def test_full_coverage_loads_and_is_recorded(self, tmp_path, monkeypatch):
        ext = self._build(tmp_path, {'w1': T.ones(4), 'w2': T.ones(4)}, monkeypatch,
                          self._projection(tmp_path))
        assert ext.checkpoint_coverage['missing_keys'] == []
        assert ext.checkpoint_coverage['unexpected_keys'] == []
        assert ext.checkpoint_coverage['coverage_pct'] == 100.0

    def test_missing_model_parameter_hard_fails(self, tmp_path, monkeypatch):
        """A checkpoint that cannot fill the model must raise, not run with random weights."""
        with pytest.raises(RuntimeError, match='does not cover'):
            self._build(tmp_path, {'w1': T.ones(4)}, monkeypatch, self._projection(tmp_path))

    def test_tolerated_jpm_heads_stay_unexpected_but_pass(self, tmp_path, monkeypatch):
        payload = {'w1': T.ones(4), 'w2': T.ones(4)}
        for i in (1, 2, 3, 4):
            payload[f'classifier_{i}.weight'] = T.ones(751, 4)
        ext = self._build(tmp_path, payload, monkeypatch, self._projection(tmp_path))
        assert set(ext.checkpoint_coverage['unexpected_keys']) == \
            set(TransReIdExtractor.TOLERATED_UNEXPECTED_KEYS)

    def test_intolerable_unexpected_key_hard_fails(self, tmp_path, monkeypatch):
        payload = {'w1': T.ones(4), 'w2': T.ones(4), 'base.blocks.99.qkv.weight': T.ones(1)}
        with pytest.raises(RuntimeError, match='does not define'):
            self._build(tmp_path, payload, monkeypatch, self._projection(tmp_path))

    def test_classifier_head_keys_are_dropped_by_the_filter_not_unexpected(
            self, tmp_path, monkeypatch):
        payload = {'w1': T.ones(4), 'w2': T.ones(4), 'classifier.weight': T.ones(751, 4)}
        ext = self._build(tmp_path, payload, monkeypatch, self._projection(tmp_path))
        assert 'classifier.weight' in ext.checkpoint_coverage['dropped_by_filter']
        assert 'classifier.weight' not in ext.checkpoint_coverage['unexpected_keys']


def test_real_checkpoint_fully_covers_the_feature_model():
    """Confirmation against the deployed 415 MB checkpoint (matches probe PF-01). Heavy: builds
    the 100,769,512-parameter model, so it is run explicitly rather than in the fast suite."""
    if not os.path.isfile(REAL_CKPT) or not os.path.isfile(PROJECTION):
        pytest.skip(f'asset absent: {REAL_CKPT}')
    if os.environ.get('ES07_HEAVY') != '1':
        pytest.skip('heavy: loads the 415 MB checkpoint and builds 100.8M parameters — '
                    'run with ES07_HEAVY=1')
    ext = TransReIdExtractor(checkpoint_path=REAL_CKPT, projection_path=PROJECTION)
    cov = ext.checkpoint_coverage
    assert cov['missing_keys'] == []
    assert set(cov['unexpected_keys']) == set(TransReIdExtractor.TOLERATED_UNEXPECTED_KEYS)
    assert cov['torch_parameters'] == 100769512


def make_stub_extractor(model_callable, mean=None, proj=None):
    """An extractor instance without building the 100M-parameter model.

    Uses object.__new__ and injects only the attributes extract_embedding() touches, so the
    sentinel branches are tested in isolation from ViT inference.
    """
    ext = object.__new__(TransReIdExtractor)
    ext.checkpoint_path = 'stub.pth'
    ext.projection_path = 'stub.npz'
    ext.input_height, ext.input_width = 256, 128
    ext._device = T.device('cpu')
    ext._model = model_callable
    ext._svd_mean = np.zeros(768, dtype=np.float32) if mean is None else mean
    ext._svd_projection = (np.eye(768, 512, dtype=np.float32) if proj is None else proj)
    ext.last_failure = None
    ext.failure_counts = {}
    ext.embedding_count = 0
    ext.checkpoint_coverage = None
    return ext


class _Callable:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, tensor, cam_label=None):
        return self.fn(tensor, cam_label)


class TestTR02AttributableSentinel:
    """The frozen contract says: return zeros(512), never raise. These tests keep that and add
    the failure attribution the contract cannot express."""

    GOOD = T.ones(1, 768)

    def test_none_crop_returns_sentinel_without_raising(self):
        ext = make_stub_extractor(lambda t, cam_label=None: self.GOOD)
        out = ext.extract_embedding(None)
        assert out.shape == (512,) and np.all(out == 0.0)
        assert ext.last_failure['stage'] == 'input_empty'

    def test_two_dimensional_crop_recorded_as_shape_failure(self):
        ext = make_stub_extractor(lambda t, cam_label=None: self.GOOD)
        out = ext.extract_embedding(np.zeros((8, 8), dtype=np.uint8))
        assert np.all(out == 0.0)
        assert ext.last_failure['stage'] == 'input_shape'

    def test_tiny_crop_recorded_as_too_small(self):
        ext = make_stub_extractor(lambda t, cam_label=None: self.GOOD)
        out = ext.extract_embedding(np.zeros((10, 10, 3), dtype=np.uint8))
        assert np.all(out == 0.0)
        assert ext.last_failure['stage'] == 'input_too_small'

    def test_backbone_non_finite_recorded(self):
        bad = T.full((1, 768), float('nan'))
        ext = make_stub_extractor(lambda t, cam_label=None: bad)
        crop = np.full((64, 32, 3), 128, dtype=np.uint8)
        out = ext.extract_embedding(crop)
        assert np.all(out == 0.0)
        assert ext.last_failure['stage'] == 'backbone_non_finite'

    def test_projection_non_finite_recorded(self):
        ext = make_stub_extractor(lambda t, cam_label=None: self.GOOD,
                                  proj=np.full((768, 512), np.inf, dtype=np.float32))
        crop = np.full((64, 32, 3), 128, dtype=np.uint8)
        out = ext.extract_embedding(crop)
        assert np.all(out == 0.0)
        assert ext.last_failure['stage'] == 'projection_non_finite'

    def test_degenerate_projection_norm_recorded(self):
        ext = make_stub_extractor(lambda t, cam_label=None: self.GOOD,
                                  proj=np.zeros((768, 512), dtype=np.float32))
        crop = np.full((64, 32, 3), 128, dtype=np.uint8)
        out = ext.extract_embedding(crop)
        assert np.all(out == 0.0)
        assert ext.last_failure['stage'] == 'projection_degenerate'

    def test_inference_exception_is_captured_as_sentinel_not_propagated(self):
        def boom(tensor, cam_label=None):
            raise RuntimeError('CUDA out of memory')
        ext = make_stub_extractor(boom)
        crop = np.full((64, 32, 3), 128, dtype=np.uint8)
        out = ext.extract_embedding(crop)          # must not raise (frozen contract)
        assert np.all(out == 0.0)
        assert ext.last_failure['stage'] == 'inference_exception'
        assert 'CUDA out of memory' in ext.last_failure['reason']

    def test_failure_counts_accumulate_per_stage(self):
        ext = make_stub_extractor(lambda t, cam_label=None: self.GOOD)
        for _ in range(3):
            ext.extract_embedding(None)
        ext.extract_embedding(np.zeros((4, 4), dtype=np.uint8))
        assert ext.failure_counts['input_empty'] == 3
        assert ext.failure_counts['input_shape'] == 1

    def test_success_clears_last_failure_and_counts_embedding(self):
        rng = np.random.default_rng(7)
        proj = rng.standard_normal((768, 512)).astype(np.float32)
        ext = make_stub_extractor(lambda t, cam_label=None: T.rand(1, 768), proj=proj)
        crop = np.full((64, 32, 3), 200, dtype=np.uint8)
        ext.extract_embedding(None)                       # fail first
        assert ext.last_failure is not None
        out = ext.extract_embedding(crop)                 # then succeed
        assert ext.last_failure is None
        assert ext.embedding_count == 1
        assert pytest.approx(float(np.linalg.norm(out)), abs=1e-5) == 1.0
        assert np.all(np.isfinite(out))
        assert not np.all(out == 0.0)

    def test_sentinel_carries_checkpoint_path_for_post_hoc_diagnosis(self):
        ext = make_stub_extractor(lambda t, cam_label=None: self.GOOD)
        ext.extract_embedding(None)
        assert ext.last_failure['checkpoint'] == 'stub.pth'
        assert 'monotonic_s' in ext.last_failure

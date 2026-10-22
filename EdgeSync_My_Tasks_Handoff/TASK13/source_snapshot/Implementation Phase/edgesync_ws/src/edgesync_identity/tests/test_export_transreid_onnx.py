"""TR-09 — focused regression tests for scripts/export_transreid_onnx.py.

UNIT-level: they exercise the pure helpers and the exporter plumbing on a tiny resolution-
independent MLP, so graph validation, the input-contract rules and the parity arithmetic are all
checked without loading the 414 MB checkpoint. The real TransReID export and parity run is a
separate executed command recorded in ES-EXP07-EXECUTION-AND-CHANGE-LOG.md §5; nothing here
measures model accuracy.

They verify the design claims that matter:
  * the agnostic graph exposes exactly one input, so "no SIE" cannot be faked with cam_label=0;
  * the SIE graph exposes image + int64 cam_label;
  * a graph input that violates the deployed wire contract is reported, not accepted;
  * a degenerate or non-finite feature raises instead of returning a substitute vector.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

_SCRIPT = os.path.join(pkg_dir, 'scripts', 'export_transreid_onnx.py')
_spec = importlib.util.spec_from_file_location('export_transreid_onnx', _SCRIPT)
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)

torch = pytest.importorskip('torch')
onnx = pytest.importorskip('onnx')

CONTRACT_SHAPE = (1, 3, 256, 128)


class _RecordingBackbone(torch.nn.Module):
    """Stand-in for the ViT: resolution-independent, uses cam_label, and records what it got.

    Using the camera input matters — a traced graph silently drops an input that contributes to
    no output, and that is exactly the failure the expected-input check exists to catch.
    """

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, kernel_size=1)
        self.proj = torch.nn.Linear(1, 4)
        self.seen_cam_label = 'NOT_CALLED'

    def forward(self, image, cam_label=None):
        self.seen_cam_label = cam_label
        pooled = self.conv(image).flatten(2).mean(dim=2)
        if cam_label is not None:
            pooled = pooled + cam_label.to(pooled.dtype).reshape(-1, 1)
        return self.proj(pooled)


def _vi(name, elem_type, dims):
    return onnx.helper.make_tensor_value_info(name, elem_type, dims)


def _export_tiny(tmp_path, with_cam, shape=CONTRACT_SHAPE):
    backbone = _RecordingBackbone()
    wrapper = ex.make_feature_graph(backbone, with_cam=with_cam)
    names = ['image', 'cam_label'] if with_cam else ['image']
    inputs = [torch.zeros(*shape)]
    axes = {'image': {0: 'batch'}, 'feature_768': {0: 'batch'}}
    if with_cam:
        inputs.append(torch.tensor([2], dtype=torch.long))
        axes['cam_label'] = {0: 'batch'}
    path = str(tmp_path / f'tiny_cam{int(with_cam)}_{shape[2]}x{shape[3]}.onnx')
    ex.export(wrapper, inputs, names, path, opset=17, dynamic_axes=axes, dynamo=False)
    return path, backbone, wrapper, inputs


class TestSha256:
    def test_digest_matches_known_content(self, tmp_path):
        f = tmp_path / 'blob.bin'
        f.write_bytes(b'abc')
        assert ex.sha256_of(str(f)) == (
            'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')

    def test_chunked_read_covers_the_whole_file(self, tmp_path):
        f = tmp_path / 'big.bin'
        f.write_bytes(b'x' * 1000)
        assert ex.sha256_of(str(f), chunk=64) == ex.sha256_of(str(f), chunk=1 << 20)


class TestParityMetrics:
    def test_reports_all_five_statistics(self):
        m = ex.parity_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 5.0])
        assert m['max_abs'] == pytest.approx(2.0)
        assert m['mean_abs'] == pytest.approx(2.0 / 3.0)
        assert m['rmse'] == pytest.approx(np.sqrt(4.0 / 3.0))
        assert m['l2_dist'] == pytest.approx(2.0)
        assert m['cosine'] == pytest.approx(20.0 / (np.sqrt(14) * np.sqrt(30)))

    def test_identical_vectors_are_exact(self):
        v = np.linspace(-1.0, 1.0, 768)
        m = ex.parity_metrics(v, v)
        assert (m['max_abs'], m['l2_dist']) == (0.0, 0.0)
        assert m['cosine'] == pytest.approx(1.0)

    def test_shape_mismatch_raises_rather_than_broadcasting(self):
        with pytest.raises(ValueError, match='parity shape mismatch'):
            ex.parity_metrics([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_zero_reference_gives_nan_cosine_not_a_division_crash(self):
        assert np.isnan(ex.parity_metrics([0.0, 0.0], [1.0, 1.0])['cosine'])

    def test_row_and_column_shapes_compare_flattened(self):
        assert ex.parity_metrics(np.zeros((1, 4)), np.zeros((4,)))['max_abs'] == 0.0


class TestProjectAndNormalise:
    def setup_method(self):
        self.mean = np.arange(4, dtype=np.float32)
        self.proj = np.eye(4, dtype=np.float32)

    def test_matches_the_documented_centre_then_project_then_normalise(self):
        out = ex.project_and_normalise([1.0, 2.0, 3.0, 4.0], self.mean, self.proj)
        expected = np.array([1, 2, 3, 4], dtype=np.float32) - self.mean
        expected /= np.linalg.norm(expected)
        assert np.allclose(out, expected)
        assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-6)

    def test_non_finite_feature_raises_instead_of_returning_a_substitute(self):
        with pytest.raises(ValueError, match='non-finite'):
            ex.project_and_normalise([1.0, np.nan, 3.0, 4.0], self.mean, self.proj)
        with pytest.raises(ValueError, match='non-finite'):
            ex.project_and_normalise([np.inf, 2.0, 3.0, 4.0], self.mean, self.proj)

    def test_degenerate_projection_raises_instead_of_returning_zeros(self):
        with pytest.raises(ValueError, match='degenerate'):
            ex.project_and_normalise(self.mean, self.mean, self.proj)

    def test_uses_the_real_projection_geometry(self):
        proj = np.ones((4, 2), dtype=np.float32)
        out = ex.project_and_normalise([1.0] * 4, np.zeros(4, np.float32), proj)
        assert out.shape == (2,)
        assert np.allclose(out, 1.0 / np.sqrt(2.0))


class TestIdentityOf:
    def test_parses_pid_and_camera_number(self):
        assert ex.identity_of('/x/0752_c1s4_001234_05.jpg') == (752, 1)
        assert ex.identity_of('/x/0_c6s2_019999_03.jpg') == (0, 6)

    def test_uses_the_population_grammar_not_a_naive_split(self):
        assert ex.identity_of('/x/1501_c5s1_083100_00.jpg') == (1501, 5)

    def test_non_market_filename_raises(self):
        with pytest.raises(ValueError, match='grammar'):
            ex.identity_of('/x/README.jpg')

    def test_junk_identity_name_is_parseable_and_negative(self):
        assert ex.identity_of('/x/-1_c1s1_000151_03.jpg') == (-1, 1)


class TestCheckInputs:
    def test_contract_inputs_produce_no_violations(self):
        ins = [_vi('image', onnx.TensorProto.FLOAT, ['batch', 3, 256, 128]),
               _vi('cam_label', onnx.TensorProto.INT64, ['batch'])]
        assert ex.check_inputs(ins, {'image', 'cam_label'}) == []

    def test_cam_label_is_not_wrongly_required_to_be_float(self):
        """The regression a blanket FLOAT rule would have caused: int64 camera input flagged."""
        ins = [_vi('image', onnx.TensorProto.FLOAT, [1, 3, 256, 128]),
               _vi('cam_label', onnx.TensorProto.INT64, [1])]
        assert ex.check_inputs(ins, {'image', 'cam_label'}) == []

    def test_wrong_spatial_resolution_is_reported(self):
        ins = [_vi('image', onnx.TensorProto.FLOAT, [1, 3, 128, 64])]
        v = ex.check_inputs(ins, {'image'})
        assert len(v) == 1 and 'spatial dims' in v[0]

    def test_wrong_rank_and_dtype_are_reported(self):
        ins = [_vi('image', onnx.TensorProto.FLOAT, [3, 256, 128]),
               _vi('cam_label', onnx.TensorProto.FLOAT, [1])]
        v = ex.check_inputs(ins, {'image', 'cam_label'})
        assert any('rank 3' in x for x in v)
        assert any('dtype FLOAT' in x and 'INT64' in x for x in v)

    def test_extra_graph_input_is_reported(self):
        ins = [_vi('image', onnx.TensorProto.FLOAT, [1, 3, 256, 128]),
               _vi('bogus', onnx.TensorProto.FLOAT, [1])]
        v = ex.check_inputs(ins, {'image'})
        assert any(x.startswith('bogus') and 'not part of the expected' in x for x in v)

    def test_missing_expected_input_is_reported(self):
        ins = [_vi('image', onnx.TensorProto.FLOAT, [1, 3, 256, 128])]
        assert ex.check_inputs(ins, {'image', 'cam_label'}) == [
            'cam_label: expected input absent from the graph']


class TestExportPlumbing:
    """Real exports of a tiny network through the same code path the checkpoint uses."""

    def test_agnostic_graph_exposes_exactly_one_input(self, tmp_path):
        path, _, _, _ = _export_tiny(tmp_path, with_cam=False)
        assert [i.name for i in onnx.load(path).graph.input] == ['image']

    def test_sie_graph_exposes_image_and_int64_cam_label(self, tmp_path):
        path, _, _, _ = _export_tiny(tmp_path, with_cam=True)
        ins = {i.name: i for i in onnx.load(path).graph.input}
        assert set(ins) == {'image', 'cam_label'}
        assert ins['cam_label'].type.tensor_type.elem_type == onnx.TensorProto.INT64

    def test_agnostic_wrapper_calls_the_backbone_with_none(self, tmp_path):
        """cam_label=0 is not "no SIE"; the wrapper must pass None through."""
        _, backbone, wrapper, inputs = _export_tiny(tmp_path, with_cam=False)
        with torch.no_grad():
            wrapper(*inputs[:1])
        assert backbone.seen_cam_label is None

    def test_sie_wrapper_passes_the_camera_tensor_through(self, tmp_path):
        _, backbone, wrapper, inputs = _export_tiny(tmp_path, with_cam=True)
        cam = torch.tensor([3], dtype=torch.long)
        with torch.no_grad():
            wrapper(inputs[0], cam)
        assert torch.equal(backbone.seen_cam_label, cam)

    def test_inspect_graph_reports_the_structure(self, tmp_path):
        path, _, _, _ = _export_tiny(tmp_path, with_cam=True)
        info = ex.inspect_graph(path, {'image', 'cam_label'})
        assert info['checker'].startswith('PASS')
        assert info['bytes'] == os.path.getsize(path)
        assert info['sha256'] == ex.sha256_of(path)
        assert [o['name'] for o in info['outputs']] == ['feature_768']
        assert info['outputs'][0]['dtype'] == 'FLOAT'
        assert info['n_nodes'] > 0
        assert sum(info['op_type_histogram'].values()) == info['n_nodes']
        assert 'Conv' in info['op_type_histogram'] and 'Gemm' in info['op_type_histogram']
        assert info['external_data_tensors'] == []
        # Only the batch axis is dynamic in this graph; the feature width stays a constant.
        assert info['output_symbolic_dims'] == {'feature_768': ['batch']}
        assert info['n_initializers'] == 4  # conv w+b, gemm w+b
        assert info['initializer_float32_bytes'] == (3 + 1 + 4 + 4) * 4
        assert info['opset_imports']['ai.onnx'] == 17

    def test_inspect_graph_passes_the_deployed_input_contract(self, tmp_path):
        path, _, _, _ = _export_tiny(tmp_path, with_cam=True)
        info = ex.inspect_graph(path, {'image', 'cam_label'})
        assert info['input_contract_violations'] == []
        image = [i for i in info['inputs'] if i['name'] == 'image'][0]
        assert image['dtype'] == 'FLOAT' and image['shape'][1:] == [3, 256, 128]

    def test_inspect_graph_reports_an_off_contract_exported_resolution(self, tmp_path):
        """The validator bites on a real graph, not only on hand-built protos."""
        path, _, _, _ = _export_tiny(tmp_path, with_cam=False, shape=(1, 3, 128, 64))
        info = ex.inspect_graph(path, {'image'})
        assert any('spatial dims' in v for v in info['input_contract_violations'])

    def test_dynamic_batch_is_accepted_by_the_exported_graph(self, tmp_path):
        ort = pytest.importorskip('onnxruntime')
        path, _, _, _ = _export_tiny(tmp_path, with_cam=True)
        sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        out = sess.run(None, {'image': np.zeros((2, 3, 256, 128), np.float32),
                              'cam_label': np.array([0, 1], np.int64)})[0]
        assert out.shape == (2, 4)

    def test_export_failure_is_catchable_by_the_caller(self, tmp_path):
        """A failed export is recorded, never written up as a successful deployment."""
        class _Broken(torch.nn.Module):
            def forward(self, image):
                raise RuntimeError('unsupported op')

        with pytest.raises(RuntimeError, match='unsupported op'):
            ex.export(_Broken(), [torch.zeros(1, 3, 256, 128)], ['image'],
                      str(tmp_path / 'broken.onnx'), 17, {}, False)


class TestReportWriting:
    def test_numpy_scalars_stay_numeric_in_the_json(self, tmp_path):
        ex._write({'worst': np.float32(0.001), 'n': np.int64(8), 'flag': True}, str(tmp_path))
        loaded = json.loads((tmp_path / 'transreid_onnx_report.json').read_text())
        assert loaded['worst'] == pytest.approx(0.001, abs=1e-6)
        assert isinstance(loaded['worst'], float) and isinstance(loaded['n'], int)
        assert loaded['flag'] is True

    def test_numpy_arrays_become_lists_instead_of_raising(self, tmp_path):
        ex._write({'digest_bytes': np.arange(4, dtype=np.uint8)}, str(tmp_path))
        loaded = json.loads((tmp_path / 'transreid_onnx_report.json').read_text())
        assert loaded['digest_bytes'] == [0, 1, 2, 3]

    def test_unserialisable_objects_degrade_to_text(self, tmp_path):
        ex._write({'device': object()}, str(tmp_path))
        assert 'object' in (tmp_path / 'transreid_onnx_report.json').read_text()

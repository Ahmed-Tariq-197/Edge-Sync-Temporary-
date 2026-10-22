"""TR-02 / TR-07 — offline evaluation population and exclusion accounting.

UNIT-level tests over a synthetic Market-1501-shaped tree (a handful of filenames in a tmp dir).
They verify the bookkeeping in scripts/evaluate_offline.py: which images enter the population,
how every exclusion is recorded, and that an unusable image is removed instead of receiving a
substitute embedding.

They are NOT accuracy measurements: no real dataset, no real checkpoint, no Re-ID forward pass.
The metrics themselves are covered by the validated canonical implementation imported by
``compute_cmc_map`` and tested in ``test_experiment7_fixes.py``.
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

_SCRIPT = os.path.join(pkg_dir, 'scripts', 'evaluate_offline.py')
_spec = importlib.util.spec_from_file_location('evaluate_offline', _SCRIPT)
eo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eo)

def _touch_tree(root, gallery_names, query_names):
    """Create a Market-1501-shaped tree of empty files — population selection globs names only."""
    gdir = os.path.join(root, 'bounding_box_test')
    qdir = os.path.join(root, 'query')
    os.makedirs(gdir, exist_ok=True)
    os.makedirs(qdir, exist_ok=True)
    for name in gallery_names:
        open(os.path.join(gdir, name), 'wb').close()
    for name in query_names:
        open(os.path.join(qdir, name), 'wb').close()
    return root


GALLERY = [
    '-1_c1s1_000001_00.jpg',          # junk, pid -1
    '0000_c1s1_000001_00.jpg',        # distractor, pid 0
    '0751_c1s1_000001_00.jpg',        # below the ID band
    '0752_c1s1_000001_00.jpg',        # inside both populations
    '1501_c2s3_000001_00.jpg',        # upper band edge
    'README_frame.jpg',               # does not match the Market-1501 grammar
]
QUERY = [
    '-1_c3s1_000001_00.jpg',
    '0000_c3s1_000001_00.jpg',        # distractor is never a valid query
    '0751_c3s1_000001_00.jpg',
    '0752_c3s1_000001_00.jpg',
]


class TestFilenameClassification:
    def test_standard_name_parses_pid_and_camera(self):
        assert eo._classify_name('0752_c1s1_000001_03.jpg') == (752, 1, None)

    def test_junk_and_distractor_numbers_parse(self):
        assert eo._classify_name('-1_c1s1_000001_00.jpg') == (-1, 1, None)
        assert eo._classify_name('0000_c6s2_033716_02.jpg') == (0, 6, None)

    def test_unparsable_name_is_rejected_not_defaulted(self):
        pid, camid, reason = eo._classify_name('README_frame.jpg')
        assert reason == 'unparsable_filename'
        assert pid is None and camid is None

    def test_non_jpg_suffix_is_rejected(self):
        assert eo._classify_name('0752_c1s1_000001_00.png')[2] == 'unparsable_filename'

    def test_missing_sequence_field_is_rejected(self):
        # The lenient parse_market1501_filename() would return (752, 1) for this; the strict
        # grammar is what decides population membership, so it has to reject it.
        assert eo._classify_name('0752_c1_000001.jpg')[2] == 'unparsable_filename'


class TestPopulationSelection:
    def test_directory_population_matches_official_split_semantics(self, tmp_path):
        root = _touch_tree(str(tmp_path), GALLERY, QUERY)
        gallery, query, manifest = eo.collect_gallery_and_query(root, population='directory')

        g_pids = sorted(p for _, p, _ in gallery)
        q_pids = sorted(p for _, p, _ in query)
        assert g_pids == [0, 751, 752, 1501]
        assert q_pids == [751, 752]
        assert manifest['gallery']['excluded'] == {'junk_identity_-1': 1,
                                                  'unparsable_filename': 1}
        assert manifest['query']['excluded'] == {'junk_identity_-1': 1,
                                                 'distractor_not_valid_query': 1}

    def test_counts_are_measured_from_the_directories_not_assumed(self, tmp_path):
        root = _touch_tree(str(tmp_path), GALLERY, QUERY)
        _, _, manifest = eo.collect_gallery_and_query(root, population='directory')
        assert manifest['gallery']['files_scanned'] == len(GALLERY)
        assert manifest['query']['files_scanned'] == len(QUERY)
        assert manifest['gallery']['kept'] == 4
        assert manifest['query']['kept'] == 2
        assert manifest['gallery']['distractors_pid_0'] == 1
        assert manifest['query']['identities'] == 2
        assert manifest['counts_asserted_from']

    def test_id_band_population_drops_distractors_and_low_ids(self, tmp_path):
        root = _touch_tree(str(tmp_path), GALLERY, QUERY)
        gallery, query, manifest = eo.collect_gallery_and_query(root, population='id_band_752_1501')
        assert sorted(p for _, p, _ in gallery) == [752, 1501]
        assert sorted(p for _, p, _ in query) == [752]
        assert manifest['gallery']['distractors_pid_0'] == 0
        assert manifest['gallery']['excluded']['outside_id_band_752_1501'] == 2
        assert manifest['id_band'] == {'id_min': 752, 'id_max': 1501}

    def test_band_bounds_are_cli_controlled_not_hardcoded(self, tmp_path):
        root = _touch_tree(str(tmp_path), GALLERY, QUERY)
        gallery, _, _ = eo.collect_gallery_and_query(
            root, population='id_band_752_1501', id_min=1, id_max=1000)
        assert sorted(p for _, p, _ in gallery) == [751, 752]

    def test_manifest_carries_the_rule_text_and_validity_statement(self, tmp_path):
        root = _touch_tree(str(tmp_path), GALLERY, QUERY)
        _, _, manifest = eo.collect_gallery_and_query(root, population='id_band_752_1501')
        assert '752 <= pid <= 1501' in manifest['rules']['gallery']
        assert manifest['provenance']
        assert 'INVALID' in manifest['validity']
        assert manifest['id_band'] is not None
        _, _, dir_manifest = eo.collect_gallery_and_query(root, population='directory')
        assert dir_manifest['id_band'] is None
        assert 'MEASURED AS INVALID' not in dir_manifest['validity']

    def test_default_population_is_the_valid_one(self, tmp_path):
        assert eo.DEFAULT_POPULATION == 'directory'
        root = _touch_tree(str(tmp_path), GALLERY, QUERY)
        _, _, manifest = eo.collect_gallery_and_query(root)
        assert manifest['population'] == 'directory'

    def test_band_paths_are_absolute_and_custom_bounds_relabel_the_validity_text(
            self, tmp_path, monkeypatch):
        root = _touch_tree(str(tmp_path), GALLERY, QUERY)
        monkeypatch.chdir(str(tmp_path.parent))
        _, _, manifest = eo.collect_gallery_and_query(
            tmp_path.name, population='id_band_752_1501', id_min=10, id_max=1000)
        assert os.path.isabs(manifest['gallery_dir'])
        assert os.path.isdir(manifest['gallery_dir'])
        assert 'this run used 10-1000' in manifest['validity']

        _, _, documented = eo.collect_gallery_and_query(root, population='id_band_752_1501')
        assert 'this run used' not in documented['validity']

    def test_unknown_population_raises_instead_of_silently_defaulting(self, tmp_path):
        root = _touch_tree(str(tmp_path), GALLERY, QUERY)
        with pytest.raises(ValueError, match='unknown population'):
            eo.collect_gallery_and_query(root, population='whatever')

    def test_missing_split_directories_raise(self, tmp_path):
        with pytest.raises(FileNotFoundError, match='Gallery directory'):
            eo.collect_gallery_and_query(str(tmp_path))


class TestDistractorRejection:
    def _feats(self, rows):
        return np.asarray(rows, dtype=np.float32)

    def test_no_distractors_yields_none_not_zero(self):
        rate, rejected, total = eo.compute_distractor_rejection(
            self._feats([[1.0, 0.0]]), self._feats([[1.0, 0.0]]),
            self._feats([752]), threshold=0.70)
        assert (rate, rejected, total) == (None, 0, 0)

    def test_distractor_far_from_every_query_is_rejected(self):
        query = self._feats([[1.0, 0.0]])
        gallery = self._feats([[0.0, 1.0], [1.0, 0.0]])
        pids = self._feats([0, 752])
        rate, rejected, total = eo.compute_distractor_rejection(query, gallery, pids, 0.70)
        assert total == 1 and rejected == 1 and rate == pytest.approx(1.0)

    def test_distractor_similar_to_a_query_is_not_rejected(self):
        # cosine 0.994 against the query: a false match at tau = 0.70, so not rejected.
        query = self._feats([[1.0, 0.0]])
        gallery = self._feats([[0.9, 0.1], [1.0, 0.0]])
        pids = self._feats([0, 752])
        rate, rejected, total = eo.compute_distractor_rejection(query, gallery, pids, 0.70)
        assert (rejected, total) == (0, 1)
        assert rate == pytest.approx(0.0)


class _ScriptedExtractor:
    """Returns a pre-scripted vector per call so a failure path can be aimed at one image."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.last_failure = {'stage': 'input_too_small', 'reason': 'crop shorter than 16px'}
        self.shapes = []

    def extract_embedding(self, image_crop, cam_label=None):
        self.shapes.append(image_crop.shape)
        return self.outputs.pop(0)


class TestExclusionInsteadOfZeroInjection:
    """``extract_features_batch`` returns kept indices into its input list, so the caller can
    keep pid/camid metadata aligned after an image is dropped (evaluate_model relies on this)."""

    _UNIT = np.ones(512, dtype=np.float32) / np.sqrt(512)
    _ZERO = np.zeros(512, dtype=np.float32)

    def _write_images(self, tmp_path, names):
        import cv2
        paths = []
        for name in names:
            path = str(tmp_path / name)
            if name == 'corrupt.jpg':
                with open(path, 'wb') as handle:
                    handle.write(b'\x00\x01\x02not an image')
            else:
                assert cv2.imwrite(path, np.full((24, 12, 3), 128, dtype=np.uint8))
            paths.append(path)
        return paths

    def test_undecodable_and_sentinel_images_are_excluded_with_reasons(self, tmp_path):
        paths = self._write_images(tmp_path, ['a.jpg', 'corrupt.jpg', 'b.jpg'])
        # The script feeds decodable images only: a.jpg -> valid, b.jpg -> sentinel.
        extractor = _ScriptedExtractor([self._UNIT.copy(), self._ZERO.copy()])

        feats, lat, kept, excluded = eo.extract_features_batch(extractor, paths, batch_size=2)

        assert feats.shape == (1, 512)
        assert kept == [0]
        assert len(lat) == 1
        reasons = {e['index']: e['reason'] for e in excluded}
        assert reasons[1] == 'decode_failed'
        assert 'invalid_embedding' in reasons[2]

    def test_no_zero_row_ever_reaches_the_feature_matrix(self, tmp_path):
        paths = self._write_images(tmp_path, ['a.jpg', 'b.jpg', 'c.jpg'])
        extractor = _ScriptedExtractor([self._UNIT.copy(), self._ZERO.copy(), self._UNIT.copy()])

        feats, _, kept, excluded = eo.extract_features_batch(extractor, paths)

        assert kept == [0, 2]
        assert np.all(np.linalg.norm(feats, axis=1) > 1e-6)
        assert 'invalid_embedding' in excluded[0]['reason']

    def test_kept_indices_realign_the_gallery_metadata(self, tmp_path):
        paths = self._write_images(tmp_path, ['a.jpg', 'b.jpg', 'c.jpg', 'd.jpg'])
        triples = [(paths[i], pid, cam)
                   for i, (pid, cam) in enumerate([(0, 1), (751, 5), (752, 1), (1501, 6)])]
        extractor = _ScriptedExtractor(
            [self._UNIT.copy(), self._ZERO.copy(), self._UNIT.copy(), self._UNIT.copy()])

        _, _, kept, _ = eo.extract_features_batch(extractor, [t[0] for t in triples])

        assert kept == [0, 2, 3]
        assert [triples[i][1] for i in kept] == [0, 752, 1501]

    def test_extractor_stage_is_surfaced_when_the_extractor_reports_one(self, tmp_path):
        paths = self._write_images(tmp_path, ['a.jpg', 'b.jpg'])
        extractor = _ScriptedExtractor([self._ZERO.copy(), self._UNIT.copy()])
        _, _, _, excluded = eo.extract_features_batch(extractor, paths)
        assert excluded[0]['extractor_stage'] == 'input_too_small'

    def test_run_fails_when_no_image_yields_an_embedding(self, tmp_path):
        paths = self._write_images(tmp_path, ['a.jpg'])
        extractor = _ScriptedExtractor([self._ZERO.copy()])
        with pytest.raises(RuntimeError, match='0 of 1 images'):
            eo.extract_features_batch(extractor, paths)

    def test_wrong_dimension_is_excluded_not_padded(self, tmp_path):
        paths = self._write_images(tmp_path, ['a.jpg', 'b.jpg'])
        extractor = _ScriptedExtractor([np.ones(768, dtype=np.float32), self._UNIT.copy()])
        feats, _, kept, excluded = eo.extract_features_batch(extractor, paths)
        assert kept == [1] and feats.shape[0] == 1
        assert 'dimension 768' in excluded[0]['reason']

    def test_every_kept_image_carries_one_real_latency(self, tmp_path):
        paths = self._write_images(tmp_path, ['a.jpg', 'b.jpg', 'c.jpg'])
        extractor = _ScriptedExtractor([self._UNIT.copy()] * 3)
        feats, lat, kept, excluded = eo.extract_features_batch(extractor, paths, batch_size=1)
        assert excluded == [] and kept == [0, 1, 2]
        assert len(lat) == len(kept) == feats.shape[0]
        assert all(isinstance(v, float) and v >= 0.0 for v in lat)

    def test_input_is_decoded_bgr_to_rgb_before_extraction(self, tmp_path):
        import cv2
        path = str(tmp_path / 'a.jpg')
        bgr = np.zeros((24, 12, 3), dtype=np.uint8)
        bgr[:, :, 2] = 200  # blue channel only
        assert cv2.imwrite(path, bgr)
        extractor = _ScriptedExtractor([self._UNIT.copy()])
        eo.extract_features_batch(extractor, [path])
        assert extractor.shapes == [(24, 12, 3)]
        # nothing further is asserted about channel order here; that contract belongs to
        # test_transreid_input_adapter.py and the extractor guardrails.


class TestComparisonTableRendering:
    def test_none_metric_prints_not_applicable(self):
        assert eo._format_metric('{:.2f}', None) == 'NOT APPLICABLE'

    def test_numeric_metric_still_formats(self):
        assert eo._format_metric('{:.2f}', 91.234) == '91.23'
        assert eo._format_metric('{:d}', 512) == '512'

    def test_string_metric_passes_through(self):
        assert eo._format_metric('{:.2f}', 'NOT MEASURED') == 'NOT MEASURED'

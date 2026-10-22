"""ES-EXP07 §4.4 — the offline experiment id must survive script -> manifest -> report.

``--exp-id`` on the two Task 7 scripts is only worth adding if the value is *used*: the generator
stamps it into ``task7_embeddings_manifest.json``, the analysis compares its own label against that
stamp and aborts on a disagreement, and the outcome is written into both report files. The
end-to-end cases here run the real ``analyze_task7_statistics.py`` CLI against the synthetic
embeddings built by the shipped ``test_analyze_task7_statistics.py`` fixture, so the gate, the abort
and the report field are measured rather than re-implemented. No recorded artefact is touched —
every ``--out-dir`` is a fresh temporary directory — and no extraction runs (charter §1.4).

Test class: UNIT (pure function + subprocess of a CPU-only analysis on 5x4 synthetic rows).
"""
import ast
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

SCRIPTS = os.path.join(pkg_dir, 'scripts')
GENERATOR = os.path.join(SCRIPTS, 'generate_task7_embeddings.py')
ANALYSER = os.path.join(SCRIPTS, 'analyze_task7_statistics.py')
NODE = os.path.join(pkg_dir, 'edgesync_identity', 'nodes', 'identity_node.py')
WS_SRC = os.path.abspath(os.path.join(pkg_dir, '..'))
REID_DEFAULTS = os.path.join(WS_SRC, 'edgesync_bringup', 'edgesync_bringup',
                             'reid_launch_defaults.py')


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = _load_by_path('generate_task7_embeddings', GENERATOR)
mod = _load_by_path('analyze_task7_statistics', ANALYSER)
shipped = _load_by_path('test_analyze_task7_statistics',
                        os.path.join(os.path.dirname(__file__),
                                     'test_analyze_task7_statistics.py'))


def _env():
    env = dict(os.environ)
    env['PYTHONIOENCODING'] = 'utf-8'
    return env


def _cli(path, extra):
    return subprocess.run([sys.executable, path] + list(extra), capture_output=True, text=True,
                          encoding='utf-8', errors='replace', timeout=600, env=_env())


def _exp_id_keyword(path):
    """The ``default=`` expression text of add_argument('--exp-id', ...) in that script."""
    for node in ast.walk(ast.parse(pathlib.Path(path).read_text(encoding='utf-8'))):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'add_argument' and node.args
                and getattr(node.args[0], 'value', None) == '--exp-id'):
            for keyword in node.keywords:
                if keyword.arg == 'default':
                    return ast.dump(keyword.value)
    raise AssertionError(f'no --exp-id argument found in {path}')


def _fixture(tmp_path, exp_id='EXP07-smoke'):
    """Synthetic two-arm run directory; ``exp_id=None`` leaves the manifest as §4.4 never existed."""
    emb_dir, config, out_dir = shipped.TestMainRunEndToEnd()._build(tmp_path)
    path = pathlib.Path(emb_dir) / 'task7_embeddings_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if exp_id is not None:
        manifest['exp_id'] = exp_id
    path.write_text(json.dumps(manifest), encoding='utf-8')
    return emb_dir, config, out_dir, manifest


def _analyse(emb_dir, config, out_dir, extra=()):
    return subprocess.run(
        [sys.executable, ANALYSER, '--embeddings-dir', emb_dir, '--threshold-config', config,
         '--out-dir', out_dir, '--camera-low', '1,2', '--camera-high', '3,4',
         '--bootstrap-resamples', '200', *extra],
        capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=600,
        env=_env())


class TestTheChainDefaultsAgree:
    def test_both_offline_scripts_default_to_the_charter_label(self):
        assert gen.DEFAULT_EXP_ID == 'EXP07'
        assert mod.DEFAULT_EXP_ID == 'EXP07'
        assert gen.DEFAULT_EXP_ID == mod.DEFAULT_EXP_ID

    def test_the_cli_defaults_are_the_constants_not_a_forgotten_literal(self):
        expected = ast.dump(ast.Name(id='DEFAULT_EXP_ID', ctx=ast.Load()))
        assert _exp_id_keyword(GENERATOR) == expected
        assert _exp_id_keyword(ANALYSER) == expected

    def test_the_bringup_config_layer_shares_the_default(self):
        defaults = _load_by_path('reid_launch_defaults', REID_DEFAULTS)
        assert defaults.DEFAULT_EXP_ID == mod.DEFAULT_EXP_ID == gen.DEFAULT_EXP_ID == 'EXP07'

    def test_the_node_declares_the_same_default(self):
        tree = ast.parse(pathlib.Path(NODE).read_text(encoding='utf-8'))
        found = [call.args[1].value for call in ast.walk(tree)
                 if isinstance(call, ast.Call) and getattr(call.func, 'attr', '')
                 == 'declare_parameter' and getattr(call.args[0], 'value', None) == 'exp_id']
        assert found == ['EXP07'], found


class TestResolveExpId:
    def test_matching_ids_agree_and_are_reported(self):
        outcome = mod.resolve_exp_id({'exp_id': 'EXP07'}, 'EXP07')
        assert outcome == {'requested_by_analysis': 'EXP07', 'recorded_by_extraction': 'EXP07',
                           'source': 'task7_embeddings_manifest.json:exp_id',
                           'agreement': 'match', 'action': 'proceed'}

    def test_surrounding_whitespace_is_not_a_second_experiment(self):
        assert mod.resolve_exp_id({'exp_id': '  EXP07 '}, 'EXP07')['agreement'] == 'match'

    def test_a_different_label_is_not_silently_accepted(self):
        with pytest.raises(SystemExit) as excinfo:
            mod.resolve_exp_id({'exp_id': 'EXP07-armB'}, 'EXP07')
        text = str(excinfo.value)
        assert 'exp_id mismatch' in text
        assert "'EXP07-armB'" in text and "'EXP07'" in text
        assert '--allow-exp-id-mismatch' in text

    def test_the_labels_are_distinct_so_case_is_meaningful(self):
        with pytest.raises(SystemExit):
            mod.resolve_exp_id({'exp_id': 'exp07'}, 'EXP07')

    def test_an_override_proceeds_and_records_the_disagreement(self):
        outcome = mod.resolve_exp_id({'exp_id': 'EXP07-armB'}, 'EXP07', allow_mismatch=True)
        assert outcome['agreement'] == 'mismatch'
        assert outcome['action'] == 'proceed_with_override'
        assert 'exp_id mismatch' in outcome['note']

    def test_a_manifest_without_the_key_is_recorded_not_treated_as_agreement(self):
        outcome = mod.resolve_exp_id({'experiment': 'ES-EXP07 Task 7 embeddings'}, 'EXP07')
        assert outcome['agreement'] == 'manifest_has_no_exp_id'
        assert outcome['recorded_by_extraction'] is None
        assert 'predates' in outcome['note']

    def test_a_blank_recorded_id_counts_as_absent(self):
        for raw in ('', '   ', None):
            assert (mod.resolve_exp_id({'exp_id': raw}, 'EXP07')['agreement']
                    == 'manifest_has_no_exp_id'), raw

    def test_an_empty_requested_id_is_refused_before_anything_is_compared(self):
        for raw in ('', '   ', None):
            with pytest.raises(SystemExit) as excinfo:
                mod.resolve_exp_id({'exp_id': 'EXP07'}, raw)
            assert 'cannot be empty' in str(excinfo.value)


class TestGeneratorCommandLine:
    def test_the_flag_is_documented_in_the_generator_help(self):
        done = _cli(GENERATOR, ['--help'])
        assert done.returncode == 0, done.stderr[-400:]
        assert '--exp-id' in done.stdout

    def test_an_empty_id_is_rejected_rather_than_defaulted(self):
        done = _cli(GENERATOR, ['--market1501', 'M', '--transreid-checkpoint', 'C',
                                '--transreid-projection', 'P', '--output-dir', 'O',
                                '--exp-id', ''])
        assert done.returncode == 2, done.stdout + done.stderr
        assert 'cannot be empty' in (done.stdout + done.stderr)

    def test_the_generator_writes_the_id_into_the_manifest_key(self):
        """STATIC: the manifest dict is built inside main(), so this checks the source, not a run.

        The measured version of this claim is the two-image generator smoke recorded in
        ES-EXP07-VERIFICATION-REPORT.md: its manifest carries the requested exp_id.
        """
        source = pathlib.Path(GENERATOR).read_text(encoding='utf-8')
        assert "'exp_id': exp_id," in source
        assert "EXP-METADATA exp_id=" in source


class TestDegenerateSmokePopulation:
    """The smallest legal smoke run has no query with a gallery match — that is not Rank-1 = 0 %.

    Found while measuring the §4.4 handoff on a real two-image extraction: the retrieval table
    formatted a None Wilson interval (crash) and labelled an empty denominator COMPUTED.
    """

    def _npz(self, tmp_path):
        path = str(tmp_path / 'empty_denominator.npz')
        np.savez_compressed(
            path,
            gallery_features=np.eye(2, 16, dtype=np.float32),
            gallery_pids=np.array([1, 2], dtype=np.int32),
            gallery_camids=np.array([1, 2], dtype=np.int32),
            # pid 7 and 8 never appear in the gallery, so every query is invalid by the canonical rule
            query_features=np.eye(2, 16, k=4, dtype=np.float32),
            query_pids=np.array([7, 8], dtype=np.int32),
            query_camids=np.array([1, 2], dtype=np.int32))
        return path

    def _arm(self, tmp_path):
        return mod.analyse_arm(mod.canonical_metric_module(), self._npz(tmp_path), {}, 'arm_a',
                              {'tau': 0.70, 'source_file': 'synthetic.yaml', 'source_key': 'k',
                               'matches_expected_0.70': True}, None, 200)

    def test_zero_valid_queries_is_not_reported_as_a_computed_rate(self, tmp_path):
        retrieval = self._arm(tmp_path)['retrieval']
        assert (retrieval['queries_valid'], retrieval['queries_total']) == (0, 2)
        assert retrieval['label'].startswith('NOT MEASURED'), retrieval['label']
        assert 'no denominator' in retrieval['label']
        assert retrieval['Wilson_95_rank1'] is None
        assert retrieval['mAP_bootstrap_95'] is None

    def test_empty_denominators_become_null_rather_than_nan_in_json(self, tmp_path):
        coverage = self._arm(tmp_path)['retrieval']['identity_coverage']
        assert coverage['mean_genuine_candidates_per_valid_query'] is None
        assert coverage['mean_candidates_per_valid_query'] is None
        text = json.dumps({'identity_coverage': coverage}, default=mod._json_default)
        assert 'NaN' not in text and 'nan' not in text

    def test_the_retrieval_table_renders_the_empty_arm_instead_of_crashing(self, tmp_path):
        results = {'arm_a': self._arm(tmp_path)}
        report = {'exp_id': {'requested_by_analysis': 'EXP07-smoke',
                             'recorded_by_extraction': 'EXP07-smoke', 'agreement': 'match'},
                  'inputs': {'embeddings_dir': 'x'}, 'generated_utc': 't', 'host': 'h',
                  'population_rules': {'population': 'directory'},
                  'threshold': {'tau': 0.70, 'source_file': 'a.yaml', 'source_key': 'k',
                                'matches_expected_0.70': True},
                  'paired_tests': {'tests': []},
                  'osnet_comparison': {'status': 'NOT APPLICABLE', 'reason': 'r'},
                  'camera_partition_disclaimer': 'disclaimer',
                  'not_measured': {'jetson': 'BLOCKED'}}
        text = mod.markdown(report, results)
        assert 'EXP07-smoke' in text
        assert 'NOT MEASURED' in text and 'NaN' not in text


class TestAnalysisHandoff:
    def test_a_matching_id_reaches_the_report_and_carries_the_label(self, tmp_path):
        emb_dir, config, out_dir, manifest = _fixture(tmp_path, 'EXP07-smoke')
        done = _analyse(emb_dir, config, out_dir, ['--exp-id', 'EXP07-smoke'])
        assert done.returncode == 0, done.stdout + done.stderr
        assert 'EXP-METADATA exp_id=EXP07-smoke agrees with the embeddings manifest' in \
            (done.stdout + done.stderr)
        report = json.loads((pathlib.Path(out_dir) / 'task7_statistics.json').read_text(
            encoding='utf-8'))
        assert report['exp_id']['agreement'] == 'match'
        assert report['exp_id']['recorded_by_extraction'] == manifest['exp_id']
        assert report['inputs']['extraction_run']['exp_id'] == 'EXP07-smoke'
        markdown = (pathlib.Path(out_dir) / 'task7_statistics.md').read_text(encoding='utf-8')
        assert 'EXP07-smoke' in markdown

    def test_a_disagreement_aborts_before_a_single_number_is_computed(self, tmp_path):
        emb_dir, config, out_dir, _manifest = _fixture(tmp_path, 'EXP07-armB')
        done = _analyse(emb_dir, config, out_dir, ['--exp-id', 'EXP07'])
        assert done.returncode != 0
        text = done.stdout + done.stderr
        assert 'exp_id mismatch' in text
        assert not (pathlib.Path(out_dir) / 'task7_statistics.json').exists()

    def test_the_override_writes_the_report_and_says_it_was_overridden(self, tmp_path):
        emb_dir, config, out_dir, _manifest = _fixture(tmp_path, 'EXP07-armB')
        done = _analyse(emb_dir, config, out_dir,
                        ['--exp-id', 'EXP07', '--allow-exp-id-mismatch'])
        assert done.returncode == 0, done.stdout + done.stderr
        report = json.loads((pathlib.Path(out_dir) / 'task7_statistics.json').read_text(
            encoding='utf-8'))
        assert report['exp_id']['agreement'] == 'mismatch'
        assert report['exp_id']['action'] == 'proceed_with_override'
        assert 'exp_id mismatch' in report['exp_id']['note']

    def test_a_manifest_predating_the_field_still_analyses_and_says_so(self, tmp_path):
        emb_dir, config, out_dir, _manifest = _fixture(tmp_path, None)
        done = _analyse(emb_dir, config, out_dir)
        assert done.returncode == 0, done.stdout + done.stderr
        assert 'predates ES-EXP07' in (done.stdout + done.stderr)
        report = json.loads((pathlib.Path(out_dir) / 'task7_statistics.json').read_text(
            encoding='utf-8'))
        assert report['exp_id']['agreement'] == 'manifest_has_no_exp_id'
        assert report['exp_id']['recorded_by_extraction'] is None

    def test_an_empty_label_never_reaches_the_metrics(self, tmp_path):
        emb_dir, config, out_dir, _manifest = _fixture(tmp_path, 'EXP07-smoke')
        done = _analyse(emb_dir, config, out_dir, ['--exp-id', ' '])
        assert done.returncode != 0
        assert 'cannot be empty' in (done.stdout + done.stderr)

    def test_the_shipped_main_path_still_works_when_the_flag_is_not_used(self, tmp_path):
        """Default --exp-id EXP07 against a manifest labelled EXP07: unchanged behaviour, exit 0."""
        emb_dir, config, out_dir, _manifest = _fixture(tmp_path, 'EXP07')
        done = _analyse(emb_dir, config, out_dir)
        assert done.returncode == 0, done.stdout + done.stderr
        report = json.loads((pathlib.Path(out_dir) / 'task7_statistics.json').read_text(
            encoding='utf-8'))
        assert report['exp_id']['agreement'] == 'match'
        assert set(report['arms']) == {'transreid_512_agnostic', 'transreid_512_sie'}

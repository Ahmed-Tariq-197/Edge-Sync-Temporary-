"""Focused regression tests for scripts/analyze_task7_statistics.py.

UNIT tests on synthetic populations — charter §9 classification. No real Market-1501 embeddings and
no model inference are involved; what these prove is the arithmetic and the decision rules the Task 7
statistics rest on (Wilson intervals, exact McNemar, Holm, the production similarity semantics, the
canonical-metric equivalence assertion, and the fail-closed integrity/population gates).

The most important test here is TestPerQueryRetrieval.test_matches_canonical_implementation: the
script's per-query pass is what paired inference is built on, while the reported Rank-k/mAP numbers
come from the project's canonical implementation. If those two ever diverge, the script aborts — and
this test is what catches the divergence before a run does.
"""
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
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import evaluate_offline  # noqa: E402  the shipped metric caller that analyse_arm consumes


def load_script():
    path = os.path.join(SCRIPTS, 'analyze_task7_statistics.py')
    spec = importlib.util.spec_from_file_location('analyze_task7_statistics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = load_script()


def _canonical_cmc_map():
    """The canonical metric implementation, reached the same way evaluate_offline reaches it."""
    from run_real_pipeline import compute_cmc_map
    return compute_cmc_map


class TestWilson:
    def test_known_value_for_half_of_twenty(self):
        lo, hi = mod.wilson_interval(10, 20)
        # Published Wilson 95 % interval for 10/20 is [0.2993, 0.7007]; the narrower Wald interval
        # (0.281-0.719) is the one this project must not use for small strata.
        assert lo == pytest.approx(0.29929, abs=1e-5)
        assert hi == pytest.approx(0.70071, abs=1e-5)

    def test_zero_successes_stays_inside_the_unit_interval(self):
        lo, hi = mod.wilson_interval(0, 50)
        assert lo == 0.0
        assert 0.0 < hi < 0.08

    def test_all_successes_stays_inside_the_unit_interval(self):
        lo, hi = mod.wilson_interval(50, 50)
        assert hi == 1.0
        assert 0.92 < lo < 1.0

    def test_empty_denominator_is_none_not_zero(self):
        assert mod.wilson_interval(0, 0) is None


class TestExactMcNemar:
    def test_no_discordant_pairs_is_undefined_not_one(self):
        assert mod.exact_mcnemar(0, 0) is None

    def test_perfectly_symmetric_discordance_gives_p_one(self):
        assert mod.exact_mcnemar(7, 7) == pytest.approx(1.0)

    def test_all_discordance_one_sided_uses_the_exact_binomial_tail(self):
        # b=0, c=5 -> two-sided exact p = 2 * (1/2)^5
        assert mod.exact_mcnemar(0, 5) == pytest.approx(2 * 0.5 ** 5)
        assert mod.exact_mcnemar(5, 0) == pytest.approx(2 * 0.5 ** 5)

    def test_p_is_capped_at_one(self):
        assert mod.exact_mcnemar(1, 2) <= 1.0


class TestHolm:
    def test_multiplies_by_the_remaining_family_size_and_is_monotone(self):
        out = mod.holm_corrections([('H1', 0.001), ('H2', 0.02), ('H3', 0.03)])
        assert out['H1']['adjusted'] == pytest.approx(0.003)
        # 0.02*2 = 0.04 and 0.03*1 = 0.03 would be non-monotone, so the step-down raises it to 0.04
        assert out['H2']['adjusted'] == pytest.approx(0.04)
        assert out['H3']['adjusted'] == pytest.approx(0.04)
        assert out['H1']['decision'].startswith('REJECT')

    def test_undefined_p_values_are_not_counted_in_the_family(self):
        out = mod.holm_corrections([('H1', 0.01), ('H2', None)])
        assert out['H1']['adjusted'] == pytest.approx(0.01)
        assert out['H2']['decision'].startswith('NOT APPLICABLE')

    def test_all_undefined_still_returns_every_id(self):
        out = mod.holm_corrections([('H1', None)])
        assert set(out) == {'H1'}


class TestBootstrap:
    def test_constant_input_has_a_degenerate_interval(self):
        lo, hi = mod.percentile_bootstrap_ci([0.5] * 40, resamples=100)
        assert lo == pytest.approx(0.5) and hi == pytest.approx(0.5)

    def test_single_value_is_not_a_measurement(self):
        assert mod.percentile_bootstrap_ci([0.5], resamples=100) is None

    def test_seed_makes_it_reproducible(self):
        v = list(np.arange(30) / 30.0)
        assert (mod.percentile_bootstrap_ci(v, resamples=200)
                == mod.percentile_bootstrap_ci(v, resamples=200))


class TestProductionCosine:
    def test_matches_the_definition_on_unit_vectors(self):
        q = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        g = np.array([[1.0, 0.0, 0.0], [0.6, 0.8, 0.0]], dtype=np.float32)
        out = mod.production_cosine_matrix(q, g)
        assert out[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert out[0, 1] == pytest.approx(0.6, abs=1e-6)
        assert out[1, 1] == pytest.approx(0.8, abs=1e-6)

    def test_zero_norm_row_yields_zero_similarity_not_nan(self):
        q = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)
        g = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        out = mod.production_cosine_matrix(q, g)
        assert np.all(out[0] == 0.0)
        assert np.isfinite(out).all()

    def test_unnormalised_inputs_are_still_cosines(self):
        q = np.array([[3.0, 4.0]], dtype=np.float32)
        g = np.array([[3.0, 4.0]], dtype=np.float32)
        assert mod.production_cosine_matrix(q, g)[0, 0] == pytest.approx(1.0, abs=1e-6)

    def test_output_stays_in_minus_one_one(self):
        rng = np.random.default_rng(3)
        q, g = rng.normal(size=(5, 8)).astype(np.float32), rng.normal(size=(4, 8)).astype(np.float32)
        out = mod.production_cosine_matrix(q, g)
        assert out.min() >= -1.0 and out.max() <= 1.0


class TestRateAt:
    def test_counts_inclusive_threshold(self):
        scores = np.array([0.1, 0.5, 0.7, 0.7, 0.9])
        assert mod.rate_at(scores, 0.7) == pytest.approx(3 / 5)

    def test_empty_pool_is_none(self):
        assert mod.rate_at(np.array([], dtype=np.float32), 0.5) is None

    def test_extremes(self):
        scores = np.array([0.2, 0.4])
        assert mod.rate_at(scores, -1.0) == 1.0
        assert mod.rate_at(scores, 2.0) == 0.0


class TestThresholdSolvers:
    def test_threshold_is_the_largest_tau_still_meeting_the_target_tar(self):
        genuine = np.sort(np.array([0.9, 0.8, 0.7, 0.6]))
        tau = mod.threshold_at_tar(0.50, genuine)
        # 0.8 admits exactly two of four genuine scores (TAR = 0.50); 0.8+epsilon admits one.
        assert tau == pytest.approx(0.8, abs=1e-6)
        assert mod.rate_at(genuine, tau) == pytest.approx(0.50)

    def test_target_above_the_pool_resolution_uses_every_score(self):
        genuine = np.sort(np.array([0.9, 0.8, 0.7, 0.6]))
        # 90 % of 4 queries needs ceil(3.6) = 4 accepts, so the threshold is the minimum score.
        tau = mod.threshold_at_tar(0.90, genuine)
        assert tau == pytest.approx(0.6, abs=1e-6)
        assert mod.rate_at(genuine, tau) == 1.0

    def test_unreachable_tar_returns_none_instead_of_extrapolating(self):
        genuine = np.sort(np.array([0.1, 0.2]))
        assert mod.threshold_at_tar(1.01, genuine) is None
        assert mod.threshold_at_tar(0.5, np.array([], dtype=np.float64)) is None

    def test_eer_is_the_crossing_point(self):
        genuine = np.sort(np.array([0.9, 0.8, 0.7, 0.6]))
        impostor = np.sort(np.array([0.75, 0.5, 0.3, 0.1]))
        eer = mod.solve_eer(genuine, impostor)
        assert eer is not None
        # at EER the two rates are equal, and each is within [0, 1]
        assert abs(mod.rate_at(impostor, eer['tau']) - (1 - mod.rate_at(genuine, eer['tau']))) < 1e-3
        assert 0.0 <= eer['eer'] <= 1.0

    def test_separable_pools_report_eer_zero_not_a_step_artifact(self):
        # Bisection on a step function lands on a tied score and reports a phantom non-zero EER;
        # the operating-point solver must report perfect separation as 0.0.
        eer = mod.solve_eer(np.sort(np.array([0.9, 0.95])), np.sort(np.array([0.1, 0.2])))
        assert eer['eer'] == 0.0
        assert eer['interpolated'] is False
        assert 0.2 < eer['tau'] <= 0.9   # the first threshold that separates the two pools

    def test_eer_bracket_is_recorded_so_the_number_can_be_recomputed(self):
        genuine = np.sort(np.array([0.9, 0.8, 0.7, 0.6]))
        impostor = np.sort(np.array([0.75, 0.5, 0.3, 0.1]))
        eer = mod.solve_eer(genuine, impostor)
        # At tau = 0.7 both rates are 0.25: an exact empirical operating point, not an interpolation.
        assert eer['interpolated'] is False and eer['eer'] == pytest.approx(0.25)
        assert mod.rate_at(impostor, eer['tau']) == pytest.approx(0.25)
        assert 1 - mod.rate_at(genuine, eer['tau']) == pytest.approx(0.25)
        assert eer['bracket'] == [eer['tau'], eer['tau']]

    def test_crossing_inside_a_step_gap_is_interpolated(self):
        genuine = np.sort(np.array([0.9, 0.8, 0.7]))
        impostor = np.sort(np.array([0.85]))
        eer = mod.solve_eer(genuine, impostor)
        # bracketing operating points: (FMR=1.00, FNMR=0.667) at 0.85 and (FMR=0, FNMR=0.667) at 0.9
        assert eer['interpolated'] is True
        assert 0.85 <= eer['tau'] <= 0.9
        assert 0.0 <= eer['eer'] <= 1.0

    def test_empty_pool_has_no_eer(self):
        assert mod.solve_eer(np.array([], dtype=np.float64),
                             np.sort(np.array([0.1, 0.2]))) is None
        assert mod.solve_eer(np.sort(np.array([0.1, 0.2])),
                             np.array([], dtype=np.float64)) is None

    def test_solver_cost_does_not_grow_a_python_loop_with_the_pool(self):
        """Cost guard: the real impostor pool is 5.3e7 scores.

        The first version of solve_eer looped over every distinct score in Python and never finished
        a real arm; this keeps that from coming back without depending on a wall-clock promise.
        """
        import time

        rng = np.random.default_rng(11)
        pool = np.sort(rng.uniform(0.0, 0.85, 2_000_000).astype(np.float32))
        genuine = np.sort(rng.uniform(0.3, 1.0, 20_000).astype(np.float32))
        started = time.perf_counter()
        eer = mod.solve_eer(genuine, pool)
        elapsed = time.perf_counter() - started
        assert 0.0 <= eer['eer'] <= 1.0
        assert elapsed < 20.0, f'solve_eer took {elapsed:.1f} s on a 2e6 pool'


class TestPerQueryRetrieval:
    def _tiny_population(self):
        # 4 gallery rows: two identities, two cameras; 3 queries.
        g_pids = np.array([1, 1, 2, 0], dtype=np.int32)
        g_cams = np.array([1, 2, 1, 1], dtype=np.int32)
        q_pids = np.array([1, 2, 1], dtype=np.int32)
        q_cams = np.array([1, 1, 2], dtype=np.int32)
        features = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
        q_features = np.array([[1.0, 0.05], [0.05, 1.0], [0.95, 0.05]], dtype=np.float32)
        return q_features, q_pids, q_cams, features, g_pids, g_cams

    def test_matches_canonical_implementation(self):
        args = self._tiny_population()
        per_query, summary = mod.per_query_retrieval(*args)
        cmc, canonical_map = _canonical_cmc_map()(
            args[0].astype(np.float64) / (np.linalg.norm(args[0], axis=1, keepdims=True) + 1e-8),
            args[1], args[2],
            args[3].astype(np.float64) / (np.linalg.norm(args[3], axis=1, keepdims=True) + 1e-8),
            args[4], args[5], max_rank=mod.MAX_RANK)
        assert summary['rank1'] == pytest.approx(float(cmc[0]), abs=1e-12)
        assert summary['rank5'] == pytest.approx(float(cmc[4]), abs=1e-12)
        assert summary['mAP'] == pytest.approx(float(canonical_map), abs=1e-12)

    def test_same_pid_same_camera_duplicate_is_removed_not_counted_as_a_match(self):
        # gallery row 0 (pid 1, cam 1) is the duplicate of query 0 (pid 1, cam 1): the genuine set
        # for query 0 must be row 1 only, so candidate_count excludes exactly one row.
        qf, qp, qc, gf, gp, gc = self._tiny_population()
        per_query, _ = mod.per_query_retrieval(qf, qp, qc, gf, gp, gc)
        assert per_query['candidate_count'][0] == 3
        assert per_query['genuine_count'][0] == 1

    def test_query_without_any_gallery_match_is_invalid_and_excluded(self):
        qf, qp, qc, gf, gp, gc = self._tiny_population()
        gp_bad = np.array([9, 9, 9, 9], dtype=np.int32)          # no identity can match
        _, summary = mod.per_query_retrieval(qf, qp, qc, gf, gp_bad, gc)
        assert summary['queries_valid'] == 0
        assert summary['queries_invalid_no_gallery_match'] == 3
        assert summary['rank1'] == 0.0

    def test_hit_vectors_are_boolean_and_aligned_to_valid_queries(self):
        qf, qp, qc, gf, gp, gc = self._tiny_population()
        per_query, summary = mod.per_query_retrieval(qf, qp, qc, gf, gp, gc)
        valid = per_query['valid']
        assert per_query['hits'][1].dtype == np.bool_
        assert summary['rank1'] == pytest.approx(float(per_query['hits'][1][valid].mean()))
        assert np.isnan(per_query['ap'][~valid]).all()

    def test_rank_band_counts_partition_the_valid_queries_exactly(self):
        qf, qp, qc, gf, gp, gc = self._tiny_population()
        per_query, summary = mod.per_query_retrieval(qf, qp, qc, gf, gp, gc)
        bands = mod.rank_distribution(per_query['first_rank'], per_query['valid'])
        assert sum(bands.values()) == summary['queries_valid']
        assert 'invalid' not in bands


class TestVerificationPools:
    def _pools(self):
        # Gallery: pid/cam (1,1) (1,2) (2,1) (0,1) distractor, (-1,3) junk.
        sim = np.array([[0.9, 0.95, 0.2, 0.1, 0.50],
                        [0.3, 0.4, 0.85, 0.2, 0.45],
                        [0.6, 0.62, 0.61, 0.1, 0.55]], dtype=np.float32)
        q_pids, q_cams = np.array([1, 2, 1]), np.array([1, 1, 2])
        g_pids, g_cams = np.array([1, 1, 2, 0, -1]), np.array([1, 2, 1, 1, 3])
        return sim, q_pids, q_cams, g_pids, g_cams

    def test_duplicate_and_junk_pairs_leave_both_pools(self):
        sim, qp, qc, gp, gc = self._pools()
        pools = mod.verification_pools(sim, qp, qc, gp, gc)
        # q0/g0, q1/g2 and q2/g1 are the same identity in the same camera: the protocol's duplicate
        # exclusion removes them from BOTH pools, so 15 pairs reduce to 2 genuine + 7 impostor.
        assert pools['duplicate_pairs_excluded'] == 3
        assert pools['junk_gallery_rows'] == 1
        assert pools['genuine_scores'].size == 2
        assert pools['impostor_scores'].size == 7
        assert sorted(float(x) for x in pools['genuine_scores']) == pytest.approx([0.6, 0.95])
        junk_score = float(sim[0, 4])
        assert junk_score not in [float(x) for x in pools['impostor_scores']]
        assert pools['impostor_queries_with_no_candidate'] == 0

    def test_distractor_pid_zero_counts_as_an_impostor_not_as_nothing(self):
        sim, qp, qc, gp, gc = self._pools()
        pools = mod.verification_pools(sim, qp, qc, gp, gc)
        assert float(sim[0, 3]) in [float(x) for x in pools['impostor_scores']]

    def test_per_query_maxima_drive_the_paired_units(self):
        sim, qp, qc, gp, gc = self._pools()
        pools = mod.verification_pools(sim, qp, qc, gp, gc)
        assert pools['genuine_max_per_query'][0] == pytest.approx(0.95, abs=1e-6)
        assert pools['impostor_max_per_query'][0] == pytest.approx(0.2, abs=1e-6)
        # query 1 (pid 2, cam 1) has one gallery match and it is its own duplicate -> no genuine pool
        assert list(pools['queries_with_genuine']) == [True, False, True]

    def test_query_with_no_genuine_candidate_is_flagged_undefined(self):
        sim = np.array([[0.4, 0.3]], dtype=np.float32)
        pools = mod.verification_pools(sim, np.array([7]), np.array([1]),
                                       np.array([1, 2]), np.array([1, 2]))
        assert not pools['queries_with_genuine'][0]
        assert np.isneginf(pools['genuine_max_per_query'][0])

    def test_query_whose_only_gallery_row_is_its_own_duplicate_has_no_impostor_pool(self):
        sim = np.array([[0.9]], dtype=np.float32)
        pools = mod.verification_pools(sim, np.array([1]), np.array([1]),
                                       np.array([1]), np.array([1]))
        # The count is a plain int, not an array: analyse_arm consumes it directly.
        assert pools['impostor_queries_with_no_candidate'] == 1
        assert pools['queries_with_impostor'][0] == np.False_
        assert np.isneginf(pools['impostor_max_per_query'][0])

    def test_at_tau_uses_accept_inclusive_rule(self):
        sim, qp, qc, gp, gc = self._pools()
        pools = mod.verification_pools(sim, qp, qc, gp, gc)
        # The genuine pool is [0.6, 0.95] stored in float32, exactly as the production verifier
        # promotes its similarity. Compared against the float64 literal 0.95 the stored score is
        # 0.949999988, so >= excludes it; against the promoted value it is accepted. Both branches
        # are production-faithful, and the boundary case is the one that proves >= rather than >.
        boundary = float(pools['genuine_scores'].max())
        out = mod.verification_at_tau(pools, boundary)
        assert out['TAR'] == pytest.approx(1 / 2)
        assert out['FNMR'] == pytest.approx(1 / 2)
        assert out['FMR'] == 0.0
        assert out['genuine_pairs'] == 2 and out['impostor_pairs'] == 7
        assert 'accept' in out['acceptance_rule'] and '>=' in out['acceptance_rule']
        assert mod.verification_at_tau(pools, 0.95)['TAR'] == 0.0

    def test_at_tau_fmr_counts_distractor_accepts_against_every_impostor_pair(self):
        sim, qp, qc, gp, gc = self._pools()
        pools = mod.verification_pools(sim, qp, qc, gp, gc)
        out = mod.verification_at_tau(pools, 0.60)
        # impostor pool = [0.1, 0.1, 0.2, 0.2, 0.3, 0.4, 0.61]; only 0.61 clears 0.60.
        assert out['FMR'] == pytest.approx(1 / 7)
        assert out['Wilson_95_FMR'][0] < out['FMR'] < out['Wilson_95_FMR'][1]
        assert out['EER']['eer'] >= 0.0

    def test_at_tau_survives_an_empty_genuine_pool(self):
        sim = np.array([[0.4, 0.3]], dtype=np.float32)
        pools = mod.verification_pools(sim, np.array([7]), np.array([1]),
                                       np.array([1, 2]), np.array([1, 2]))
        out = mod.verification_at_tau(pools, 0.70)
        assert out['TAR'] is None and out['FNMR'] is None
        assert out['genuine_pairs'] == 0
        assert out['EER'].startswith('NOT MEASURED')


class TestPopulationGate:
    def _data(self, junk=0, zero_rows=0, dim=512):
        g = np.zeros((4, dim), dtype=np.float32)
        g[:, 0] = 1.0
        if zero_rows:
            g[0] = 0.0
        pids = np.array([1, 1, 2, 0], dtype=np.int32)
        if junk:
            pids = np.array([1, 1, 2, -1], dtype=np.int32)
        return {'gallery_features': g, 'gallery_pids': pids,
                'gallery_camids': np.array([1, 2, 1, 1], dtype=np.int32),
                'query_features': g[:2].copy(), 'query_pids': np.array([1, 2], dtype=np.int32),
                'query_camids': np.array([1, 1], dtype=np.int32)}

    def test_clean_population_reports_its_measured_counts(self):
        report, bad, fatal = mod.verify_population(self._data(), {}, 'arm')
        assert fatal == []
        assert report['gallery']['rows'] == 4
        assert report['gallery']['distractor_pid_0'] == 1
        assert report['query']['identities_unique'] == 2
        assert report['gallery']['unit_norm_max_deviation'] == pytest.approx(0.0, abs=1e-6)

    def test_768_arm_is_recorded_as_raw_features_not_a_norm_failure(self):
        report, _, fatal = mod.verify_population(self._data(dim=768), {}, 'arm')
        assert fatal == []
        assert 'unit_norm_max_deviation' not in report['gallery']
        assert '768-D' in report['gallery']['norm_note']

    def test_zero_norm_rows_are_fatal_because_a_sentinel_would_be_scored(self):
        _, _, fatal = mod.verify_population(self._data(zero_rows=1), {}, 'arm')
        assert any('zero-norm' in f for f in fatal)

    def test_junk_rows_are_fatal(self):
        _, _, fatal = mod.verify_population(self._data(junk=1), {}, 'arm')
        assert any('pid -1' in f for f in fatal)

    def test_manifest_disagreement_is_reported_not_silently_ignored(self):
        manifest = {'population': {'gallery': {'kept': 999, 'identities': 999,
                                               'distractors_pid_0': 999},
                                   'query': {'kept': 2, 'identities': 2,
                                             'distractors_pid_0': 0}}}
        _, bad, _ = mod.verify_population(self._data(), manifest, 'arm')
        assert 'gallery.rows' in bad and 'gallery.identities' in bad

    def test_a_key_the_manifest_does_not_claim_is_not_reported_as_a_disagreement(self):
        # The shipped manifest's query split has no distractor count (its rule is pid > 0), which the
        # real run hit: absence of a claim is None, not False, or the gate refuses a valid file.
        manifest = {'population': {'gallery': {'kept': 4, 'identities': 3,
                                               'distractors_pid_0': 1,
                                               'files_scanned': 5,
                                               'excluded': {'junk_identity_-1': 1},
                                               'identity_number_range': [0, 2]},
                                   'query': {'kept': 2, 'identities': 2,
                                             'identity_number_range': [1, 2]}}}
        report, bad, fatal = mod.verify_population(self._data(), manifest, 'arm')
        assert bad == [] and fatal == []
        assert report['query']['manifest_agreement']['distractors_pid_0'] is None
        assert report['query']['manifest_agreement']['rows'] is True
        assert report['gallery']['manifest_agreement']['scan_accounting'] is True

    def test_scan_accounting_that_does_not_add_up_is_reported(self):
        manifest = {'population': {'gallery': {'kept': 4, 'files_scanned': 9,
                                               'excluded': {'junk_identity_-1': 1}}}}
        report, bad, _ = mod.verify_population(self._data(), manifest, 'arm')
        assert 'gallery.scan_accounting' in bad
        assert report['gallery']['manifest_agreement']['scan_accounting'] is False


class TestThresholdProvenance:
    def _write(self, tmp_path, value):
        path = tmp_path / 'algorithm_thresholds.yaml'
        path.write_text(f'algorithm_thresholds:\n  ros__parameters:\n'
                        f'    reid_similarity_threshold: {value}\n', encoding='utf-8')
        return str(path)

    def test_reads_the_shipped_key(self, tmp_path):
        out = mod.read_threshold(self._write(tmp_path, 0.70))
        assert out['tau'] == 0.70
        assert out['matches_expected_0.70'] is True

    def test_a_retailored_threshold_is_flagged_not_silently_used(self, tmp_path):
        out = mod.read_threshold(self._write(tmp_path, 0.65))
        assert out['matches_expected_0.70'] is False

    def test_missing_key_is_none_and_not_a_pass(self, tmp_path):
        path = tmp_path / 'empty.yaml'
        path.write_text('algorithm_thresholds:\n  ros__parameters: {}\n', encoding='utf-8')
        out = mod.read_threshold(str(path))
        assert out['tau'] is None and out['matches_expected_0.70'] is False


class TestPairedTests:
    def _result(self, rank1, genuine_max, impostor_max, tau=0.7, impostor_eligible=None):
        valid = np.array([True, True, True, True])
        return {'verification': {'threshold': {'tau': tau}},
                '_arrays': {'valid': valid,
                            'rank1': np.asarray(rank1, dtype=bool),
                            'genuine_max': np.asarray(genuine_max, dtype=np.float64),
                            'impostor_max': np.asarray(impostor_max, dtype=np.float64),
                            'genuine_eligible': valid,
                            'impostor_eligible': (valid if impostor_eligible is None
                                                  else np.asarray(impostor_eligible, dtype=bool)),
                            'ap': np.zeros(4)}}

    def test_discordant_counts_and_exact_p(self, monkeypatch):
        a = self._result([True, True, False, False], [0.9, 0.9, 0.9, 0.1],
                         [0.1, 0.1, 0.9, 0.1])
        b = self._result([True, False, False, False], [0.9, 0.1, 0.9, 0.1],
                         [0.1, 0.1, 0.1, 0.1])
        # hide the 512 names so the declared family matches these synthetic keys
        out = mod.paired_tests({'transreid_512_agnostic': a, 'transreid_512_sie': b})
        h1 = next(t for t in out['tests'] if t['id'] == 'H1')
        assert h1['discordant_A_only'] == 1 and h1['discordant_B_only'] == 0
        assert h1['p_value'] == pytest.approx(2 * 0.5 ** 1)
        assert out['family_size'] == len([t for t in out['tests']
                                          if t.get('p_value') is not None])

    def test_no_discordance_is_not_applied_instead_of_p_one(self):
        same = self._result([True, False, True, False], [0.9] * 4, [0.1] * 4)
        out = mod.paired_tests({'transreid_512_agnostic': same, 'transreid_512_sie':
                                self._result([True, False, True, False], [0.9] * 4, [0.1] * 4)})
        h1 = next(t for t in out['tests'] if t['id'] == 'H1')
        assert h1['status'].startswith('NOT APPLICABLE')

    def test_missing_arm_is_not_measured(self):
        a = self._result([True] * 4, [0.9] * 4, [0.1] * 4)
        out = mod.paired_tests({'transreid_512_agnostic': a})
        assert all(t['status'].startswith('NOT MEASURED') for t in out['tests'])

    def test_mismatched_paired_units_are_rejected(self):
        a = self._result([True] * 4, [0.9] * 4, [0.1] * 4)
        b = self._result([True] * 4, [0.9] * 4, [0.1] * 4)
        b['_arrays']['valid'] = np.array([True, True, False, False])
        out = mod.paired_tests({'transreid_512_agnostic': a, 'transreid_512_sie': b})
        h1 = next(t for t in out['tests'] if t['id'] == 'H1')
        assert h1['status'].startswith('NOT APPLICABLE')


class TestNoSilentZeroInjection:
    def test_distractor_rejection_returns_not_applicable_when_there_are_no_distractors(self):
        class _EO:
            @staticmethod
            def compute_distractor_rejection(qf, gf, gp, threshold):
                return None, 0, 0

        data = {'query_features': np.zeros((1, 2), np.float32),
                'gallery_features': np.zeros((1, 2), np.float32),
                'gallery_pids': np.array([1], dtype=np.int32)}
        out = mod._distractor_rejection(_EO(), data, 0.70)
        assert out['status'] == 'NOT APPLICABLE'

    def test_real_rate_carries_its_own_wilson_interval(self):
        class _EO:
            @staticmethod
            def compute_distractor_rejection(qf, gf, gp, threshold):
                return 0.5, 5, 10

        data = {'query_features': np.zeros((1, 2), np.float32),
                'gallery_features': np.zeros((1, 2), np.float32),
                'gallery_pids': np.array([1], dtype=np.int32)}
        out = mod._distractor_rejection(_EO(), data, 0.70)
        assert out['rejected'] == 5 and out['distractors'] == 10
        assert out['Wilson_95_rejection'][0] < 0.5 < out['Wilson_95_rejection'][1]


class TestArmEndToEnd:
    """analyse_arm on a hand-checked population.

    The production run of this script found two defects that no helper-level test could see
    (an int called with .sum(), and a manifest claim read as a disagreement). This class runs the
    whole per-arm path on a population whose every number is derived by hand below.
    """

    DIM = 16

    @classmethod
    def _vec(cls, **index_value):
        """A unit vector from {axis_index: value} keyword pairs."""
        v = np.zeros(cls.DIM, dtype=np.float32)
        for key, value in index_value.items():
            v[int(key[1:])] = value
        return v

    def _features(self, gallery_g1=None):
        v = self._vec
        gallery = np.stack([v(x0=1.0),                                  # g0 pid1 cam1
                            gallery_g1 if gallery_g1 is not None
                            else v(x0=1.0),                             # g1 pid1 cam2
                            v(x1=1.0),                                  # g2 pid2 cam1
                            v(x0=0.75, x2=0.661435),                    # g3 pid0 cam3 distractor
                            v(x1=1.0)])                                 # g4 pid2 cam4
        query = np.stack([v(x0=0.8, x8=0.6),                            # q0 pid1 cam1
                          v(x1=1.0),                                    # q1 pid2 cam2
                          v(x11=1.0),                                   # q2 pid9 cam3, no match
                          v(x0=0.95, x9=0.31225)])                      # q3 pid1 cam4
        return {'gallery_features': gallery,
                'gallery_pids': np.array([1, 1, 2, 0, 2], dtype=np.int32),
                'gallery_camids': np.array([1, 2, 1, 3, 4], dtype=np.int32),
                'query_features': query,
                'query_pids': np.array([1, 2, 9, 1], dtype=np.int32),
                'query_camids': np.array([1, 2, 3, 4], dtype=np.int32)}

    def _arm(self, tmp_path, name, **kwargs):
        path = str(tmp_path / f'{name}.npz')
        np.savez_compressed(path, **self._features(**kwargs))
        return path

    def _tau(self):
        return {'tau': 0.70, 'source_file': 'synthetic.yaml', 'source_key': 'k',
                'matches_expected_0.70': True}

    def test_retrieval_pools_and_decisions_match_the_hand_computation(self, tmp_path):
        out = mod.analyse_arm(evaluate_offline, self._arm(tmp_path, 'arm_a'), {}, 'arm_a',
                              self._tau(), None, 200)
        r = out['retrieval']
        assert (r['queries_total'], r['queries_valid']) == (4, 3)   # q2 has no gallery identity
        assert r['rank1_pct'] == pytest.approx(100.0)
        assert r['canonical_vs_per_query_max_abs_difference'] <= 1e-12

        pool = out['verification']['pair_pool']
        # 4 queries x 5 gallery rows = 20 pairs = 5 genuine + 14 impostor + 1 same-pid/same-camera
        assert (pool['genuine_pairs'], pool['impostor_pairs']) == (5, 14)
        assert pool['duplicate_pairs_excluded'] == 1
        assert pool['junk_gallery_rows'] == 0
        assert pool['queries_without_any_impostor_candidate'] == 0

        v = out['verification']['at_tau']
        assert v['TAR'] == pytest.approx(1.0) and v['FNMR'] == 0.0
        assert v['FMR'] == pytest.approx(1 / 14)      # only the distractor g3 clears tau, for q3
        # per-query denominators exclude q2 (invalid) and count only queries with a candidate
        decisions = out['verification']['per_query_decisions']
        assert decisions['genuine_accept_n'] == 3
        assert decisions['false_accept_n'] == 3
        assert decisions['false_accept_rate'] == pytest.approx(1 / 3)

    def test_a_score_at_the_float32_boundary_below_tau_is_rejected(self, tmp_path):
        data = self._features()
        data['query_features'][3] = self._vec(x0=0.7, x9=0.714143)
        path = str(tmp_path / 'boundary.npz')
        np.savez_compressed(path, **data)
        out = mod.analyse_arm(evaluate_offline, path, {}, 'boundary', self._tau(), None, 200)
        # float32(0.7) promotes to 0.699999988, which is < the float64 tau the verifier compares
        # against, so the production rule rejects it. Same rule, same direction, as identity_verifier.
        # q3 has TWO genuine rows (g0 and g1) and both sit on that boundary, so 3 of 5 are accepted.
        assert out['verification']['at_tau']['TAR'] == pytest.approx(3 / 5)

    def test_paired_outcomes_and_markdown_render_from_two_real_arm_results(self, tmp_path):
        arm_a = mod.analyse_arm(evaluate_offline, self._arm(tmp_path, 'transreid_512_agnostic'),
                               {}, 'transreid_512_agnostic', self._tau(), ([1, 2], [3, 4]), 200)
        arm_b = mod.analyse_arm(
            evaluate_offline,
            self._arm(tmp_path, 'transreid_512_sie',
                      gallery_g1=self._vec(x0=0.6, x2=0.8)),
            {}, 'transreid_512_sie', self._tau(), ([1, 2], [3, 4]), 200)
        # Arm B loses q0's only surviving genuine candidate (0.48) and its rank-1 with it.
        assert arm_b['retrieval']['rank1_pct'] == pytest.approx(200 / 3)
        # q0g1 (0.48) and q3g1 (0.57) both drop below tau, so 3 of 5 genuine pairs are accepted.
        assert arm_b['verification']['at_tau']['FNMR'] == pytest.approx(2 / 5)

        results = {'transreid_512_agnostic': arm_a, 'transreid_512_sie': arm_b}
        paired = mod.paired_tests(results)
        h1 = next(t for t in paired['tests'] if t['id'] == 'H1')
        assert h1['status'] == 'COMPUTED'
        assert (h1['discordant_A_only'], h1['discordant_B_only']) == (1, 0)
        assert h1['paired_units'] == 3
        report = {'inputs': {'embeddings_dir': 'x'}, 'generated_utc': 't', 'host': 'h',
                  'population_rules': {'population': 'directory'},
                  'threshold': self._tau(),
                  'paired_tests': paired,
                  'osnet_comparison': {'status': 'NOT APPLICABLE', 'reason': 'r'},
                  'camera_partition_disclaimer': 'disclaimer',
                  'not_measured': {'jetson': 'BLOCKED'}}
        text = mod.markdown(report, results)
        assert 'cams_' in text and 'NOT APPLICABLE' in text
        assert 'None' not in text.split('## What is not measured here')[0]


class TestMainRunEndToEnd:
    """main() itself: integrity gate, τ gate, outputs. Found the manifest-claim defect."""

    def _build(self, tmp_path):
        builder = TestArmEndToEnd()
        emb_dir = tmp_path / 'embeddings'
        emb_dir.mkdir()
        files = {}
        for name in ('transreid_512_agnostic', 'transreid_512_sie'):
            path = emb_dir / f'{name}.npz'
            np.savez_compressed(path, **builder._features(
                gallery_g1=None if 'agnostic' in name else builder._vec(x0=0.6, x2=0.8)))
            files[name] = {'path': str(path), 'bytes': int(path.stat().st_size),
                           'sha256': mod.sha256_of(str(path))}
        manifest = {'experiment': 'synthetic', 'generated_utc': 't', 'elapsed_s': 0,
                    'host': 'h', 'assets': {}, 'files': files,
                    'population': {
                        'population': 'directory',
                        'gallery': {'kept': 5, 'files_scanned': 5, 'excluded': {},
                                    'identities': 3, 'distractors_pid_0': 1,
                                    'identity_number_range': [0, 2]},
                        'query': {'kept': 4, 'files_scanned': 4, 'excluded': {},
                                  'identities': 3, 'identity_number_range': [1, 9]}}}
        manifest_path = emb_dir / 'task7_embeddings_manifest.json'
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
        config = tmp_path / 'algorithm_thresholds.yaml'
        config.write_text('algorithm_thresholds:\n  ros__parameters:\n'
                          '    reid_similarity_threshold: 0.70\n', encoding='utf-8')
        return str(emb_dir), str(config), str(tmp_path / 'out')

    def _run(self, emb_dir, config, out_dir):
        return subprocess.run(
            [sys.executable, os.path.join(SCRIPTS, 'analyze_task7_statistics.py'),
             '--embeddings-dir', emb_dir, '--threshold-config', config,
             '--out-dir', out_dir, '--camera-low', '1,2', '--camera-high', '3,4',
             '--bootstrap-resamples', '200'],
            capture_output=True, text=True)

    def test_main_writes_all_three_outputs_and_labels_the_paired_test(self, tmp_path):
        emb_dir, config, out_dir = self._build(tmp_path)
        proc = self._run(emb_dir, config, out_dir)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        report = json.loads((pathlib.Path(out_dir) / 'task7_statistics.json').read_text(
            encoding='utf-8'))
        assert set(report['arms']) == {'transreid_512_agnostic', 'transreid_512_sie'}
        assert report['threshold']['tau'] == 0.70
        assert all(v['match'] for v in report['inputs']['integrity'].values())
        h1 = next(t for t in report['paired_tests']['tests'] if t['id'] == 'H1')
        assert h1['status'] == 'COMPUTED'
        assert (pathlib.Path(out_dir) / 'task7_statistics.md').stat().st_size > 0
        assert (pathlib.Path(out_dir) / 'task7_population.json').stat().st_size > 0

    def test_a_tampered_embedding_file_fails_the_integrity_gate(self, tmp_path):
        emb_dir, config, out_dir = self._build(tmp_path)
        target = pathlib.Path(emb_dir) / 'transreid_512_sie.npz'
        target.write_bytes(target.read_bytes() + b'tamper')
        proc = self._run(emb_dir, config, out_dir)
        assert proc.returncode != 0
        assert 'integrity gate FAILED' in (proc.stdout + proc.stderr)

    def test_a_config_that_does_not_carry_the_expected_threshold_is_refused(self, tmp_path):
        emb_dir, config, out_dir = self._build(tmp_path)
        pathlib.Path(config).write_text(
            'algorithm_thresholds:\n  ros__parameters:\n'
            '    reid_similarity_threshold: 0.62\n', encoding='utf-8')
        proc = self._run(emb_dir, config, out_dir)
        assert proc.returncode != 0
        assert 'not 0.70' in (proc.stdout + proc.stderr)


class TestReportingHelpers:
    def test_json_default_keeps_numpy_scalars_numeric(self):
        payload = {'a': np.float32(0.5), 'b': np.int64(3), 'c': np.bool_(True),
                   'd': np.array([1.0, 2.0])}
        restored = json.loads(json.dumps(payload, default=mod._json_default))
        assert restored == {'a': 0.5, 'b': 3, 'c': True, 'd': [1.0, 2.0]}

    def test_pct_renders_none_as_not_measured(self):
        assert mod._pct(None) == 'NOT MEASURED'
        assert mod._pct(0.25) == '25.00 %'

    def test_markdown_contains_no_placeholders_and_cites_every_arm(self):
        report = {
            'inputs': {'embeddings_dir': 'x'}, 'generated_utc': 't', 'host': 'h',
            'population_rules': {'population': 'directory'},
            'threshold': {'tau': 0.7, 'source_file': 'a.yaml', 'source_key': 'k',
                          'matches_expected_0.70': True},
            'paired_tests': {'tests': [{'id': 'H1', 'metric': 'rank1_hit', 'status': 'COMPUTED',
                                        'arms': ['a', 'b'], 'rate_A': 0.5, 'rate_B': 0.4,
                                        'difference_A_minus_B': 0.1, 'discordant_A_only': 3,
                                        'discordant_B_only': 1, 'paired_units': 10,
                                        'p_value': 0.5,
                                        'holm_adjusted': {'adjusted': 0.5,
                                                          'decision': 'FAIL TO REJECT'}}]},
            'osnet_comparison': {'status': 'NOT APPLICABLE', 'reason': 'r'},
            'camera_partition_disclaimer': 'disclaimer',
            'not_measured': {'jetson': 'BLOCKED'}}
        res = {'arm_x': {
            'feature_dim': 512,
            'retrieval': {'queries_valid': 2, 'queries_total': 2, 'rank1_pct': 50.0,
                          'rank5_pct': 50.0, 'rank10_pct': 50.0, 'mAP_pct': 40.0,
                          'Wilson_95_rank1': [0.1, 0.9], 'mAP_bootstrap_95': [0.1, 0.7],
                          'label': 'COMPUTED'},
            'verification': {'threshold': {'tau': 0.7},
                             'at_tau': {'genuine_pairs': 3, 'impostor_pairs': 5,
                                        'FNMR': 0.25, 'FMR': 0.2,
                                        'Wilson_95_FNMR': [0.05, 0.6],
                                        'Wilson_95_FMR': [0.02, 0.5],
                                        'EER': {'eer': 0.3, 'tau': 0.5},
                                        'FMR_at_TAR_0.90': 0.4},
                             'per_query_decisions': {'genuine_accept_rate': 0.75,
                                                     'false_accept_rate': 0.2}},
            'camera_partition_proxy': {'status': 'PROXY', 'cams_[1]_to_[2]': {
                'queries': 2, 'queries_valid': 2, 'gallery_rows': 3, 'rank1': 0.5,
                'rank10': 0.5, 'mAP': 0.4, 'Wilson_95_rank1': [0.1, 0.9],
                'verification_at_tau': {'FNMR': 0.1, 'FMR': 0.2}}}}}
        text = mod.markdown(report, res)
        assert 'Rank-1' in text and 'arm_x' in text and 'NOT APPLICABLE' in text
        assert 'None' not in text.split('## What is not measured here')[0]

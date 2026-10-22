#!/usr/bin/env python3
"""TR-08 / Task 7 statistics — retrieval, verification and paired inference from real embeddings.

Charter §8 is the specification, and every quantity this script prints is either measured from the
embedding files it is given or explicitly labelled NOT MEASURED / NOT APPLICABLE. It never invents a
number and never estimates one from a previous run.

What it computes, and from what:

  1. **Integrity gate** — the ``.npz`` files are re-hashed and compared with the digests recorded in
     the embeddings manifest. A mismatch aborts, because then the file being analysed is not the one
     that was produced by the audited extraction run.
  2. **Population verification** (§8: "do not assume standard dataset counts") — gallery/query sizes,
     identity and camera sets, junk and distractor counts, unit-norm status and zero-row status are
     read out of the arrays themselves and cross-checked against the manifest. Agreement is reported
     as measured, disagreement is fatal.
  3. **Retrieval** — Rank-1/5/10 and mAP per arm, using the project's single canonical CMC/mAP
     implementation (``run_real_pipeline.compute_cmc_map``, the copy validated against torchreid
     0.2.5), reached through ``evaluate_offline.compute_cmc_map``. Alongside it, a per-query pass
     produces the rank-hit and average-precision vectors needed for paired inference; the script
     asserts that this pass reproduces the canonical aggregates to 1e-12 and aborts otherwise, so
     there cannot be two divergent metric implementations in the results (§6, TR-07's lesson).
  4. **Intervals** — Wilson 95 % score intervals for every per-query proportion (Rank-k, genuine
     acceptance, false acceptance). mAP is not a binomial proportion, so it gets a documented
     percentile bootstrap (fixed seed, resamples reported) instead of a mislabelled Wilson interval.
  5. **Verification scores** — genuine and impostor cosine-similarity pools, computed with the
     *production* similarity rule (``IdentityVerifier.compute_cosine_similarity`` semantics,
     cross-checked against that function on sampled pairs) and the *production* decision rule
     (accept iff ``sim >= τ``), with τ read from the shipped ``algorithm_thresholds.yaml`` and
     asserted to be 0.70. FNMR/FMR at τ, EER and FMR at matched TAR = 0.90. The spatial gate cannot
     be evaluated offline (no trajectory/covariance data exists), so these are the **appearance gate
     in isolation**, which is stated in the output rather than hidden.
  6. **Paired inference** — exact McNemar on the paired per-query outcomes of the two deployed 512-D
     arms, with Holm-Boferoni correction across the pre-declared family. The family is declared
     before any p-value is computed (§8: no post-hoc comparison shopping). The OSNet arm gets
     ``NOT APPLICABLE`` because no OSNet embeddings exist on this host — charter §3B forbids creating
     them by altering OSNet, and §8 forbids a paired test without paired outcomes.
  7. **Camera-partition breakdown** — queries from cameras 1-3 matched against a gallery restricted
     to cameras 4-6 and vice versa. This is reported as a **labelled proxy for cross-view matching**,
     never as handoff success or false-handoff rate (§8 explicitly forbids that inference, and no
     robot/handoff data exists in this project).

Justified breakdowns only: query camera (needed for the proxy and because SIE is camera-indexed) and
arm (the experimental factor). No breakdown is computed that the population cannot support.

Usage:
    python scripts/analyze_task7_statistics.py \\
        --embeddings-dir C:/Users/PC/edgesync_es07_runs/embeddings_full_directory \\
        --threshold-config ../edgesync_bringup/config/algorithm_thresholds.yaml \\
        --out-dir C:/Users/PC/edgesync_es07_runs/statistics

Exit status is non-zero if an integrity, population or metric-equivalence check fails.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent))

MAX_RANK = 50
LABELS = ('COMPUTED', 'REPRODUCED', 'BLOCKED', 'NOT MEASURED', 'NOT APPLICABLE')

#: Pre-declared hypothesis family (§8). Decided before any p-value exists.
HYPOTHESIS_FAMILY = [
    {'id': 'H1', 'metric': 'rank1_hit', 'unit': 'query',
     'arms': ('transreid_512_agnostic', 'transreid_512_sie'),
     'h0': 'the two 512-D arms have the same Rank-1 hit rate'},
    {'id': 'H2', 'metric': 'genuine_accept_at_tau', 'unit': 'query',
     'arms': ('transreid_512_agnostic', 'transreid_512_sie'),
     'h0': 'the two arms accept the same fraction of queries with at least one genuine match at tau'},
    {'id': 'H3', 'metric': 'false_accept_at_tau', 'unit': 'query',
     'arms': ('transreid_512_agnostic', 'transreid_512_sie'),
     'h0': 'the two arms falsely accept the same fraction of queries at tau'},
]

DEPLOYED_512_ARMS = ('transreid_512_agnostic', 'transreid_512_sie')

#: ES-EXP07 §4.4 — must equal generate_task7_embeddings.py's default; pinned by a unit test.
DEFAULT_EXP_ID = 'EXP07'


# ─────────────────────────────── basic helpers ───────────────────────────────

def sha256_of(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def wilson_interval(successes, trials, z=1.959963984540054):
    """Wilson score interval for a binomial proportion (95 % by default).

    Chosen over the normal (Wald) interval because Rank-1 proportions near 0 or 1 and small strata
    (a single query camera) make Wald intervals miss the boundary or leave [0, 1] entirely.
    """
    if trials <= 0:
        return None
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    margin = (z / denom) * math.sqrt(max(p * (1 - p) / trials, 0.0) + z * z / (4 * trials * trials))
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def exact_mcnemar(b, c):
    """Two-sided exact McNemar on the discordant counts b and c.

    Exact (binomial, p=1/2 under H0) rather than chi-square: the discordant count can be small in a
    stratum, and the chi-square approximation is the wrong tool there. Returns None when there are
    no discordant pairs — the statistic is undefined, which is reported as such rather than as p=1.
    """
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def holm_corrections(pairs):
    """Holm-Boferoni step-down adjustment over a declared family of p-values.

    Input [(id, p)] with None p-values dropped (and recorded); output [(id, p, adjusted, reject_at_
    0.05)] in the original order.
    """
    usable = [(pid, p) for pid, p in pairs if p is not None]
    if not usable:
        return {pid: {'p_value': None, 'adjusted': None,
                      'decision': 'NOT APPLICABLE (no discordant pairs)'} for pid, _ in pairs}
    ordered = sorted(usable, key=lambda kv: kv[1])
    m = len(ordered)
    adjusted = {}
    running_max = 0.0
    for rank, (pid, p) in enumerate(ordered):
        candidate = min(1.0, (m - rank) * p)
        running_max = max(running_max, candidate)  # enforce monotonicity of the step-down
        adjusted[pid] = running_max
    out = {}
    for pid, p in pairs:
        if p is None:
            out[pid] = {'p_value': None, 'adjusted': None,
                        'decision': 'NOT APPLICABLE (no discordant pairs)'}
        else:
            out[pid] = {'p_value': p, 'adjusted': adjusted[pid],
                        'decision': ('REJECT H0 at alpha=0.05' if adjusted[pid] < 0.05
                                     else 'FAIL TO REJECT H0 at alpha=0.05')}
    return out


def percentile_bootstrap_ci(values, resamples=2000, seed=20261019, alpha=0.05):
    """Percentile bootstrap for a mean of per-query values (mAP). Method and seed recorded."""
    v = np.asarray(values, dtype=np.float64)
    if v.size < 2:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(resamples, v.size))
    means = v[idx].mean(axis=1)
    return [float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))]


# ───────────────────────── similarity and score pools ─────────────────────────

def production_cosine_matrix(query_features, gallery_features):
    """Cosine similarity with the production verifier's semantics, in float32.

    Mirrors IdentityVerifier.compute_cosine_similarity (core/identity_verifier.py:37-51): a row (or
    column) whose norm is below 1e-6 yields similarity 0.0, and the result is clipped to [-1, 1].
    float32 matches the verifier's own dtype for the dot product; the retrieval path keeps float64
    because the canonical metric implementation is written in float64.
    """
    q = np.asarray(query_features, dtype=np.float32)
    g = np.asarray(gallery_features, dtype=np.float32)
    qn = np.linalg.norm(q, axis=1, keepdims=True).astype(np.float32)
    gn = np.linalg.norm(g, axis=1, keepdims=True).astype(np.float32)
    q_ok = (qn.reshape(-1) >= 1e-6)
    g_ok = (gn.reshape(-1) >= 1e-6)
    safe_q = np.where(q_ok[:, None], q / np.maximum(qn, 1e-12), 0.0)
    safe_g = np.where(g_ok[:, None], g / np.maximum(gn, 1e-12), 0.0)
    sim = np.clip(safe_q @ safe_g.T, -1.0, 1.0).astype(np.float32)
    # A degenerate vector on either side must read 0.0, exactly as the verifier returns 0.0.
    sim[~q_ok, :] = 0.0
    sim[:, ~g_ok] = 0.0
    return sim


def cross_check_against_verifier(query_features, gallery_features, q_pids, g_pids, n_pairs=200):
    """Measure the matrix path against the scalar production function on sampled pairs."""
    from edgesync_identity.core.identity_verifier import IdentityVerifier

    rng = np.random.default_rng(11)
    gi_by_pid = collections.defaultdict(list)
    for i, pid in enumerate(g_pids):
        gi_by_pid[int(pid)].append(i)
    eligible = [qi for qi, pid in enumerate(q_pids) if gi_by_pid.get(int(pid))]
    if not eligible:
        return {'status': 'NOT MEASURED (no query with a gallery identity match to sample)'}
    max_abs = 0.0
    checked = 0
    for _ in range(n_pairs):
        qi = int(rng.choice(eligible))
        gi = int(rng.choice(gi_by_pid[int(q_pids[qi])]))
        ref = float(IdentityVerifier.compute_cosine_similarity(
            query_features[qi], gallery_features[gi]))
        max_abs = max(max_abs, abs(ref - float(production_cosine_matrix(
            query_features[qi:qi + 1], gallery_features[gi:gi + 1])[0, 0])))
        checked += 1
    return {'status': 'COMPUTED', 'pairs_checked': checked,
            'max_abs_difference_vs_production_function': max_abs,
            'sampling': 'queries paired with one of their own gallery matches, seed 11'}


def rate_at(sorted_scores, tau):
    """Fraction of scores >= tau in an ascending-sorted array, via binary search (O(log n))."""
    n = sorted_scores.size
    if n == 0:
        return None
    idx = int(np.searchsorted(sorted_scores, tau, side='left'))
    return float(n - idx) / n


def threshold_at_tar(tar_target, genuine_sorted):
    """Largest tau whose TAR (fraction of genuine >= tau) is still >= tar_target.

    Solved exactly on the sorted pool rather than by bisection: the rate is a step function, so a
    bracket always collapses onto a data value and can report the operating point on the wrong side of
    a tie. With ties counted at side='left' (the verifier's accept-iff-sim>=tau rule), the count of
    scores >= genuine[i] is n - i, so the largest admissible threshold is genuine[n - ceil(t*n)].
    Returns None when no threshold reaches the target.
    """
    n = int(genuine_sorted.size)
    if n == 0 or not (0.0 <= tar_target <= 1.0):
        return None
    required = int(math.ceil(tar_target * n - 1e-9))
    if required == 0:
        return 1.0
    index = n - required
    if index < 0:
        return None
    return float(genuine_sorted[index])


def _pool_value_at_or_above(pools, tau, strict=False):
    """The smallest score present in any pool that is >= tau (or > tau when strict)."""
    found = []
    for pool in pools:
        if pool.size == 0:
            continue
        index = int(np.searchsorted(pool, tau, side='right' if strict else 'left'))
        if index < pool.size:
            found.append(float(pool[index]))
    return min(found) if found else None


def _pool_value_below(pools, tau):
    """The largest score present in any pool that is strictly below tau."""
    found = []
    for pool in pools:
        if pool.size == 0:
            continue
        index = int(np.searchsorted(pool, tau, side='left')) - 1
        if index >= 0:
            found.append(float(pool[index]))
    return max(found) if found else None


def solve_eer(genuine_sorted, impostor_sorted, iterations=80):
    """EER: the exchange point where FNMR(tau) == FMR(tau) on the empirical ROC.

    FNMR - FMR is non-decreasing in tau, so the crossing region is found by bisection — but the
    number reported comes from the two empirical operating points that bracket it, never from the
    bisected tau itself. The rates are step functions, so a tau that lands a hair above a tied score
    reads the operating point after the jump: that is what turned two perfectly separable pools into
    a phantom non-zero EER. Perfect separation now reports EER = 0.0 at the first separating
    threshold, and a crossing inside a step is interpolated linearly between its bracketing points
    and labelled interpolated=True with the band kept.

    Cost is O(iterations * log n): the impostor pool here is 5.3e7 float32 scores, so this must not
    enumerate or unique them.
    """
    if genuine_sorted.size == 0 or impostor_sorted.size == 0:
        return None
    pools = (genuine_sorted, impostor_sorted)

    def operating_point(tau):
        fnmr = 1.0 - rate_at(genuine_sorted, tau)
        fmr = rate_at(impostor_sorted, tau)
        return (float(tau), fnmr, fmr, fnmr - fmr)

    lo = float(min(genuine_sorted[0], impostor_sorted[0])) - 0.5
    hi = float(max(genuine_sorted[-1], impostor_sorted[-1])) + 0.5
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if operating_point(mid)[3] >= 0.0:
            hi = mid
        else:
            lo = mid

    plus = _pool_value_at_or_above(pools, hi)
    for _ in range(2):  # bisection can settle exactly on a score whose own point is still negative
        if plus is None or operating_point(plus)[3] >= 0.0:
            break
        plus = _pool_value_at_or_above(pools, plus, strict=True)
    if plus is None:
        return None

    upper = operating_point(plus)
    if upper[3] == 0.0:
        return {'tau': plus, 'eer': upper[2], 'interpolated': False,
                'bracket': [plus, plus]}

    below = _pool_value_below(pools, plus)
    lower = operating_point(below) if below is not None else (plus, 0.0, 1.0, -1.0)
    fraction = lower[3] / (lower[3] - upper[3])
    return {'tau': 0.5 * (lower[0] + upper[0]),
            'eer': lower[2] + fraction * (upper[2] - lower[2]),
            'interpolated': True,
            'bracket': [lower[0], upper[0]]}


# ─────────────────────────── retrieval, per query ───────────────────────────

def canonical_metric_module():
    """Load evaluate_offline by path — the shipped, test-covered metric caller."""
    path = script_dir / 'evaluate_offline.py'
    spec = importlib.util.spec_from_file_location('evaluate_offline', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def per_query_retrieval(query_features, query_pids, query_camids,
                        gallery_features, gallery_pids, gallery_camids):
    """Per-query rank hits and average precision, replicating the canonical protocol.

    Protocol (identical to run_real_pipeline.compute_cmc_map:425-475):
      * float64 features, unit-normalised with the same ``+1e-8`` denominator the canonical caller
        applies in evaluate_offline.compute_cmc_map;
      * cosine distance ``1 - q·g``;
      * gallery entries with the same pid AND the same camera as the query are removed (the standard
        Market-1501 multi-distractor rule), as are junk pid -1;
      * a query with no surviving gallery match is INVALID and contributes to no aggregate.

    Returns (per_query dict, summary dict) — the caller asserts the aggregates match the canonical
    function, so this cannot silently drift.
    """
    def unit_norm(features):
        f = np.asarray(features, dtype=np.float64)
        return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)

    qn, gn = unit_norm(query_features), unit_norm(gallery_features)
    dist = 1.0 - qn @ gn.T
    max_rank = MAX_RANK
    hits = {1: np.zeros(dist.shape[0], dtype=bool),
            5: np.zeros(dist.shape[0], dtype=bool),
            10: np.zeros(dist.shape[0], dtype=bool)}
    ap = np.full(dist.shape[0], np.nan, dtype=np.float64)
    first_rank = np.full(dist.shape[0], -1, dtype=np.int32)
    genuine_count = np.zeros(dist.shape[0], dtype=np.int64)
    candidate_count = np.zeros(dist.shape[0], dtype=np.int64)
    valid = np.zeros(dist.shape[0], dtype=bool)

    g_pids, g_camids = np.asarray(gallery_pids), np.asarray(gallery_camids)
    for qi in range(dist.shape[0]):
        order = np.argsort(dist[qi])
        keep = ~((g_pids[order] == query_pids[qi]) & (g_camids[order] == query_camids[qi]))
        keep &= (g_pids[order] != -1)
        ordered = g_pids[order][keep]
        candidate_count[qi] = int(ordered.size)
        matches = (ordered == query_pids[qi])
        genuine_count[qi] = int(matches.sum())
        if matches.sum() == 0:
            continue
        valid[qi] = True
        cumulative = np.cumsum(matches)
        first = int(np.argmax(cumulative > 0))
        first_rank[qi] = first + 1
        for k in (1, 5, 10):
            hits[k][qi] = first < k
        n_true = int(matches.sum())
        precisions = cumulative / (np.arange(matches.size) + 1.0)
        ap[qi] = float((precisions * matches).sum() / n_true)

    n_valid = int(valid.sum())
    summary = {
        'queries_total': int(dist.shape[0]),
        'queries_valid': n_valid,
        'queries_invalid_no_gallery_match': int(dist.shape[0] - n_valid),
        'rank1': float(hits[1][valid].mean()) if n_valid else 0.0,
        'rank5': float(hits[5][valid].mean()) if n_valid else 0.0,
        'rank10': float(hits[10][valid].mean()) if n_valid else 0.0,
        'mAP': float(np.mean(ap[valid])) if n_valid else 0.0,
    }
    return {'valid': valid, 'hits': hits, 'ap': ap, 'first_rank': first_rank,
            'genuine_count': genuine_count, 'candidate_count': candidate_count}, summary


def rank_distribution(first_rank, valid):
    """Rank-1..∞ histogram over valid queries, plus the >10 tail, so Rank-10 is not read as 'rest'."""
    ranks = first_rank[valid]
    out = collections.Counter()
    for r in ranks:
        if r <= 0:
            out['invalid'] += 1
        elif r == 1:
            out['rank1'] += 1
        elif r <= 5:
            out['rank2_5'] += 1
        elif r <= 10:
            out['rank6_10'] += 1
        elif r <= 50:
            out['rank11_50'] += 1
        else:
            out['rank_gt_50'] += 1
    return dict(out)


# ─────────────────────────────── population ───────────────────────────────

def verify_population(data, manifest, arm_name):
    """Read the population out of the arrays, then compare with what the manifest claims.

    The array-derived numbers are the measurement; the manifest is only the expectation they are
    checked against. Any junk row, zero-norm row (a failed-inference sentinel that would be scored as
    an embedding) or non-finite row is fatal, because the population being measured would then not be
    the population that was audited.
    """
    report = {}
    for split in ('gallery', 'query'):
        pids = np.asarray(data[f'{split}_pids'], dtype=np.int64)
        cams = np.asarray(data[f'{split}_camids'], dtype=np.int64)
        features = data[f'{split}_features']
        norms = np.linalg.norm(features.astype(np.float64), axis=1)
        entry = {
            'rows': int(pids.size),
            'identities_unique': int(np.unique(pids).size),
            'identity_number_range': [int(pids.min()), int(pids.max())],
            'junk_pid_minus_1': int((pids == -1).sum()),
            'distractor_pid_0': int((pids == 0).sum()),
            'cameras_present': sorted(int(c) for c in np.unique(cams)),
            'zero_rows': int((norms < 1e-6).sum()),
            'non_finite_rows': int((~np.isfinite(features).all(axis=1)).sum()),
        }
        if features.shape[1] == 512:
            entry['unit_norm_max_deviation'] = float(np.max(np.abs(norms - 1.0)))
            entry['mean_norm'] = float(np.mean(norms))
        else:
            entry['norm_note'] = (f'raw post-BNNeck {features.shape[1]}-D features, deliberately '
                                  f'not unit-norm; cosine is computed after normalisation')
        claimed = (manifest.get('population', {}).get(split)
                   if split in ('gallery', 'query') else None)

        def _claim(*path):
            """The manifest's value for a nested key, or None when it makes no such claim.

            "Not claimed" must not become a disagreement: the query rule is pid > 0, so the manifest
            records no distractor count for the query split and a False here would be an invented
            mismatch rather than a measurement.
            """
            node = claimed
            for key in path:
                if not isinstance(node, dict) or key not in node:
                    return None
                node = node[key]
            return node

        def _agree(actual, *path):
            expected = _claim(*path)
            return None if expected is None else actual == expected

        def _agree_any(actual, *paths):
            for path in paths:
                if _claim(*path) is not None:
                    return _agree(actual, *path)
            return None

        excluded = _claim('excluded') or {}
        files_scanned = _claim('files_scanned')
        kept = _claim('kept')
        entry['manifest_agreement'] = {
            'rows': _agree(entry['rows'], 'kept'),
            'identities': _agree(entry['identities_unique'], 'identities'),
            'distractors_pid_0': _agree_any(entry['distractor_pid_0'],
                                           ('distractors_pid_0',), ('distractors',)),
            'identity_number_range': _agree(entry['identity_number_range'],
                                           'identity_number_range'),
            # The manifest's junk count is files dropped during the scan, not rows in the array, so it
            # is checked against the scan total rather than against entry['junk_pid_minus_1']. Junk
            # reaching the arrays is fatal regardless (see below).
            'scan_accounting': (None if files_scanned is None or kept is None
                                else files_scanned == kept + sum(
                                    v for v in excluded.values() if isinstance(v, int))),
        }
        # Not an agreement: an absent manifest section is recorded, never read as a mismatch.
        entry['manifest_counts_claimed'] = claimed is not None
        report[split] = entry
    report['arm'] = arm_name
    bad = [f'{split}.{key}' for split, entry in report.items() if isinstance(entry, dict)
           for key, ok in (entry.get('manifest_agreement') or {}).items()
           if ok is False]
    fatal = []
    for split in ('gallery', 'query'):
        if report[split]['junk_pid_minus_1'] != 0:
            fatal.append(f"{split} contains pid -1 junk rows that must not enter the population")
        if report[split]['zero_rows']:
            fatal.append(f"{split} contains {report[split]['zero_rows']} zero-norm rows "
                         f"(a failed-inference sentinel would be scored as an embedding)")
        if report[split]['non_finite_rows']:
            fatal.append(f"{split} contains non-finite rows")
    return report, bad, fatal


# ─────────────────────── verification score pools ───────────────────────

def verification_pools(sim, q_pids, q_camids, g_pids, g_camids):
    """Genuine and impostor pair-score pools, plus the per-query decisions a verifier makes.

    The pair definition follows the retrieval protocol so the two sections cannot disagree: a gallery
    row with the query's pid AND camera is the same physical appearance captured twice, so it is
    excluded from both pools; junk pid -1 is excluded. Every other gallery row is an impostor
    candidate for that query, including the pid-0 distractors — they are non-target identities, and a
    false accept against one is exactly what the threshold exists to prevent.

    Per query, ``genuine_max`` is the best score against a true match (missing it is the FNMR event)
    and ``impostor_max`` the best score against a wrong identity (exceeding tau is the FMR event).
    These two are the paired units for McNemar; a query with no genuine candidate is undefined for
    the former and is excluded from that statistic rather than counted as a miss.
    """
    q_pid = np.asarray(q_pids)[:, None]
    q_cam = np.asarray(q_camids)[:, None]
    g_pid = np.asarray(g_pids)[None, :]
    g_cam = np.asarray(g_camids)[None, :]
    same_identity = (g_pid == q_pid)
    same_camera = (g_cam == q_cam)
    junk = (g_pid == -1)
    excluded = (same_identity & same_camera) | junk
    genuine_mask = same_identity & ~excluded
    impostor_mask = ~same_identity & ~junk

    genuine_max = np.where(genuine_mask, sim, np.float32(-np.inf)).max(axis=1).astype(np.float64)
    impostor_max = np.where(impostor_mask, sim, np.float32(-np.inf)).max(axis=1).astype(np.float64)

    return {
        'genuine_scores': sim[genuine_mask].astype(np.float32),
        'impostor_scores': sim[impostor_mask].astype(np.float32),
        'genuine_max_per_query': genuine_max,
        'impostor_max_per_query': impostor_max,
        'queries_with_genuine': np.isfinite(genuine_max),
        'queries_with_impostor': np.isfinite(impostor_max),
        'duplicate_pairs_excluded': int((same_identity & same_camera & ~junk).sum()),
        'junk_gallery_rows': int((np.asarray(g_pids) == -1).sum()),
        'impostor_queries_with_no_candidate': int((~np.isfinite(impostor_max)).sum()),
    }


def score_summary(scores):
    if scores.size == 0:
        return None
    return {'n': int(scores.size), 'mean': float(np.mean(scores)),
            'std': (float(np.std(scores, ddof=1)) if scores.size > 1 else None),
            'min': float(np.min(scores)), 'max': float(np.max(scores)),
            'percentiles': {str(q): float(np.quantile(scores, q))
                            for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)}}


def verification_at_tau(pools, tau):
    """TAR/FNMR/FMR at the shipped tau, plus EER and FMR at matched TAR = 0.90.

    The spatial (Mahalanobis) gate of IdentityVerifier cannot be evaluated offline — no trajectory or
    covariance data exists for these still images — so every number here is the appearance gate in
    isolation and is labelled as such.
    """
    genuine = np.sort(pools['genuine_scores'])
    impostor = np.sort(pools['impostor_scores'])
    tar, fmr = rate_at(genuine, tau), rate_at(impostor, tau)
    out = {
        'tau': tau,
        'acceptance_rule': ('accept iff cosine_similarity >= tau '
                            '(core/identity_verifier.py rejects when sim < tau)'),
        'scope': 'appearance gate only; the Mahalanobis spatial gate is NOT MEASURED offline',
        'TAR': tar, 'FNMR': (None if tar is None else 1.0 - tar), 'FMR': fmr,
        'genuine_pairs': int(genuine.size), 'impostor_pairs': int(impostor.size),
        'genuine_score_summary': score_summary(genuine),
        'impostor_score_summary': score_summary(impostor),
        'Wilson_95_FNMR': (None if tar is None else
                           wilson_interval(int(round((1.0 - tar) * genuine.size)),
                                           int(genuine.size))),
        'Wilson_95_FMR': (None if fmr is None else
                          wilson_interval(int(round(fmr * impostor.size)), int(impostor.size))),
        'label': 'COMPUTED',
    }
    if tar is None or fmr is None:
        # The key is always present so a consumer cannot read a missing EER as an EER of zero.
        out['EER'] = 'NOT MEASURED (a score pool is empty, so the ROC has no operating range)'
        return out
    tau_90 = threshold_at_tar(0.90, genuine)
    if tau_90 is None:
        out['FMR_at_TAR_0.90'] = 'NOT MEASURED (TAR never reaches 0.90 in this population)'
    else:
        out['threshold_at_TAR_0.90'] = tau_90
        out['FMR_at_TAR_0.90'] = rate_at(impostor, tau_90)
    eer = solve_eer(genuine, impostor)
    out['EER'] = ('NOT MEASURED (a score pool is empty, so the ROC has no operating range)'
                  if eer is None else eer)
    return out


# ─────────────────────── camera-partition proxy ───────────────────────

def _distractor_rejection(evaluate_offline, data, tau):
    """Gallery-side distractor rejection, through the shipped function (no second copy).

    Returns None rather than 0.0 when the population has no distractors — TR-07 made that reachable,
    and a rate over an empty set is not a measurement.
    """
    rate, rejected, total = evaluate_offline.compute_distractor_rejection(
        data['query_features'], data['gallery_features'],
        np.asarray(data['gallery_pids']), threshold=tau)
    if rate is None:
        return {'status': 'NOT APPLICABLE',
                'reason': 'this population contains no pid-0 distractor gallery rows'}
    return {'status': 'COMPUTED', 'rejection_rate': float(rate), 'rejected': int(rejected),
            'distractors': int(total), 'tau': tau,
            'definition': 'a pid-0 gallery row is rejected when its maximum cosine similarity to '
                          'any query is below tau',
            'Wilson_95_rejection': wilson_interval(int(rejected), int(total))}


def camera_partition_proxy(evaluate_offline, data, low, high, tau):
    """Cross-view matching between two disjoint camera groups — a labelled PROXY, nothing more.

    Charter §8 forbids inferring cross-robot handoff success from Market-1501 retrieval metrics. This
    is a camera partition on one fixed dataset: it measures whether appearance identity survives a
    viewpoint change between two groups of cameras — the property a handoff depends on — but it
    involves no robot, no handoff event, no trajectory and no ground-truth handoff label.
    """
    directions = {}
    for src, dst in ((low, high), (high, low)):
        q_mask = np.isin(data['query_camids'], src)
        g_mask = np.isin(data['gallery_camids'], dst)
        qf, qp, qc = (data['query_features'][q_mask], data['query_pids'][q_mask],
                      np.asarray(data['query_camids'])[q_mask])
        gf, gp, gc = (data['gallery_features'][g_mask], np.asarray(data['gallery_pids'])[g_mask],
                      np.asarray(data['gallery_camids'])[g_mask])
        key = f'cams_{src}_queries_to_cams_{dst}_gallery'
        if qf.size == 0 or gf.size == 0:
            directions[key] = 'NOT MEASURED (empty query or gallery subset)'
            continue
        per_query, summary = per_query_retrieval(qf, qp, qc, gf, gp, gc)
        cmc, canonical_map = evaluate_offline.compute_cmc_map(qf, qp, qc, gf, gp, gc,
                                                             max_rank=MAX_RANK)
        pools = verification_pools(production_cosine_matrix(qf, gf), qp, qc, gp, gc)
        n_valid = summary['queries_valid']
        directions[key] = {
            'query_camera_group': list(src), 'gallery_camera_group': list(dst),
            'queries': int(qf.shape[0]), 'queries_valid': n_valid,
            'queries_invalid_no_gallery_match': summary['queries_invalid_no_gallery_match'],
            'gallery_rows': int(gf.shape[0]),
            'query_identities': int(np.unique(qp).size),
            'query_identities_with_a_gallery_match': int(np.unique(qp[per_query['valid']]).size),
            'rank1': summary['rank1'], 'rank5': summary['rank5'], 'rank10': summary['rank10'],
            'mAP': summary['mAP'],
            'canonical_agreement_abs': {'rank1': abs(float(cmc[0]) - summary['rank1']),
                                        'rank5': abs(float(cmc[4]) - summary['rank5']),
                                        'rank10': abs(float(cmc[9]) - summary['rank10']),
                                        'mAP': abs(float(canonical_map) - summary['mAP'])},
            'Wilson_95_rank1': wilson_interval(int(round(summary['rank1'] * n_valid)), n_valid),
            'Wilson_95_rank10': wilson_interval(int(round(summary['rank10'] * n_valid)), n_valid),
            'verification_at_tau': verification_at_tau(pools, tau),
            'protocol_note': 'the same-pid-same-camera duplicate exclusion is inert here because the '
                             'two camera groups are disjoint by construction',
        }
    return directions


# ────────────────────────────── threshold provenance ──────────────────────────────

def read_threshold(config_path, expected=0.70):
    """τ must be the value the shipped production config actually carries, not a remembered number."""
    import yaml

    with open(config_path, encoding='utf-8') as handle:
        loaded = yaml.safe_load(handle)
    params = loaded.get('algorithm_thresholds', {}).get('ros__parameters', {})
    tau = params.get('reid_similarity_threshold')
    return {'tau': tau, 'source_file': os.path.abspath(config_path),
            'source_key': 'algorithm_thresholds.ros__parameters.reid_similarity_threshold',
            'matches_expected_0.70': (float(tau) == float(expected) if tau is not None else False),
            'applied_by': 'identity_verifier.verify_candidate rejects when similarity < tau; '
                          'this script applies accept iff score >= tau, the same rule'}


# ────────────────────────────── experiment identity provenance ──────────────────────────────

def resolve_exp_id(manifest, requested, allow_mismatch=False):
    """Compare this analysis' ``--exp-id`` with the id stamped into the embeddings manifest.

    A disagreement means the report would attribute numbers to an experiment that did not produce
    them, so the default is to abort; an explicit override is allowed but stays visible in the
    report. A manifest without the key predates §4.4, so there is nothing to compare — that is
    recorded rather than treated as agreement.
    """
    requested_id = str(requested or '').strip()
    if not requested_id:
        raise SystemExit('--exp-id cannot be empty: it is the label this report is traced by')
    raw = manifest.get('exp_id')
    recorded_id = str(raw).strip() if raw is not None and str(raw).strip() else None

    outcome = {'requested_by_analysis': requested_id, 'recorded_by_extraction': recorded_id,
               'source': 'task7_embeddings_manifest.json:exp_id'}
    if recorded_id is None:
        outcome['agreement'] = 'manifest_has_no_exp_id'
        outcome['action'] = 'proceed'
        outcome['note'] = (f'the embeddings manifest carries no exp_id, so it predates ES-EXP07 '
                           f'§4.4; the numbers are reported under {requested_id!r} on that '
                           f'assumption alone')
        logger.warning('EXP-METADATA ' + outcome['note'])
    elif recorded_id == requested_id:
        outcome['agreement'] = 'match'
        outcome['action'] = 'proceed'
        logger.info(f'EXP-METADATA exp_id={requested_id} agrees with the embeddings manifest')
    else:
        detail = (f'exp_id mismatch: the embeddings manifest was generated for {recorded_id!r} '
                  f'but this analysis is labelled {requested_id!r}. Re-run with the matching '
                  f'--exp-id, or pass --allow-exp-id-mismatch if these really are the same '
                  f'experiment under two labels.')
        outcome['agreement'] = 'mismatch'
        if allow_mismatch:
            outcome['action'] = 'proceed_with_override'
            outcome['note'] = detail
            logger.warning('EXP-METADATA ' + detail)
        else:
            outcome['action'] = 'abort'
            raise SystemExit('EXP-METADATA ' + detail)
    return outcome


# ────────────────────────────── one arm end to end ──────────────────────────────

def analyse_arm(evaluate_offline, path, manifest, arm_name, tau, camera_groups,
                bootstrap_resamples, strict=True):
    data = dict(np.load(path))
    population, manifest_disagreements, fatal = verify_population(data, manifest, arm_name)
    if fatal:
        raise SystemExit(f'population defect in {arm_name}: {fatal}')
    if manifest_disagreements and strict:
        raise SystemExit(f'{arm_name} disagrees with the embeddings manifest on '
                         f'{manifest_disagreements}; refusing to analyse an unexpected file')

    per_query, summary = per_query_retrieval(
        data['query_features'], data['query_pids'], data['query_camids'],
        data['gallery_features'], data['gallery_pids'], data['gallery_camids'])
    cmc, canonical_map = evaluate_offline.compute_cmc_map(
        data['query_features'], data['query_pids'], data['query_camids'],
        data['gallery_features'], data['gallery_pids'], data['gallery_camids'],
        max_rank=MAX_RANK)

    # Two implementations of one metric is the defect TR-07 was; this refuses to ship it.
    agreement = {'rank1': abs(float(cmc[0]) - summary['rank1']),
                 'rank5': abs(float(cmc[4]) - summary['rank5']),
                 'rank10': abs(float(cmc[9]) - summary['rank10']),
                 'mAP': abs(float(canonical_map) - summary['mAP'])}
    worst = max(agreement.values())
    if worst > 1e-12:
        raise SystemExit(f'{arm_name}: per-query metric pass does not reproduce the canonical '
                         f'implementation (max abs difference {worst:.3e}); refusing to report '
                         f'either number')

    n_valid = summary['queries_valid']
    sim = production_cosine_matrix(data['query_features'], data['gallery_features'])
    pools = verification_pools(sim, data['query_pids'], data['query_camids'],
                              data['gallery_pids'], data['gallery_camids'])
    valid = per_query['valid']
    genuine_eligible = pools['queries_with_genuine'] & valid
    impostor_eligible = pools['queries_with_impostor'] & valid

    by_camera = {}
    for cam in population['query']['cameras_present']:
        cam_mask = (np.asarray(data['query_camids']) == cam) & valid
        n = int(cam_mask.sum())
        if n == 0:
            continue
        by_camera[str(cam)] = {
            'queries_valid': n,
            'rank1': float(per_query['hits'][1][cam_mask].mean()),
            'rank10': float(per_query['hits'][10][cam_mask].mean()),
            'Wilson_95_rank1': wilson_interval(int(per_query['hits'][1][cam_mask].sum()), n),
        }

    out = {
        'arm': arm_name,
        'file': {'path': os.path.abspath(path), 'bytes': os.path.getsize(path),
                 'sha256': sha256_of(path)},
        'feature_dim': int(data['gallery_features'].shape[1]),
        'inferential_role': ('deployed 512-D arm (part of the declared hypothesis family)'
                             if arm_name in DEPLOYED_512_ARMS else
                             'descriptive only: 768-D post-BNNeck features before the fitted '
                             'projection are not a deployed contract output, so no inference is '
                             'drawn from this arm'),
        'population': population,
        'retrieval': {
            'protocol': 'canonical Market-1501 CMC/mAP via evaluate_offline.compute_cmc_map '
                        '(run_real_pipeline.compute_cmc_map, validated against torchreid 0.2.5)',
            'max_rank': MAX_RANK,
            'queries_total': summary['queries_total'],
            'queries_valid': n_valid,
            'queries_invalid_no_gallery_match': summary['queries_invalid_no_gallery_match'],
            'rank1_pct': summary['rank1'] * 100, 'rank5_pct': summary['rank5'] * 100,
            'rank10_pct': summary['rank10'] * 100, 'mAP_pct': summary['mAP'] * 100,
            'Wilson_95_rank1': wilson_interval(int(per_query['hits'][1][valid].sum()), n_valid),
            'Wilson_95_rank5': wilson_interval(int(per_query['hits'][5][valid].sum()), n_valid),
            'Wilson_95_rank10': wilson_interval(int(per_query['hits'][10][valid].sum()), n_valid),
            'mAP_bootstrap_95': percentile_bootstrap_ci(per_query['ap'][valid],
                                                        resamples=bootstrap_resamples),
            'mAP_bootstrap_method': (f'percentile bootstrap, {bootstrap_resamples} resamples of the '
                                     f'per-query average-precision values, seed 20261019'),
            'rank_band_counts': rank_distribution(per_query['first_rank'], valid),
            'identity_coverage': {
                'evaluated_query_identities': int(np.unique(
                    np.asarray(data['query_pids'])[valid]).size),
                'query_identities_total': int(np.unique(np.asarray(data['query_pids'])).size),
                'mean_genuine_candidates_per_valid_query': (
                    float(per_query['genuine_count'][valid].mean()) if n_valid else None),
                'mean_candidates_per_valid_query': (
                    float(per_query['candidate_count'][valid].mean()) if n_valid else None)},
            'canonical_vs_per_query_max_abs_difference': worst,
            'agreement_detail': agreement,
            'distractor_rejection': _distractor_rejection(
                evaluate_offline, data, tau['tau']),
            # A zero-valid population has no measured retrieval number; 0.00 % is not a result.
            'label': ('COMPUTED' if n_valid else
                      f'NOT MEASURED — none of the {summary["queries_total"]} queries has a gallery '
                      f'identity match, so Rank-k and mAP have no denominator'),
        },
        'by_query_camera': by_camera,
        'verification': {
            'threshold': tau,
            'pair_pool': {'genuine_pairs': int(pools['genuine_scores'].size),
                          'impostor_pairs': int(pools['impostor_scores'].size),
                          'duplicate_pairs_excluded': pools['duplicate_pairs_excluded'],
                          'junk_gallery_rows': pools['junk_gallery_rows'],
                          'queries_without_any_impostor_candidate':
                              pools['impostor_queries_with_no_candidate']},
            'at_tau': verification_at_tau(pools, tau['tau']),
            'per_query_decisions': {
                'unit': 'query',
                'denominator_rule': ('a query contributes to the genuine rate only if a true match '
                                     'survives the duplicate exclusion, and to the impostor rate only '
                                     'if an impostor candidate exists; -inf maxima are never counted '
                                     'as rejections'),
                'genuine_accept_n': int(genuine_eligible.sum()),
                'genuine_accept_rate': float(
                    (pools['genuine_max_per_query'][genuine_eligible] >= tau['tau']).mean()),
                'false_accept_n': int(impostor_eligible.sum()),
                'false_accept_rate': float(
                    (pools['impostor_max_per_query'][impostor_eligible] >= tau['tau']).mean())},
        },
        'similarity_cross_check': None,
        '_arrays': {'valid': valid, 'rank1': per_query['hits'][1],
                    'genuine_max': pools['genuine_max_per_query'],
                    'impostor_max': pools['impostor_max_per_query'],
                    'genuine_eligible': genuine_eligible,
                    'impostor_eligible': impostor_eligible,
                    'ap': per_query['ap']},
    }
    out['similarity_cross_check'] = cross_check_against_verifier(
        data['query_features'], data['gallery_features'], data['query_pids'],
        data['gallery_pids'])
    if camera_groups:
        low, high = camera_groups
        out['camera_partition_proxy'] = camera_partition_proxy(
            evaluate_offline, data, low, high, tau['tau'])
        out['camera_partition_proxy']['status'] = (
            'PROXY ONLY — a camera partition on one fixed dataset. Charter §8: cross-robot handoff '
            'success, false-handoff rates and cross-robot performance must NOT be inferred from '
            'Market-1501 retrieval metrics; no robot, handoff event or handoff label is involved.')
    return out


# ─────────────────────────────── paired inference ───────────────────────────────

def paired_tests(results, family=HYPOTHESIS_FAMILY):
    """Exact McNemar per declared comparison, then Holm over the family.

    Pairing is by query index over the same population, so the units are identical across arms by
    construction; the script asserts the valid masks agree rather than assuming it.
    """
    outcomes = {
        'rank1_hit': lambda r: (r['_arrays']['valid'], r['_arrays']['rank1']),
        'genuine_accept_at_tau': lambda r: (r['_arrays']['genuine_eligible'],
                                            r['_arrays']['genuine_max'] >= r['verification']
                                            ['threshold']['tau']),
        'false_accept_at_tau': lambda r: (r['_arrays']['impostor_eligible'],
                                          r['_arrays']['impostor_max']
                                          >= r['verification']['threshold']['tau']),
    }
    tests = []
    for spec in family:
        a, b = spec['arms']
        if a not in results or b not in results:
            tests.append({'id': spec['id'], 'metric': spec['metric'], 'arms': list(spec['arms']),
                          'status': 'NOT MEASURED (arm absent from this run)'})
            continue
        mask_a, hit_a = outcomes[spec['metric']](results[a])
        mask_b, hit_b = outcomes[spec['metric']](results[b])
        if not np.array_equal(mask_a, mask_b):
            tests.append({'id': spec['id'], 'metric': spec['metric'],
                          'status': 'NOT APPLICABLE (the two arms do not share the same paired '
                                    'unit set, so McNemar has no valid 2x2 table)'})
            continue
        both = mask_a
        a_hit, b_hit = hit_a & both, hit_b & both
        n = int(both.sum())
        discordant_a = int((a_hit & ~b_hit).sum())
        discordant_b = int((b_hit & ~a_hit).sum())
        p = exact_mcnemar(discordant_a, discordant_b)
        tests.append({
            'id': spec['id'], 'metric': spec['metric'], 'unit': spec['unit'],
            'arms': list(spec['arms']), 'h0': spec['h0'], 'paired_units': n,
            'rate_A': float(a_hit.sum() / n) if n else None,
            'rate_B': float(b_hit.sum() / n) if n else None,
            'difference_A_minus_B': (float(a_hit.sum() / n - b_hit.sum() / n) if n else None),
            'discordant_A_only': discordant_a, 'discordant_B_only': discordant_b,
            'test': 'exact McNemar (binomial, two-sided) on the discordant pairs',
            'p_value': p,
            'status': 'COMPUTED' if p is not None else 'NOT APPLICABLE (no discordant pairs)',
        })
    adjusted = holm_corrections([(t['id'], t.get('p_value')) for t in tests
                                 if 'id' in t])
    for test in tests:
        if test['id'] in adjusted:
            test['holm_adjusted'] = adjusted[test['id']]
    return {'family_declared_before_analysis': family,
            'family_size': len([t for t in tests if t.get('p_value') is not None]),
            'correction': 'Holm-Boferoni step-down, alpha = 0.05, monotone-adjusted',
            'tests': tests}


# ────────────────────────────── markdown rendering ──────────────────────────────

def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def markdown(report, results):
    """Render the measured numbers. Nothing here computes: it only formats what is in the report."""
    ident = report.get('exp_id') or {}
    identity = (f"Experiment `{ident.get('requested_by_analysis', '?')}`"
                + (f" — extraction manifest recorded `{ident['recorded_by_extraction']}` "
                   f"({ident.get('agreement')})."
                   if ident.get('recorded_by_extraction') is not None else
                   ' — the extraction manifest records no exp_id '
                   f"({ident.get('agreement', 'unknown')}).")
                if ident else 'Experiment id: not recorded in this report.')
    lines = [f"# Task 7 statistics — measured from `{report['inputs']['embeddings_dir']}`",
             '',
             identity,
             f"Generated {report['generated_utc']} on {report['host']}. "
             f"Every figure is labelled COMPUTED / REPRODUCED / BLOCKED / NOT MEASURED / "
             f"NOT APPLICABLE; a cell that is not COMPUTED says why.",
             '',
             '## Retrieval (population: '
             f"`{report['population_rules']['population']}`)",
             '',
             '| Arm | dim | Queries valid | Rank-1 % (95 % CI) | Rank-5 % | Rank-10 % | '
             'mAP % (bootstrap CI) | Label |',
             '|---|---|---|---|---|---|---|---|']
    for arm, res in results.items():
        r = res['retrieval']
        ci = r['Wilson_95_rank1']
        boot = r['mAP_bootstrap_95']
        lines.append(
            f"| `{arm}` | {res['feature_dim']} | {r['queries_valid']} / {r['queries_total']} | "
            f"{r['rank1_pct']:.2f} ({_ci_text(ci)}) | {r['rank5_pct']:.2f} | "
            f"{r['rank10_pct']:.2f} | {r['mAP_pct']:.2f} "
            f"({_ci_text(boot)}) | {r['label']} |")

    lines += ['', '## Verification at the shipped threshold',
              '',
              f"τ = {report['threshold']['tau']} read from "
              f"`{os.path.basename(report['threshold']['source_file'])}` "
              f"(`{report['threshold']['source_key']}`); equals 0.70: "
              f"**{report['threshold']['matches_expected_0.70']}**. "
              'Accept iff similarity >= τ, the production rule. Appearance gate only — the '
              'Mahalanobis spatial gate is NOT MEASURED offline (no trajectory data exists).',
              '',
              '| Arm | genuine pairs | impostor pairs | FNMR at τ (95 % CI) | FMR at τ (95 % CI) | '
              'EER | τ at EER | FMR at TAR=0.90 |',
              '|---|---|---|---|---|---|---|---|']
    for arm, res in results.items():
        v = res['verification']['at_tau']
        eer = v.get('EER')
        if isinstance(eer, dict):
            eer_rate, eer_tau = f"{eer['eer'] * 100:.2f} %", f"{eer['tau']:.4f}"
        else:
            eer_rate, eer_tau = str(eer), '—'
        lines.append(
            f"| `{arm}` | {v['genuine_pairs']:,} | {v['impostor_pairs']:,} | "
            f"{_pct(v['FNMR'])} ({_ci_text(v['Wilson_95_FNMR'])}) | "
            f"{_pct(v['FMR'], 4)} ({_ci_text(v['Wilson_95_FMR'])}) | "
            f"{eer_rate} | {eer_tau} | {_plain(v.get('FMR_at_TAR_0.90'))} |")

    lines += ['', '## Paired inference (declared family, Holm-corrected)', '']
    for test in report['paired_tests']['tests']:
        if test.get('status') != 'COMPUTED':
            lines.append(f"- **{test['id']}** `{test.get('metric')}`: **{test['status']}**")
            continue
        h = test['holm_adjusted']
        lines.append(
            f"- **{test['id']}** `{test['metric']}` on {test['arms'][0]} vs {test['arms'][1]}: "
            f"rates {test['rate_A']:.4f} vs {test['rate_B']:.4f} "
            f"(Δ {test['difference_A_minus_B']:+.4f}), discordant "
            f"{test['discordant_A_only']} / {test['discordant_B_only']} of "
            f"{test['paired_units']} pairs, exact McNemar p = {test['p_value']:.6g}, "
            f"Holm-adjusted {h['adjusted']:.6g} → **{h['decision']}**")
    lines += ['', f"- OSNet vs TransReID paired test: "
                  f"**{report['osnet_comparison']['status']}** — {report['osnet_comparison']['reason']}",
              '', '## Camera-partition proxy', '',
              report['camera_partition_disclaimer'], '']
    for arm, res in results.items():
        proxy = res.get('camera_partition_proxy')
        if not proxy:
            continue
        for key, value in proxy.items():
            if key == 'status':
                continue
            if isinstance(value, str):
                lines.append(f"- `{arm}` {key}: {value}")
                continue
            lines.append(
                f"- `{arm}` **{key}**: {value['queries_valid']} valid queries of "
                f"{value['queries']} (gallery rows {value['gallery_rows']}), Rank-1 "
                f"{_pct(value['rank1'])} ({_ci_text(value['Wilson_95_rank1'])}), Rank-10 "
                f"{_pct(value['rank10'])}, mAP {_pct(value['mAP'])}; FNMR at τ "
                f"{_pct(value['verification_at_tau']['FNMR'])}, FMR at τ "
                f"{_pct(value['verification_at_tau']['FMR'], 4)}")
    lines += ['', '## What is not measured here', '']
    for key, value in report['not_measured'].items():
        lines.append(f"- **{key}**: {value}")
    return '\n'.join(lines) + '\n'


# ─────────────────────────────────── main ───────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='ES-EXP07 Task 7 statistics: retrieval, verification, paired inference.')
    parser.add_argument('--embeddings-dir', required=True,
                        help='Directory holding task7_embeddings_manifest.json and the .npz arms.')
    parser.add_argument('--threshold-config', required=True,
                        help='algorithm_thresholds.yaml — the tau actually shipped is read from '
                             'here and asserted to be 0.70 (§8).')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--arms', default=','.join(
        ['transreid_768_agnostic', 'transreid_512_agnostic', 'transreid_512_sie']))
    parser.add_argument('--camera-low', default='1,2,3', help='Camera group treated as "robot A".')
    parser.add_argument('--camera-high', default='4,5,6', help='Camera group treated as "robot B".')
    parser.add_argument('--bootstrap-resamples', type=int, default=2000)
    parser.add_argument('--skip-camera-partition', action='store_true')
    parser.add_argument('--allow-manifest-drift', action='store_true',
                        help='Analyse an arm whose counts disagree with the manifest. Off by '
                             'default: a silent re-population would make the recorded numbers '
                             'untraceable.')
    parser.add_argument('--exp-id', default=DEFAULT_EXP_ID,
                        help='Experiment label for this report, compared against the exp_id '
                             'recorded by generate_task7_embeddings.py (ES-EXP07 §4.4).')
    parser.add_argument('--allow-exp-id-mismatch', action='store_true',
                        help='Report anyway when the manifest was generated under a different '
                             'exp_id. Off by default; the override is written into the report.')
    args = parser.parse_args()

    emb_dir = os.path.abspath(args.embeddings_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(emb_dir, 'task7_embeddings_manifest.json')
    if not os.path.isfile(manifest_path):
        raise SystemExit(f'no embeddings manifest at {manifest_path} — run '
                         f'generate_task7_embeddings.py first')
    with open(manifest_path, encoding='utf-8') as handle:
        manifest = json.load(handle)

    # §4.4 gate: abort before computing anything if these numbers belong to another experiment.
    exp_id = resolve_exp_id(manifest, args.exp_id,
                            allow_mismatch=args.allow_exp_id_mismatch)

    tau = read_threshold(args.threshold_config)
    if not tau['matches_expected_0.70']:
        raise SystemExit(f"§8 requires the shipped threshold to be verified, not assumed: the "
                         f"config carries {tau['tau']!r}, not 0.70. Refusing to compute FNMR/FMR.")

    # ── integrity gate: the files analysed must be the files the audited run produced ──
    integrity = {}
    for arm, info in manifest.get('files', {}).items():
        present = os.path.isfile(info['path'])
        now = sha256_of(info['path']) if present else None
        integrity[arm] = {'path': info['path'], 'exists': present,
                          'manifest_sha256': info.get('sha256'), 'now_sha256': now,
                          'match': bool(now == info.get('sha256'))}
    mismatched = [a for a, v in integrity.items() if not v['match']]
    if mismatched:
        raise SystemExit(f'integrity gate FAILED: {mismatched} no longer hash to the digest '
                         f'recorded by the extraction run. The embeddings behind these numbers are '
                         f'not the ones that were audited.')

    evaluate_offline = canonical_metric_module()
    low = [int(c) for c in args.camera_low.split(',') if c.strip()]
    high = [int(c) for c in args.camera_high.split(',') if c.strip()]
    groups = None if args.skip_camera_partition else (low, high)

    results = {}
    for arm in [a.strip() for a in args.arms.split(',') if a.strip()]:
        info = manifest['files'].get(arm)
        if info is None:
            logger.warning(f'arm {arm} absent from the manifest — NOT MEASURED')
            continue
        logger.info(f'=== {arm} ===')
        started = time.perf_counter()
        results[arm] = analyse_arm(evaluate_offline, info['path'], manifest, arm, tau, groups,
                                   args.bootstrap_resamples,
                                   strict=not args.allow_manifest_drift)
        r = results[arm]['retrieval']
        at_tau = results[arm]['verification']['at_tau']
        logger.info(f'{arm}: valid {r["queries_valid"]}/{r["queries_total"]}  '
                    f'Rank-1 {r["rank1_pct"]:.2f} %  Rank-10 {r["rank10_pct"]:.2f} %  '
                    f'mAP {r["mAP_pct"]:.2f} %  '
                    f'FNMR@tau {_pct(at_tau["FNMR"], 4)}  FMR@tau {_pct(at_tau["FMR"], 4)}  '
                    f'({time.perf_counter() - started:.1f} s)')

    if not results:
        raise SystemExit('no arm analysed — nothing to report')

    paired = paired_tests(results)
    report = {
        'exp_id': exp_id,
        'experiment': 'ES-EXP07 Task 7 statistics (TR-08)',
        'evaluation_type': 'retrieval + verification + paired inference on real embeddings',
        'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'host': {'platform': sys.platform, 'python': sys.version.split()[0],
                 'numpy': np.__version__},
        'inputs': {'embeddings_dir': emb_dir,
                   'manifest': {'path': manifest_path, 'sha256': sha256_of(manifest_path)},
                   'integrity': integrity,
                   'extraction_run': {'exp_id': manifest.get('exp_id'),
                                      'generated_utc': manifest.get('generated_utc'),
                                      'elapsed_s': manifest.get('elapsed_s'),
                                      'host': manifest.get('host'),
                                      'assets': manifest.get('assets')}},
        'population_rules': manifest.get('population'),
        'threshold': tau,
        'metric_source': {'canonical': 'run_real_pipeline.compute_cmc_map via '
                                       'evaluate_offline.compute_cmc_map '
                                       '(validated against torchreid 0.2.5)',
                          'per_query_pass': 'this script, asserted equal to the canonical '
                                            'aggregates before any number is reported',
                          'similarity': 'production_cosine_matrix, cross-checked against '
                                        'IdentityVerifier.compute_cosine_similarity'},
        'arms': {arm: {k: v for k, v in res.items() if k != '_arrays'}
                 for arm, res in results.items()},
        'paired_tests': paired,
        'osnet_comparison': {
            'status': 'NOT APPLICABLE on this host',
            'reason': (manifest.get('osnet_factory_attempt', {}).get('result')
                       or 'no OSNet arm in the embeddings manifest'),
            'effect_on_comparison': manifest.get('osnet_factory_attempt', {}).get(
                'effect_on_comparison',
                'no OSNet embeddings => no paired per-query outcomes => no paired test'),
            'charter': '§3B forbids altering OSNet to make the comparison possible; §8 forbids a '
                       'paired test without paired outcomes for both arms',
        },
        'camera_partition_disclaimer': (
            'A camera partition on one fixed dataset. It measures whether appearance identity '
            'survives a viewpoint change between two camera groups — the property a handoff depends '
            'on — but involves no robot, no handoff event, no trajectory and no handoff label. '
            'Charter §8 explicitly forbids inferring cross-robot handoff success or false-handoff '
            'rates from Market-1501 retrieval metrics, so nothing in this section is a handoff '
            'measurement.'),
        'not_measured': {
            'jetson_tensorrt_latency': 'BLOCKED — Pending Jetson Orin Nano access (TR-11)',
            'osnet_retrieval_on_this_population': (
                'BLOCKED on this host — the frozen baseline cannot be imported '
                '(torchvision::nms); §3B forbids repairing it'),
            'spatial_gate_mahalanobis': 'NOT MEASURED — requires per-robot trajectory and '
                                        'covariance data that do not exist offline',
            'cross_robot_handoff_success': 'NOT MEASURED and NOT INFERABLE — no robot or handoff '
                                           'dataset exists in this project',
            'statistical_comparison_against_recorded_gate3_numbers': (
                'NOT APPLICABLE — the recorded run used the invalid id_band population '
                '(TR-07), so its per-query outcomes are not the same population as this one'),
        },
        'labels_legend': list(LABELS),
    }

    with open(os.path.join(out_dir, 'task7_statistics.json'), 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, default=_json_default)
    with open(os.path.join(out_dir, 'task7_population.json'), 'w', encoding='utf-8') as handle:
        json.dump({'population_rules': report['population_rules'],
                   'derived_from_arrays': {arm: res['population']
                                           for arm, res in results.items()},
                   'integrity': integrity},
                  handle, indent=2, default=_json_default)
    markdown_path = os.path.join(out_dir, 'task7_statistics.md')
    with open(markdown_path, 'w', encoding='utf-8') as handle:
        handle.write(markdown(report, results))
    logger.info(f'Reports: {out_dir}')


def _ci_text(ci):
    return '—' if ci is None else f"{ci[0] * 100:.2f}–{ci[1] * 100:.2f}"


def _pct(value, decimals=2):
    if value is None:
        return 'NOT MEASURED'
    return f'{value * 100:.{decimals}f} %'


def _plain(value):
    return (f'{value * 100:.4f} %' if isinstance(value, float)
            else str(value))


if __name__ == '__main__':
    main()

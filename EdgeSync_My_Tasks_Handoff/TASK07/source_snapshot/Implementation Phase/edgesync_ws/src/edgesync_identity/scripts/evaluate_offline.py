#!/usr/bin/env python3
"""Gate 3 — Offline Re-ID Accuracy Baseline Evaluation.

ES-EXP07-DCP-REV10 / TR-07:
  Evaluates TransReID (ViT-Base/16 + SVD) and, when a checkpoint is supplied, the
  frozen OSNet x1.0 comparison baseline on an explicitly named Market-1501
  evaluation population (--population, default 'directory' = official test-split
  semantics). The ID band 0752-1501 is selectable only to reproduce older
  recorded numbers; it is measured-invalid as a train/test split guard and every
  number produced under it must be labelled as such.

  Uses frozen threshold tau = 0.70 (do NOT retune).

  Reports:
    - Rank-1 accuracy
    - mAP (mean Average Precision)
    - Feature dimensionality
    - Offline inference latency (CPU/GPU, per-image)
    - Distractor rejection rate at tau = 0.70
    - The population manifest: files scanned/kept/excluded per split, exclusion
      reasons, identities and the rule text the counts were asserted from

  Results are marked as "Offline Market-1501 evaluation" and must NOT be
  presented as Jetson Orin Nano latency or multi-robot results.

Usage:
    python evaluate_offline.py \\
        --transreid-checkpoint models/transreid/transformer_120.pth \\
        --transreid-projection models/projections/svd_projection_768_to_512.npz \\
        --osnet-checkpoint models/osnet/osnet_x1_0_market1501.pth \\
        --market1501 data/Market-1501-v15.09.15 \\
        --population directory \\
        --output results/experiment7/gate3_offline_results.json

Traces: ISSUE-05 (split provenance), ISSUE-07 (genuine comparison), TR-02 (no
zero-embedding injection), TR-07 (named, auditable evaluation population).
"""
import argparse
import os
import re
import sys
import glob
import json
import time
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Ensure package imports work
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent))


def parse_market1501_filename(filename):
    """Parse Market-1501 filename into (person_id, camera_id).

    Naming: <4-digit ID>_c<1-digit cam>s<seq>_<frame>.jpg
    Returns: (pid: int, camid: int) or (-1, -1) on failure.
    """
    try:
        parts = os.path.basename(filename).split('_')
        pid = int(parts[0])
        camid = int(parts[1][1])  # 'c1' -> 1
        return pid, camid
    except (ValueError, IndexError):
        return -1, -1


POPULATION_RULES = {
    'directory': {
        'gallery_rule': 'pid >= 0 in bounding_box_test/ — official test gallery: distractor '
                        'pid 0 is KEPT as gallery negatives, junk pid -1 is excluded',
        'query_rule': 'pid > 0 in query/ — official query set',
        'provenance': 'Reproduces the population the recorded arm actually used '
                      '(run_real_pipeline.py:488-498 → gate3_real_results_v2_corrected.json).',
        'validity': 'Valid: this is the published Market-1501 test population, taken from the '
                    'directory semantics rather than from an identity-number band.',
    },
    'id_band_752_1501': {
        'gallery_rule': '752 <= pid <= 1501 in bounding_box_test/',
        'query_rule': '752 <= pid <= 1501 in query/',
        'provenance': 'The band described as the "frozen test split" in the Task 7 proposal and '
                      'report and hard-coded here before TR-07.',
        'validity': 'MEASURED AS INVALID AS A LEAKAGE GUARD: train and test identity numbers '
                    'interleave in this bundle (train spans 2..1500, test spans 1..1501), so the '
                    'band neither separates train from test nor matches the official split. '
                    'Measured effect: keeps 6,226/15,913 gallery (39.13%) and 1,774/3,368 query '
                    '(52.67%), drops 355 of the 750 test identities, and leaves 0 distractors so '
                    'the distractor-rejection metric becomes structurally 0/0. Retained only to '
                    'reproduce the older numbers; it must be labelled, never presented as the '
                    'frozen split.',
    },
}
DEFAULT_POPULATION = 'directory'

_NAME_GRAMMAR = re.compile(r'^(-?\d+)_c(\d)s(\d)_.*\.jpg$')


def _classify_name(filename):
    """Return (pid, camid, exclude_reason_or_None) for one Market-1501 basename."""
    if not _NAME_GRAMMAR.match(filename):
        return None, None, 'unparsable_filename'
    pid, camid = parse_market1501_filename(filename)
    return pid, camid, None


def collect_gallery_and_query(market1501_root, population=DEFAULT_POPULATION,
                              id_min=752, id_max=1501):
    """Collect the evaluation population and return an auditable manifest of it.

    TR-07: the population is now a named, explicit choice instead of an undocumented ID band,
    and every exclusion is counted with its reason so a reported Rank-1/mAP can be traced back to
    exactly the images it was computed over.

    Args:
        market1501_root: Dataset root containing bounding_box_test/ and query/.
        population: 'directory' (official semantics, default) or 'id_band_752_1501'.
        id_min/id_max: Only used by the 'id_band_752_1501' population.

    Returns:
        (gallery, query, manifest): gallery/query are (path, pid, camid) triples.
    """
    if population not in POPULATION_RULES:
        raise ValueError(f"unknown population '{population}'; "
                         f"choose one of {sorted(POPULATION_RULES)}")

    gallery_dir = os.path.join(market1501_root, 'bounding_box_test')
    query_dir = os.path.join(market1501_root, 'query')

    if not os.path.isdir(gallery_dir):
        raise FileNotFoundError(f"Gallery directory not found: {gallery_dir}")
    if not os.path.isdir(query_dir):
        raise FileNotFoundError(f"Query directory not found: {query_dir}")

    def collect(split_dir, split):
        kept, excluded = [], {}
        scanned = 0
        for img_path in sorted(glob.glob(os.path.join(split_dir, '*.jpg'))):
            scanned += 1
            pid, camid, reason = _classify_name(os.path.basename(img_path))
            if reason is None:
                if pid == -1:
                    reason = 'junk_identity_-1'
                elif pid == 0 and split == 'query':
                    reason = 'distractor_not_valid_query'
            if reason is None:
                if population == 'directory':
                    keep = pid >= 0 if split == 'gallery' else pid > 0
                else:
                    keep = id_min <= pid <= id_max
                if not keep:
                    reason = f'outside_{population}'
            if reason is None:
                kept.append((img_path, pid, camid))
            else:
                excluded[reason] = excluded.get(reason, 0) + 1
        return kept, excluded, scanned

    gallery, gallery_excluded, gallery_scanned = collect(gallery_dir, 'gallery')
    query, query_excluded, query_scanned = collect(query_dir, 'query')

    gallery_pids = [p for _, p, _ in gallery]
    query_pids = [p for _, p, _ in query]
    rules = POPULATION_RULES[population]
    validity = rules['validity']
    if population == 'id_band_752_1501' and (id_min, id_max) != (752, 1501):
        validity += (f" NOTE: the percentages above were measured for the 752-1501 band; "
                     f"this run used {id_min}-{id_max}, so they do not describe this population.")
    manifest = {
        'population': population,
        'rules': {'gallery': rules['gallery_rule'], 'query': rules['query_rule']},
        'provenance': rules['provenance'],
        'validity': validity,
        'id_band': {'id_min': id_min, 'id_max': id_max} if population == 'id_band_752_1501' else None,
        'gallery_dir': os.path.normpath(os.path.abspath(gallery_dir)),
        'query_dir': os.path.normpath(os.path.abspath(query_dir)),
        'gallery': {
            'files_scanned': gallery_scanned,
            'kept': len(gallery),
            'excluded': gallery_excluded,
            'identities': len(set(gallery_pids)),
            'distractors_pid_0': sum(1 for p in gallery_pids if p == 0),
            'identity_number_range': [min(gallery_pids), max(gallery_pids)] if gallery_pids else None,
        },
        'query': {
            'files_scanned': query_scanned,
            'kept': len(query),
            'excluded': query_excluded,
            'identities': len(set(query_pids)),
            'identity_number_range': [min(query_pids), max(query_pids)] if query_pids else None,
        },
        'counts_asserted_from': 'this run only — no standard dataset count is assumed',
    }

    logger.info(f"Population '{population}':")
    logger.info(f"  Gallery: {len(gallery)}/{gallery_scanned} kept, excluded {gallery_excluded}")
    logger.info(f"  Query:   {len(query)}/{query_scanned} kept, excluded {query_excluded}")
    logger.info(f"  Gallery identities: {manifest['gallery']['identities']}"
                f" (incl. {manifest['gallery']['distractors_pid_0']} distractor images, pid 0)")
    logger.info(f"  Query identities:   {manifest['query']['identities']}")
    if population == 'id_band_752_1501':
        logger.warning("id_band_752_1501 is a MEASURED-INVALID split guard (TR-07); label every "
                       "number produced under it and do not present it as the frozen split.")

    return gallery, query, manifest


def extract_features_batch(extractor, image_paths, batch_size=32, split='gallery'):
    """Extract embeddings, EXCLUDING unusable images instead of zero-filling them (TR-02).

    The previous version appended ``np.zeros(512)`` for any image ``cv2.imread`` could not
    decode, which put a fake embedding into the population, counted it in Rank-1/mAP, and
    deflated the latency mean with a 0.0 ms entry. Nothing is substituted here: an image that
    cannot produce a valid embedding is removed and reported.

    Returns:
        (features, latencies_ms, kept_indices, excluded)
        features: np.ndarray (len(kept_indices), D) — only successfully embedded images.
        latencies_ms: one real measurement per kept image.
        kept_indices: indices into ``image_paths`` so pid/camid metadata stays aligned.
        excluded: list of ``{'index', 'path', 'reason'}`` records.
    """
    import cv2
    from edgesync_identity.core.reid_input_adapter import is_invalid_embedding

    features = []
    latencies_ms = []
    kept_indices = []
    excluded = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]

        for offset, img_path in enumerate(batch_paths):
            index = i + offset
            img = cv2.imread(img_path)
            if img is None:
                excluded.append({'index': index, 'path': img_path, 'reason': 'decode_failed'})
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            t0 = time.perf_counter()
            emb = extractor.extract_embedding(img)
            dt = (time.perf_counter() - t0) * 1000.0

            reason = is_invalid_embedding(np.asarray(emb, dtype=np.float32).reshape(-1))
            if reason is not None:
                # The frozen extractor contract returns the zeros sentinel on failure; record
                # which stage produced it when the extractor reports one (extractor TR-02).
                failure = getattr(extractor, 'last_failure', None)
                stage = failure.get('stage') if isinstance(failure, dict) else None
                excluded.append({'index': index, 'path': img_path,
                                 'reason': f'invalid_embedding: {reason}',
                                 'extractor_stage': stage})
                continue

            features.append(emb)
            latencies_ms.append(dt)
            kept_indices.append(index)

        if i % (batch_size * 10) == 0:
            logger.info(f"  Extracted {min(i + batch_size, len(image_paths))}/{len(image_paths)} "
                        f"from '{split}' (kept {len(kept_indices)}, excluded {len(excluded)})")

    if excluded:
        logger.warning(f"  '{split}': {len(excluded)} image(s) excluded, "
                       f"first three: {excluded[:3]}")

    if not features:
        # Fail the run rather than hand back an empty feature matrix that a downstream
        # metric would report as a score (TR-02 / charter §5: no substituted outputs).
        raise RuntimeError(
            f"extract_features_batch('{split}'): 0 of {len(image_paths)} images produced a "
            f"valid embedding. Exclusion reasons: {excluded[:5]}. No embeddings were "
            f"substituted; the evaluation cannot continue."
        )

    return np.stack(features), latencies_ms, kept_indices, excluded


def compute_cmc_map(query_features, query_pids, query_camids,
                    gallery_features, gallery_pids, gallery_camids,
                    max_rank=50):
    """CMC curve and mAP, computed by the single validated implementation.

    run_real_pipeline.compute_cmc_map is the copy checked against torchreid 0.2.5 and it assumes
    unit-norm features, so they are normalised here first -- this script's callers may pass raw
    embeddings. Replaces the second, divergent copy of the metric that lived here, which counted
    pid -1 junk gallery entries as impostors instead of excluding them.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from run_real_pipeline import compute_cmc_map as _canonical_cmc_map

    def _unit_norm(features):
        features = np.asarray(features, dtype=np.float64)
        return features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

    return _canonical_cmc_map(_unit_norm(query_features), query_pids, query_camids,
                              _unit_norm(gallery_features), gallery_pids, gallery_camids,
                              max_rank)


def compute_distractor_rejection(query_features, gallery_features,
                                  gallery_pids, threshold=0.70):
    """Compute distractor rejection rate at given threshold.

    A distractor is rejected if its maximum cosine similarity to any query
    is below the threshold.
    """
    # Normalize
    qf = query_features / (np.linalg.norm(query_features, axis=1, keepdims=True) + 1e-8)
    gf = gallery_features / (np.linalg.norm(gallery_features, axis=1, keepdims=True) + 1e-8)

    # Distractors are pid 0 (junk pid -1 never enters the population under either rule).
    distractor_mask = (gallery_pids <= 0)
    n_distractors = distractor_mask.sum()

    if n_distractors == 0:
        # None, not 0.0: a rate over an empty set is not a measurement (TR-07 made this
        # reachable — the id_band population contains no distractors at all).
        return None, 0, 0

    # Max similarity of each gallery to any query
    sim_mat = np.dot(gf, qf.T)  # (G, Q)
    max_sim = sim_mat.max(axis=1)  # (G,)

    # Distractors rejected = max_sim < threshold
    rejected = (max_sim[distractor_mask] < threshold).sum()
    rejection_rate = float(rejected) / float(n_distractors)

    return rejection_rate, int(rejected), int(n_distractors)


def evaluate_model(model_name, extractor, gallery, query, threshold=0.70):
    """Full evaluation pipeline for one model."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  Evaluating: {model_name}")
    logger.info(f"{'='*60}")

    # Extract gallery features
    logger.info("Extracting gallery features...")
    gallery_paths = [g[0] for g in gallery]
    gallery_features, gallery_latencies, g_keep, g_excluded = extract_features_batch(
        extractor, gallery_paths, split='gallery')
    gallery_pids = np.array([gallery[i][1] for i in g_keep])
    gallery_camids = np.array([gallery[i][2] for i in g_keep])

    # Extract query features
    logger.info("Extracting query features...")
    query_paths = [q[0] for q in query]
    query_features, query_latencies, q_keep, q_excluded = extract_features_batch(
        extractor, query_paths, split='query')
    query_pids = np.array([query[i][1] for i in q_keep])
    query_camids = np.array([query[i][2] for i in q_keep])

    # Feature dimension
    feat_dim = gallery_features.shape[1]

    # Compute CMC and mAP
    logger.info("Computing CMC and mAP...")
    cmc, mAP = compute_cmc_map(
        query_features, query_pids, query_camids,
        gallery_features, gallery_pids, gallery_camids,
    )

    # Compute distractor rejection
    rej_rate, n_rejected, n_total = compute_distractor_rejection(
        query_features, gallery_features, gallery_pids, threshold
    )

    # Latency stats — every entry is a real measurement now (TR-02 removed the 0.0 fillers)
    all_latencies = gallery_latencies + query_latencies
    mean_lat = float(np.mean(all_latencies)) if all_latencies else 0.0
    p95_lat = float(np.percentile(all_latencies, 95)) if all_latencies else 0.0
    p99_lat = float(np.percentile(all_latencies, 99)) if all_latencies else 0.0

    results = {
        'model_name': model_name,
        'feature_dim': feat_dim,
        'rank1': float(cmc[0]) * 100,
        'rank5': float(cmc[4]) * 100 if len(cmc) > 4 else 0.0,
        'rank10': float(cmc[9]) * 100 if len(cmc) > 9 else 0.0,
        'mAP': mAP * 100,
        'threshold_tau': threshold,
        'distractor_rejection_rate': (rej_rate * 100) if rej_rate is not None else None,
        'distractors_rejected': n_rejected,
        'distractors_total': n_total,
        'distractor_rejection_status': (
            'MEASURED' if n_total else
            'NOT APPLICABLE — this population contains no distractor images'),
        'latency_mean_ms': mean_lat,
        'latency_p95_ms': p95_lat,
        'latency_p99_ms': p99_lat,
        'n_gallery_selected': len(gallery),
        'n_query_selected': len(query),
        'n_gallery_evaluated': int(gallery_features.shape[0]),
        'n_query_evaluated': int(query_features.shape[0]),
        'n_gallery_excluded': len(g_excluded),
        'n_query_excluded': len(q_excluded),
        'gallery_exclusions': g_excluded,
        'query_exclusions': q_excluded,
        'identities_evaluated': {
            'gallery': len(set(int(p) for p in gallery_pids)),
            'query': len(set(int(p) for p in query_pids)),
        },
        'evaluation_type': 'Offline Market-1501 evaluation',
        'device': 'GPU (development workstation)' if _has_cuda() else 'CPU',
    }

    logger.info(f"\n  Results for {model_name}:")
    logger.info(f"    Feature dim:           {feat_dim}")
    logger.info(f"    Rank-1:                {results['rank1']:.2f}%")
    logger.info(f"    Rank-5:                {results['rank5']:.2f}%")
    logger.info(f"    mAP:                   {results['mAP']:.2f}%")
    if n_total:
        logger.info(f"    Distractor rejection:  {results['distractor_rejection_rate']:.2f}% "
                    f"({n_rejected}/{n_total})")
    else:
        logger.info("    Distractor rejection:  NOT APPLICABLE (0 distractors in population)")
    logger.info(f"    Mean latency:          {mean_lat:.2f} ms")
    logger.info(f"    P95 latency:           {p95_lat:.2f} ms")
    logger.info(f"    Evaluated:             {results['n_gallery_evaluated']}/{len(gallery)} gallery, "
                f"{results['n_query_evaluated']}/{len(query)} query "
                f"(excluded {len(g_excluded)}+{len(q_excluded)})")

    return results


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _format_metric(fmt, val):
    """Render one comparison-table cell.

    A metric with no measurable value (e.g. distractor rejection computed over a population
    that contains no distractors) renders as NOT APPLICABLE. It must never be collapsed to
    0.00, which would read as a measured rejection rate of zero.
    """
    if val is None:
        return 'NOT APPLICABLE'
    if isinstance(val, str):
        return val
    return fmt.format(val)


def main():
    parser = argparse.ArgumentParser(
        description='Gate 3: Offline Re-ID Accuracy Baseline Evaluation.',
    )
    parser.add_argument('--transreid-checkpoint', required=True,
                        help='Path to TransReID transformer_120.pth')
    parser.add_argument('--transreid-projection', required=True,
                        help='Path to SVD projection svd_projection_768_to_512.npz')
    parser.add_argument('--osnet-checkpoint', default='',
                        help='Path to OSNet osnet_x1_0_market1501.pth (optional)')
    parser.add_argument('--market1501', required=True,
                        help='Path to Market-1501 dataset root')
    parser.add_argument('--threshold', type=float, default=0.70,
                        help='Frozen similarity threshold tau (default: 0.70)')
    parser.add_argument('--population', default=DEFAULT_POPULATION,
                        choices=sorted(POPULATION_RULES),
                        help="Evaluation population. 'directory' = official Market-1501 test "
                             "semantics (default, reproduces the recorded arm). "
                             "'id_band_752_1501' = the band previously documented as the frozen "
                             "split; measured invalid as a split guard, kept only to reproduce "
                             "older numbers, and every number produced with it must be labelled.")
    parser.add_argument('--id-min', type=int, default=752,
                        help='Only used with --population id_band_752_1501')
    parser.add_argument('--id-max', type=int, default=1501,
                        help='Only used with --population id_band_752_1501')
    parser.add_argument('--output', default='gate3_offline_results.json',
                        help='Output JSON file for results')
    args = parser.parse_args()

    print("=" * 78)
    print("  Gate 3 -- Offline Re-ID Accuracy Baseline")
    print("  ES-EXP07-DCP-REV10")
    print("  Evaluation Type: Offline Market-1501 evaluation")
    print(f"  Population: {args.population}")
    print("  WARNING: These are NOT Jetson Orin Nano latency numbers")
    print("=" * 78)

    # Collect test set
    gallery, query, population_manifest = collect_gallery_and_query(
        args.market1501, population=args.population, id_min=args.id_min, id_max=args.id_max)

    all_results = []

    # Evaluate TransReID
    from edgesync_identity.models.extractor_factory import create_extractor
    logger.info("Loading TransReID extractor...")
    transreid = create_extractor(
        'transreid',
        checkpoint_path=args.transreid_checkpoint,
        projection_path=args.transreid_projection,
    )
    results_transreid = evaluate_model('TransReID-ViT-Base/16+SVD', transreid, gallery, query, args.threshold)
    all_results.append(results_transreid)

    # Evaluate OSNet (if checkpoint available)
    if args.osnet_checkpoint and os.path.isfile(args.osnet_checkpoint):
        logger.info("Loading OSNet extractor...")
        osnet = create_extractor(
            'osnet_torch',
            checkpoint_path=args.osnet_checkpoint,
        )
        results_osnet = evaluate_model('OSNet-x1.0', osnet, gallery, query, args.threshold)
        all_results.append(results_osnet)

    # Print comparison table
    print("\n" + "=" * 78)
    print("  COMPARISON TABLE — Offline Market-1501 Evaluation (Frozen tau=0.70)")
    print("=" * 78)
    print(f"  {'Metric':<30} ", end='')
    for r in all_results:
        print(f"  {r['model_name']:<25}", end='')
    print()
    print("  " + "-" * 76)

    metrics = [
        ('Feature Dim', 'feature_dim', '{:d}'),
        ('Rank-1 (%)', 'rank1', '{:.2f}'),
        ('Rank-5 (%)', 'rank5', '{:.2f}'),
        ('Rank-10 (%)', 'rank10', '{:.2f}'),
        ('mAP (%)', 'mAP', '{:.2f}'),
        ('Distractor Rej. (%)', 'distractor_rejection_rate', '{:.2f}'),
        ('Mean Latency (ms)', 'latency_mean_ms', '{:.2f}'),
        ('P95 Latency (ms)', 'latency_p95_ms', '{:.2f}'),
    ]

    for label, key, fmt in metrics:
        print(f"  {label:<30} ", end='')
        for r in all_results:
            print(f"  {_format_metric(fmt, r.get(key, 0)):<25}", end='')
        print()

    print(f"  {'Evaluated gallery':<30} ", end='')
    for r in all_results:
        print(f"  {r['n_gallery_evaluated']} / {r['n_gallery_selected']}"
              f" (excl {r['n_gallery_excluded']})".ljust(25), end='')
    print()
    print(f"  {'Evaluated query':<30} ", end='')
    for r in all_results:
        print(f"  {r['n_query_evaluated']} / {r['n_query_selected']}"
              f" (excl {r['n_query_excluded']})".ljust(25), end='')
    print()
    print(f"  {'Evaluated identities (q/g)':<30} ", end='')
    for r in all_results:
        ident = r['identities_evaluated']
        print(f"  {ident['query']} / {ident['gallery']}".ljust(25), end='')
    print()

    # Save results
    output = {
        'experiment': 'ES-EXP07',
        'evaluation_type': 'Offline Market-1501 evaluation',
        'threshold_tau': args.threshold,
        'note': 'NOT Jetson latency. Development workstation results only.',
        'population': population_manifest,
        'provenance': {
            'dataset_root': os.path.normpath(os.path.abspath(args.market1501)),
            'command': ' '.join(sys.argv),
            'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'python': sys.version.split()[0],
        },
        'results': all_results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()

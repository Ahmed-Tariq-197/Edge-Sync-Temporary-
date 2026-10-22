#!/usr/bin/env python3
"""Task 7 — generate and verify the TransReID embedding arms used by the statistical analysis.

ES-EXP07, charter §6 steps 1-2/9 and §8: this produces the *real* per-image embeddings behind
every Task 7 number, so Rank-1/mAP/FNMR/FMR can be recomputed from data instead of being quoted
from a script run that never persisted its inputs.

Arms (each one a separate ``.npz`` written to --output-dir, which must be OUTSIDE the repo):

  transreid_768_agnostic   post-BNNeck CLS token, before centering/SVD/normalisation
  transreid_512_agnostic   the deployed production arm: centre -> SVD(768->512) -> L2
  transreid_512_sie        same, but with the checkpoint's Side Information Embedding applied
                           (cam_label = camid - 1), i.e. the arm the recorded run used
  osnet_512                the frozen OSNet x1.0 comparison baseline, through the same factory
                           and the same decoded RGB crops. Nothing in OSNet is modified; if it
                           cannot run, the exact failure is recorded and the other arms stand.

Production-path verification (TR-02/TR-06 guardrails, charter §6 step 9): for every image this
script both (a) calls ``TransReIdExtractor.extract_embedding()`` — the function the ROS node calls
— and (b) re-derives the 512-D vector from the 768-D backbone output using the documented math
contract. The two must agree to within --parity-tolerance (float32 rounding only). A disagreement
means the deployed projection, its mean vector, or the normalisation drifted from the contract, so
the run aborts instead of writing embeddings nobody can trust.

Unusable images are EXCLUDED and reported (never zero-filled), and the saved pid/camid arrays are
aligned to the kept rows only.

Latency recorded here is incidental bookkeeping, not a benchmark: this host is a 4 GB RTX 3050
laptop with no documented warm-up/repeat protocol. Jetson Orin Nano numbers are a separate,
hardware-blocked activity (charter §7).

Usage (smoke):
    python scripts/generate_task7_embeddings.py \\
        --market1501 ../edgesync_bringup/data/Market-1501-v15.09.15 \\
        --transreid-checkpoint ../edgesync_bringup/models/transreid/transformer_120.pth \\
        --transreid-projection ../edgesync_bringup/models/transreid/svd_projection_768_to_512_real_v2.npz \\
        --osnet-checkpoint ../edgesync_bringup/models/osnet/osnet_x1_0_market1501.pth \\
        --output-dir /path/outside/repo/es07_embeddings --limit 8

Traces: ISSUE-05 (split provenance), ISSUE-07 (genuine comparison), TR-02, TR-06, TR-07.
"""
import argparse
import hashlib
import importlib.util
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent))

ARMS = ('transreid_768_agnostic', 'transreid_512_agnostic', 'transreid_512_sie', 'osnet_512')

# ES-EXP07 §4.4. analyze_task7_statistics.py compares this manifest key against its own --exp-id
# and aborts on a disagreement, so the label survives the script -> script handoff.
DEFAULT_EXP_ID = 'EXP07'


def _load_sibling_script(name):
    """Import a same-directory script by path (scripts/ is not an importable package)."""
    path = script_dir / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_of(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def project_768_to_512(x768, mean, projection):
    """The documented wire contract, re-derived independently of the extractor."""
    z = (np.asarray(x768, dtype=np.float32) - mean) @ projection
    norm = float(np.linalg.norm(z))
    if not np.all(np.isfinite(z)) or norm <= 1e-6:
        return np.zeros(z.shape[0], dtype=np.float32)
    return (z / norm).astype(np.float32)


class ArmBuffers:
    """Per-arm accumulation of features plus the bookkeeping needed to audit them."""

    def __init__(self, dim):
        self.dim = dim
        self.features = []
        self.paths = []
        self.pids = []
        self.camids = []
        self.latencies_ms = []
        self.excluded = []

    def add(self, feature, path, pid, camid, latency_ms):
        self.features.append(feature)
        self.paths.append(path)
        self.pids.append(int(pid))
        self.camids.append(int(camid))
        self.latencies_ms.append(float(latency_ms))

    def exclude(self, path, pid, camid, reason, stage=None):
        self.excluded.append({'path': path, 'pid': int(pid), 'camid': int(camid),
                              'reason': reason, 'extractor_stage': stage})

    def as_npz_arrays(self):
        features = (np.asarray(self.features, dtype=np.float32) if self.features
                    else np.zeros((0, self.dim), dtype=np.float32))
        return {
            'features': features,
            'paths': np.asarray(self.paths, dtype='U'),
            'pids': np.asarray(self.pids, dtype=np.int32),
            'camids': np.asarray(self.camids, dtype=np.int32),
            'latencies_ms': np.asarray(self.latencies_ms, dtype=np.float64),
        }


def forward_768(extractor, crop, cam_label):
    """Run the extractor's own preprocessing + backbone and return the raw 768-D CLS feature.

    Deliberately uses ``extractor.preprocess`` and ``extractor._model`` so the 768-D arm shares the
    exact normalisation, tensor layout and SIE path of the deployed arm; only the projection is
    re-derived here, which is what makes the parity check meaningful.
    """
    import torch
    tensor = extractor.preprocess(crop)
    cam_tensor = None
    if cam_label is not None:
        cam_tensor = torch.tensor([cam_label], device=extractor._device, dtype=torch.long)
    with torch.no_grad():
        out = extractor._model(tensor, cam_label=cam_tensor)
    x = out.cpu().numpy().flatten().astype(np.float32)
    if not np.all(np.isfinite(x)):
        raise FloatingPointError('non-finite 768-D backbone output')
    return x


def parity_due(n, every):
    """Which images get the production-vs-re-derived comparison (every Nth, plus the first)."""
    return n == 1 or (n - 1) % every == 0


def generate(split, items, arms, extractor, mean, projection, parity, osnet=None,
             parity_every=1, log_every=500):
    """Extract every selected arm over one split, excluding and recording anything unusable.

    ``arms`` maps arm name -> that split's ``ArmBuffers``; only the names actually requested are
    present, so a partial run (e.g. TransReID arms only) never touches a missing buffer.
    Returns the number of images on which the production path was cross-checked.
    """
    import cv2

    parity_checked = 0

    def add(name, vec, path, pid, camid, latency_ms):
        buf = arms.get(name)
        if buf is not None:
            buf.add(vec, path, pid, camid, latency_ms)

    def exclude_all(path, pid, camid, reason, stage=None):
        for buf in arms.values():
            buf.exclude(path, pid, camid, reason, stage)

    for n, (path, pid, camid) in enumerate(items, start=1):
        img = cv2.imread(path)
        if img is None:
            exclude_all(path, pid, camid, 'decode_failed')
            continue
        crop = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cam_label = camid - 1 if 1 <= camid <= 6 else None

        try:
            t0 = time.perf_counter()
            x768 = forward_768(extractor, crop, None)
            dt = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:  # noqa: BLE001 - recorded, never substituted
            failure = getattr(extractor, 'last_failure', None)
            stage = failure.get('stage') if isinstance(failure, dict) else None
            exclude_all(path, pid, camid, f'backbone_failed: {type(exc).__name__}: {exc}', stage)
            continue

        if x768.shape != (768,):
            exclude_all(path, pid, camid, f'backbone_dim_{x768.shape}')
            continue

        derived512 = project_768_to_512(x768, mean, projection)
        if parity_due(n, parity_every):
            parity_checked += 1
            production = np.asarray(extractor.extract_embedding(crop),
                                    dtype=np.float32).reshape(-1)
            diff = (float(np.max(np.abs(production - derived512)))
                    if production.shape == derived512.shape else float('inf'))
            if diff > parity:
                raise RuntimeError(
                    f'PRODUCTION-PATH PARITY VIOLATION at {path}: extract_embedding() disagrees '
                    f'with the documented center->SVD->L2 contract by {diff:.3e} (> {parity:.1e}). '
                    f'Aborting: embeddings produced from a drifting projection are not evidence.')

        try:
            t1 = time.perf_counter()
            x768_sie = forward_768(extractor, crop, cam_label) if cam_label is not None else None
            dt_sie = (time.perf_counter() - t1) * 1000.0
        except Exception:  # noqa: BLE001
            x768_sie, dt_sie = None, None

        add('transreid_768_agnostic', x768, path, pid, camid, dt)
        add('transreid_512_agnostic', derived512, path, pid, camid, dt)
        if 'transreid_512_sie' in arms:
            if x768_sie is None:
                arms['transreid_512_sie'].exclude(
                    path, pid, camid,
                    'no_sie_camera_label_out_of_range_1_6' if cam_label is None
                    else 'sie_forward_failed')
            else:
                add('transreid_512_sie', project_768_to_512(x768_sie, mean, projection),
                    path, pid, camid, dt_sie)

        if osnet is not None and 'osnet_512' in arms:
            t2 = time.perf_counter()
            emb = np.asarray(osnet.extract_embedding(crop), dtype=np.float32).reshape(-1)
            dt_os = (time.perf_counter() - t2) * 1000.0
            if (emb.shape != (512,) or not np.all(np.isfinite(emb))
                    or float(np.linalg.norm(emb)) < 1e-6):
                arms['osnet_512'].exclude(path, pid, camid,
                                          f'invalid_osnet_output shape={emb.shape}')
            else:
                add('osnet_512', (emb / np.linalg.norm(emb)).astype(np.float32),
                    path, pid, camid, dt_os)

        if n % log_every == 0 or n == len(items):
            logger.info(f"  {split}: {n}/{len(items)} processed — "
                        + ', '.join(f'{name}:{len(buf.features)}' for name, buf in arms.items()))
    return parity_checked


def main():
    parser = argparse.ArgumentParser(description='Task 7 embedding generation + parity verification.')
    parser.add_argument('--market1501', required=True)
    parser.add_argument('--transreid-checkpoint', required=True)
    parser.add_argument('--transreid-projection', required=True)
    parser.add_argument('--osnet-checkpoint', default='')
    parser.add_argument('--output-dir', required=True,
                        help='Must be OUTSIDE the repository; large arrays are not project files.')
    parser.add_argument('--population', default='directory')
    parser.add_argument('--id-min', type=int, default=752)
    parser.add_argument('--id-max', type=int, default=1501)
    parser.add_argument('--limit', type=int, default=0,
                        help='Keep only the first N images per split (smoke runs). 0 = all.')
    parser.add_argument('--parity-tolerance', type=float, default=2e-5)
    parser.add_argument('--parity-every', type=int, default=1,
                        help='Cross-check the production extract_embedding() against the '
                             're-derived contract on every Nth image (1 = every image, the '
                             'default and the value used for the recorded full run).')
    parser.add_argument('--arms', default=','.join(ARMS),
                        help='Comma-separated subset of ' + ','.join(ARMS))
    parser.add_argument('--exp-id', default=DEFAULT_EXP_ID,
                        help='Experiment identifier recorded in the embeddings manifest and '
                             'verified by analyze_task7_statistics.py (ES-EXP07 §4.4). Use a '
                             'distinct value for each extraction run that feeds a report.')
    args = parser.parse_args()

    exp_id = str(args.exp_id or '').strip()
    if not exp_id:
        parser.error('--exp-id cannot be empty: the manifest must carry the identifier that the '
                     'analysis step compares against')

    selected = [a.strip() for a in args.arms.split(',') if a.strip()]
    unknown = sorted(set(selected) - set(ARMS))
    if unknown:
        parser.error(f'unknown arm(s): {unknown}; choose from {list(ARMS)}')

    out_dir = os.path.abspath(args.output_dir)
    repo_root = os.path.abspath(os.path.join(script_dir, *[os.pardir] * 5))
    if os.path.normcase(out_dir).startswith(os.path.normcase(repo_root)):
        parser.error(f'--output-dir must be outside the repository (refusing {out_dir}); '
                     f'embedding arrays are large generated artefacts, not project files.')

    eo = _load_sibling_script('evaluate_offline')
    logger.info(f'EXP-METADATA exp_id={exp_id} arms={selected} output_dir={out_dir}')
    gallery, query, population_manifest = eo.collect_gallery_and_query(
        args.market1501, population=args.population, id_min=args.id_min, id_max=args.id_max)
    if args.limit:
        gallery, query = gallery[:args.limit], query[:args.limit]
        population_manifest['limit_applied'] = args.limit

    from edgesync_identity.models.extractor_factory import create_extractor
    extractor = create_extractor('transreid', checkpoint_path=args.transreid_checkpoint,
                                 projection_path=args.transreid_projection)
    mean = extractor._svd_mean
    projection = extractor._svd_projection

    osnet, osnet_failure = None, None
    if not args.osnet_checkpoint:
        osnet_failure = 'no --osnet-checkpoint supplied'
    elif not os.path.isfile(args.osnet_checkpoint):
        osnet_failure = f'checkpoint file not found: {args.osnet_checkpoint}'
    else:
        # Always attempted, even when the arm is not selected: the outcome is provenance for the
        # TransReID-vs-OSNet comparison. OSNet code is used exactly as it ships (charter §3B).
        try:
            osnet = create_extractor('osnet_torch', checkpoint_path=args.osnet_checkpoint)
        except Exception as exc:  # noqa: BLE001 - charter §3B: record the failure, do not repair
            osnet_failure = f'{type(exc).__name__}: {exc}'
            logger.error(f'OSNet factory construction failed, recorded not repaired: '
                         f'{osnet_failure}')

    dims = {'transreid_768_agnostic': 768, 'transreid_512_agnostic': 512,
            'transreid_512_sie': 512, 'osnet_512': 512}
    buffers = {name: {'gallery': ArmBuffers(dims[name]), 'query': ArmBuffers(dims[name])}
               for name in selected}

    started = time.time()
    parity_counts = {}
    for split, items in (('gallery', gallery), ('query', query)):
        logger.info(f'--- {split}: {len(items)} images ---')
        parity_counts[split] = generate(
            split, items, {name: buffers[name][split] for name in selected},
            extractor, mean, projection, args.parity_tolerance, osnet,
            parity_every=args.parity_every)
    elapsed = time.time() - started

    import torch
    manifest = {
        'exp_id': exp_id,
        'experiment': 'ES-EXP07 Task 7 embeddings',
        'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'elapsed_s': round(elapsed, 1),
        'host': {'platform': platform.platform(), 'python': sys.version.split()[0],
                 'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(),
                 'device_name': (torch.cuda.get_device_name(0) if torch.cuda.is_available()
                                 else 'cpu (no CUDA device present)')},
        'assets': {
            'transreid_checkpoint': {'path': os.path.normpath(os.path.abspath(args.transreid_checkpoint)),
                                     'bytes': os.path.getsize(args.transreid_checkpoint),
                                     'sha256': sha256_of(args.transreid_checkpoint)},
            'transreid_projection': {'path': os.path.normpath(os.path.abspath(args.transreid_projection)),
                                     'bytes': os.path.getsize(args.transreid_projection),
                                     'sha256': sha256_of(args.transreid_projection)},
            'osnet_checkpoint': ({'path': os.path.normpath(os.path.abspath(args.osnet_checkpoint)),
                                  'bytes': os.path.getsize(args.osnet_checkpoint),
                                  'sha256': sha256_of(args.osnet_checkpoint)}
                                 if args.osnet_checkpoint and os.path.isfile(args.osnet_checkpoint)
                                 else None),
        },
        'extractor_checkpoint_coverage': extractor.checkpoint_coverage,
        'extractor_failure_counts': extractor.failure_counts,
        'extractor_embedding_count': extractor.embedding_count,
        'extractor_last_failure': extractor.last_failure,
        'population': population_manifest,
        'parity_tolerance': args.parity_tolerance,
        'parity_check': {
            'images_cross_checked': parity_counts,
            'every_nth': args.parity_every,
            'result': 'PASS — for every image checked, the production extract_embedding() output '
                      'equalled the independently re-derived center->SVD->L2 contract within the '
                      'tolerance; a violation aborts the run',
            'note': 'the check compares the deployed extractor against this script\'s own '
                    'implementation of the documented math contract, on the same decoded crop',
        },
        'osnet_factory_attempt': {
            'result': 'LOADED (frozen baseline, unmodified, through the production factory)'
                      if osnet is not None else f'FAILED — recorded, not repaired (charter §3B): {osnet_failure}',
            'effect_on_comparison': (None if osnet is not None else
                                     'no OSNet embeddings => no paired per-query outcomes for the '
                                     'OSNet arm => a paired test (e.g. exact McNemar) is NOT '
                                     'APPLICABLE on this host; the TransReID-vs-OSNet comparison '
                                     'must be reported NOT MEASURED here and re-run where '
                                     'torchreid imports'),
        },
        'osnet_status': ('ARM NOT REQUESTED' if 'osnet_512' not in selected
                         else 'LOADED (frozen baseline, unmodified, through the same factory)'
                         if osnet is not None
                         else 'NOT MEASURED — factory construction failed, see osnet_factory_attempt'),
        'arms': {},
        'latency_note': 'latencies_ms is incidental bookkeeping on a 4 GB laptop GPU without a '
                        'warm-up/repeat protocol. It is NOT a deployment benchmark and must not '
                        'be reported as Jetson Orin Nano latency.',
        'files': {},
    }

    os.makedirs(out_dir, exist_ok=True)
    for name in selected:
        arrays = {split: buffers[name][split].as_npz_arrays() for split in ('gallery', 'query')}
        npz_path = os.path.join(out_dir, f'task7_{name}.npz')
        np.savez_compressed(npz_path, **{'gallery_' + k: v for k, v in arrays['gallery'].items()},
                            **{'query_' + k: v for k, v in arrays['query'].items()})
        combined = np.concatenate([arrays['gallery']['features'], arrays['query']['features']])
        norms = np.linalg.norm(combined, axis=1) if combined.size else np.zeros(0)
        lat_all = np.concatenate([arrays['gallery']['latencies_ms'],
                                  arrays['query']['latencies_ms']])
        manifest['arms'][name] = {
            'dim': int(dims[name]),
            'gallery_kept': int(arrays['gallery']['features'].shape[0]),
            'query_kept': int(arrays['query']['features'].shape[0]),
            'gallery_excluded': len(buffers[name]['gallery'].excluded),
            'query_excluded': len(buffers[name]['query'].excluded),
            'gallery_exclusion_reasons': _tally(buffers[name]['gallery'].excluded),
            'query_exclusion_reasons': _tally(buffers[name]['query'].excluded),
            'gallery_identities': int(len(np.unique(arrays['gallery']['pids']))),
            'query_identities': int(len(np.unique(arrays['query']['pids']))),
            'unit_norm_max_deviation': (None if name == 'transreid_768_agnostic' or norms.size == 0
                                        else float(np.max(np.abs(norms - 1.0)))),
            'mean_norm': (None if name == 'transreid_768_agnostic' or norms.size == 0
                          else float(np.mean(norms))),
            'any_zero_row': bool(combined.size and np.any(np.all(combined == 0.0, axis=-1))),
            'latency_mean_ms': (None if lat_all.size == 0 else float(np.mean(lat_all))),
            'latency_semantics': ('backbone forward only; the 512-D arms reuse the same forward, '
                                  'the SVD matvec is not timed separately'
                                  if name != 'osnet_512' else 'osnet forward only'),
        }
        manifest['files'][name] = {
            'path': npz_path, 'bytes': os.path.getsize(npz_path), 'sha256': sha256_of(npz_path)}
        with open(os.path.join(out_dir, f'task7_{name}_exclusions.json'), 'w') as handle:
            json.dump({'gallery': buffers[name]['gallery'].excluded,
                       'query': buffers[name]['query'].excluded}, handle, indent=2)

    manifest_path = os.path.join(out_dir, 'task7_embeddings_manifest.json')
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2)
    logger.info(json.dumps(manifest['arms'], indent=2))
    logger.info(f'Manifest: {manifest_path}')
    logger.info(f'Elapsed: {elapsed:.1f} s')


def _tally(excluded):
    tally = {}
    for record in excluded:
        tally[record['reason'][:60]] = tally.get(record['reason'][:60], 0) + 1
    return tally


if __name__ == '__main__':
    main()

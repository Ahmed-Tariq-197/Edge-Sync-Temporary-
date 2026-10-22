#!/usr/bin/env python3
"""TR-11b — TransReID latency / throughput / memory benchmark ON THE JETSON ORIN NANO.

Charter §7 makes Jetson execution a required target activity, and §6 forbids reporting a speedup
without measured results. This harness is what produces those measurements on the device, with a
documented repeatable protocol:

  1. record the device and stack (delegated to ``build_transreid_trt_engine.target_environment`` so
     there is one implementation of "what environment was this measured in"), plus power mode and
     clocks before and after the run;
  2. load the engine through ``TransReIdTrtExtractor`` — the production class — so the numbers
     describe the path that actually serves requests, including its preprocessing and projection;
  3. warm up with untimed inferences;
  4. run R repeats of N timed inferences over real decoded crops, recording every single latency
     (no dropping of outliers) plus optional tegrastats sampling;
  5. report per-repeat and cross-repeat statistics: mean, median, p50/p90/p95/p99, min, max, std,
     throughput, and memory;
  6. verify output correctness against the Torch arm on the same crops, and record the tolerance
     used rather than asserting agreement.

Refuses to run on a non-Jetson host unless ``--allow-non-target`` is passed, and then labels the
result ``NOT A JETSON BENCHMARK`` — a laptop benchmark is not a Jetson benchmark (charter §7).

OSNet is only ever loaded if an engine is explicitly supplied (``--osnet-engine``); it is a frozen
comparison baseline (charter §3B) and nothing here repairs or rebuilds it. Without it the
comparison is reported NOT MEASURED.

Usage on the Orin Nano:
    python3 scripts/benchmark_transreid_jetson.py \\
        --engine ../edgesync_bringup/models/engines/transreid_agnostic_orin.engine --arm agnostic \\
        --market1501 ../edgesync_bringup/data/Market-1501-v15.09.15 \\
        --iterations 200 --repeats 3 --warmup 20 --out-dir ~/edgesync_bench
"""
import argparse
import csv
import importlib.util
import json
import logging
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('benchmark_transreid_jetson')

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent))
JETSON_MARKERS = ('/etc/nv_tegra_release', '/proc/device-tree/model', '/etc/nvidia-l4t-version')
DEFAULT_PROJECTION = os.path.abspath(os.path.join(
    script_dir, '..', '..', 'edgesync_bringup', 'models', 'transreid',
    'svd_projection_768_to_512_real_v2.npz'))
DEFAULT_CHECKPOINT = os.path.abspath(os.path.join(
    script_dir, '..', '..', 'edgesync_bringup', 'models', 'transreid', 'transformer_120.pth'))


def _sibling(name):
    spec = importlib.util.spec_from_file_location(name, script_dir / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read(path):
    try:
        with open(path, 'rb') as handle:
            return handle.read().decode('utf-8', 'replace').strip('\x00\n ') or None
    except OSError:
        return None


def _sh(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        return (out.stdout or out.stderr).strip() or f'(no output, exit {out.returncode})'
    except FileNotFoundError:
        return f'NOT AVAILABLE: {cmd[0]} not on PATH'
    except subprocess.SubprocessError as exc:
        return f'FAILED: {type(exc).__name__}: {exc}'


def _mem_snapshot():
    """Host memory plus this process' RSS, read from /proc so no extra dependency is needed."""
    snap = {}
    try:
        with open('/proc/meminfo') as handle:
            for line in handle:
                key, _, rest = line.partition(':')
                if key in ('MemTotal', 'MemAvailable', 'MemFree', 'Buffers', 'Cached',
                           'SwapTotal', 'SwapFree'):
                    snap[key + '_kB'] = int(rest.strip().split()[0])
    except OSError:
        snap['meminfo'] = 'NOT READABLE (this harness expects Linux; run it on the Jetson)'
    try:
        with open(f'/proc/{os.getpid()}/status') as handle:
            for line in handle:
                if line.startswith('VmRSS:'):
                    snap['process_RSS_kB'] = int(line.split()[1])
                    break
    except OSError:
        snap['process_RSS_kB'] = None
    return snap


class Tegrastats(threading.Thread):
    """Sample tegrastats for the duration of the run; the raw lines are kept, not summarised away."""

    def __init__(self, path, interval_ms=1000):
        super().__init__(daemon=True)
        self.path, self.interval_ms = path, interval_ms
        self.lines = []
        self._stop = threading.Event()
        self.error = None

    def run(self):
        try:
            proc = subprocess.Popen(['tegrastats', '--interval', str(self.interval_ms)],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except FileNotFoundError:
            self.error = 'tegrastats not on PATH'
            return
        for line in proc.stdout:
            if self._stop.is_set():
                proc.terminate()
                break
            self.lines.append(line.rstrip())
        with open(self.path, 'w') as handle:
            handle.write('\n'.join(self.lines))

    def stop(self):
        self._stop.set()


def latencies_ms(values):
    """Summary statistics over a latency sample. Every value is kept; nothing is discarded."""
    if not values:
        return {'n': 0}
    ordered = sorted(values)

    def pct(p):
        idx = min(len(ordered) - 1, max(0, int(round(p / 100.0 * (len(ordered) - 1)))))
        return ordered[idx]

    return {'n': len(values), 'mean': statistics.fmean(values),
            'median': statistics.median(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'min': ordered[0], 'max': ordered[-1],
            'p50': pct(50), 'p90': pct(90), 'p95': pct(95), 'p99': pct(99),
            'sum_seconds': sum(values) / 1000.0}


def load_crops(market_root, count):
    """Real decoded RGB crops from the named evaluation population, in deterministic order."""
    eo = _sibling('evaluate_offline')
    import cv2
    gallery, _, _ = eo.collect_gallery_and_query(market_root, population='directory')
    crops, failures = [], []
    for entry in gallery:
        if len(crops) >= count:
            break
        path = entry[0]
        bgr = cv2.imread(path)
        if bgr is None:
            failures.append({'path': path, 'reason': 'decode_failed'})
            continue
        crops.append((path, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    return crops, failures


def timed_run(extractor, crops, cam_label, warmup, iterations, repeat_index, outliers):
    for _ in range(warmup):
        extractor.extract_embedding(crops[0][1], cam_label=cam_label)
    per_call = []
    for i in range(iterations):
        path, crop = crops[i % len(crops)]
        t0 = time.perf_counter()
        emb = extractor.extract_embedding(crop, cam_label=cam_label)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_call.append(elapsed_ms)
        if np.all(np.asarray(emb) == 0.0):
            outliers.append({'repeat': repeat_index, 'call': i, 'image': path,
                             'reason': 'sentinel embedding returned',
                             'failure': getattr(extractor, 'last_failure', None)})
    return per_call


def main():
    parser = argparse.ArgumentParser(description='TR-11b TransReID Jetson benchmark harness.')
    parser.add_argument('--engine', required=True)
    parser.add_argument('--arm', required=True, choices=['agnostic', 'sie'])
    parser.add_argument('--projection', default=DEFAULT_PROJECTION)
    parser.add_argument('--market1501', required=True,
                        help='Real crops are required; without them this is not a benchmark.')
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--crop-count', type=int, default=64)
    parser.add_argument('--correctness-n', type=int, default=8)
    parser.add_argument('--tolerance', type=float, default=2e-2)
    parser.add_argument('--tegrastats', action='store_true', default=True)
    parser.add_argument('--osnet-engine', default='',
                        help='Optional frozen-baseline engine for a like-for-like number. Nothing '
                             'here builds or repairs it (charter §3B).')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--allow-non-target', action='store_true')
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    build = _sibling('build_transreid_trt_engine')
    env = build.target_environment()
    report = {'experiment': 'ES-EXP07 TR-11b Jetson Orin Nano benchmark',
              'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'protocol': {'warmup_inferences': args.warmup, 'timed_inferences_per_repeat':
                           args.iterations, 'repeats': args.repeats, 'batch': 1,
                           'input_dims': '1x3x256x128 float32 (NCHW, mean=std=[0.5,0.5,0.5])',
                           'crop_source': f'{args.market1501} (population: directory)',
                           'timing': 'time.perf_counter around extract_embedding(), i.e. '
                                     'preprocess + engine + projection + normalise — the production '
                                     'call, not the kernel alone',
                           'outliers': 'kept and listed; none are discarded from the statistics'},
              'environment': env,
              'power_before': {'nvpmodel': env.get('nvpmodel'), 'jetson_clocks': env.get('jetson_clocks')},
              'memory_before': _mem_snapshot(),
              'platform': {'machine': platform.machine(), 'python': platform.python_version()}}

    if not env['is_jetson_marker_present'] and not args.allow_non_target:
        report['verdict'] = ('BLOCKED — Pending Jetson Orin Nano access: none of '
                             + ', '.join(JETSON_MARKERS) + ' exist on this host. Charter §7: a '
                             'laptop benchmark is not a Jetson benchmark.')
        _write(report, out_dir)
        raise SystemExit(report['verdict'])
    if not env['is_jetson_marker_present']:
        report['non_target'] = 'NOT A JETSON BENCHMARK — run with --allow-non-target'

    from edgesync_identity.models.transreid_trt_extractor import TransReIdTrtExtractor
    try:
        extractor = TransReIdTrtExtractor(engine_path=args.engine, projection_path=args.projection)
    except Exception as exc:  # noqa: BLE001 — load failure is the headline finding
        report['engine_load'] = f'FAILED: {type(exc).__name__}: {exc}'
        report['verdict'] = ('BLOCKED — the engine could not be loaded on this host; nothing was '
                             'timed and no latency is reported')
        _write(report, out_dir)
        raise SystemExit(report['verdict'])
    if extractor.has_cam_input != (args.arm == 'sie'):
        report['engine_load'] = ('ARM MISMATCH — engine has_cam_input='
                                 f'{extractor.has_cam_input} but --arm={args.arm}')
        _write(report, out_dir)
        raise SystemExit(report['engine_load'])
    report['engine_load'] = {'status': 'OK', 'class': type(extractor).__name__,
                             'engine_metadata': extractor.engine_metadata,
                             'production_path': 'preprocess + engine + SVD projection + L2, '
                                                'through the deployed extractor class'}

    crops, decode_failures = load_crops(args.market1501, args.crop_count)
    report['crops'] = {'loaded': len(crops), 'decode_failures': decode_failures[:20],
                       'first': crops[0][0] if crops else None}
    if not crops:
        report['verdict'] = 'FAILED — no real crops could be decoded; refusing to benchmark noise'
        _write(report, out_dir)
        raise SystemExit(report['verdict'])

    cam = 0 if args.arm == 'sie' else None
    tegra = Tegrastats(os.path.join(out_dir, 'tegrastats.log')) if args.tegrastats else None
    if tegra:
        tegra.start()

    repeats, outliers = [], []
    for r in range(args.repeats):
        values = timed_run(extractor, crops, cam, args.warmup, args.iterations, r, outliers)
        with open(os.path.join(out_dir, f'latencies_{args.arm}_repeat{r}.csv'), 'w',
                  newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['repeat', 'call', 'image', 'latency_ms'])
            for i, v in enumerate(values):
                writer.writerow([r, i, crops[i % len(crops)][0], f'{v:.4f}'])
        repeats.append({'repeat': r, 'latencies_ms': latencies_ms(values),
                        'memory_after_repeat': _mem_snapshot()})
        logger.info(f'repeat {r}: mean {statistics.fmean(values):.2f} ms, '
                    f'p95 {latencies_ms(values)["p95"]:.2f} ms')

    if tegra:
        tegra.stop()
        tegra.join(timeout=5)

    per_means = [r['latencies_ms']['mean'] for r in repeats if r['latencies_ms']['n']]
    report['repeats'] = repeats
    report['summary'] = {
        'arm': args.arm, 'engine': os.path.abspath(args.engine),
        'repeats_completed': len(repeats),
        'mean_of_repeat_means_ms': statistics.fmean(per_means) if per_means else None,
        'spread_of_repeat_means_ms': (max(per_means) - min(per_means)) if per_means else None,
        'throughput_embeddings_per_second': (1000.0 / statistics.fmean(per_means)
                                             if per_means else None),
        'latency_semantics': 'end-to-end production call including preprocessing, engine execution, '
                             'SVD projection and L2 normalisation',
        'failures': dict(extractor.failure_counts),
        'embeddings_produced': extractor.embedding_count,
        'outliers': outliers,
        'crop_decode_failures': len(decode_failures)}
    report['memory_after'] = _mem_snapshot()
    report['power_after'] = {'nvpmodel': _sh(['nvpmodel', '-q']),
                             'jetson_clocks': _sh(['jetson_clocks', '--show'])}
    if tegra:
        report['tegrastats'] = {'lines': len(tegra.lines), 'path': os.path.join(out_dir, 'tegrastats.log'),
                                'error': tegra.error}

    # Output correctness on the device, against the Torch arm on the same crops (§6 step 7).
    if os.path.isfile(DEFAULT_CHECKPOINT):
        try:
            from edgesync_identity.models.transreid_extractor import TransReIdExtractor
            torch_arm = TransReIdExtractor(checkpoint_path=DEFAULT_CHECKPOINT,
                                           projection_path=args.projection)
            rows = []
            for path, crop in crops[:args.correctness_n]:
                a = np.asarray(extractor.extract_embedding(crop, cam_label=cam), np.float32)
                b = np.asarray(torch_arm.extract_embedding(crop, cam_label=cam), np.float32)
                if np.all(a == 0.0) or np.all(b == 0.0):
                    rows.append({'image': path, 'status': 'sentinel',
                                 'engine_failure': getattr(extractor, 'last_failure', None)})
                    continue
                denom = float(np.linalg.norm(a) * np.linalg.norm(b))
                rows.append({'image': path, 'cosine': float(np.dot(a, b) / denom) if denom else None,
                             'max_abs': float(np.abs(a - b).max()),
                             'l2_dist': float(np.linalg.norm(a - b)),
                             'within_tolerance': bool(np.abs(a - b).max() <= args.tolerance)})
            deltas = [r['max_abs'] for r in rows if 'max_abs' in r]
            report['correctness_vs_torch'] = {
                'n': len(rows), 'images': rows, 'tolerance': args.tolerance,
                'worst_max_abs': max(deltas) if deltas else None,
                'worst_cosine': min((r['cosine'] for r in rows if r.get('cosine') is not None),
                                    default=None),
                'verdict': ('PASS' if deltas and max(deltas) <= args.tolerance
                            else 'FAIL' if deltas else 'NOT MEASURED')}
        except Exception as exc:  # noqa: BLE001
            report['correctness_vs_torch'] = f'NOT MEASURED — Torch reference unavailable: ' \
                                             f'{type(exc).__name__}: {exc}'
    else:
        report['correctness_vs_torch'] = f'NOT MEASURED — {DEFAULT_CHECKPOINT} not present'

    if args.osnet_engine:
        try:
            from edgesync_identity.models.osnet_trt_extractor import OsnetTrtExtractor
            baseline = OsnetTrtExtractor(engine_path=args.osnet_engine)
            values = timed_run(baseline, crops, None, args.warmup, args.iterations, 0, outliers)
            report['osnet_baseline'] = {
                'status': 'measured as-is through the frozen OSNet arm (charter §3B: inspected, '
                          'not modified or repaired)', 'latencies_ms': latencies_ms(values),
                'note': 'OSNet x1_0 is a much smaller network; a raw latency comparison without '
                        'accuracy context is not a quality judgement'}
        except Exception as exc:  # noqa: BLE001
            report['osnet_baseline'] = f'FAILED, recorded not repaired: {type(exc).__name__}: {exc}'
    else:
        report['osnet_baseline'] = ('NOT MEASURED — no --osnet-engine supplied. OSNet is a frozen '
                                    'baseline (charter §3B) and its engine is not rebuilt here.')

    report['not_claimed'] = {
        'speedup': 'NOT CLAIMED — a speedup requires a valid comparison arm measured on the same '
                   'device with the same protocol',
        'accuracy_parity': 'see correctness_vs_torch; retrieval accuracy on-device is a separate '
                           'evaluation, not a latency benchmark',
        'cross_robot_handoff': 'NOT MEASURED and NOT inferable from latency or Market-1501 retrieval'}
    report['verdict'] = ('COMPLETE — real crops, warm-up, repeats and per-call latencies recorded on '
                         'a Jetson-marked host' if env['is_jetson_marker_present']
                         else 'NOT A JETSON BENCHMARK — see the non_target flag')
    _write(report, out_dir)
    logger.info(json.dumps(report['summary'], indent=2, default=str)[:1500])
    logger.info(f'Report: {os.path.join(out_dir, "jetson_benchmark.json")}')


def _write(report, out_dir):
    path = os.path.join(out_dir, 'jetson_benchmark.json')
    with open(path, 'w') as handle:
        json.dump(report, handle, indent=2, default=str)


if __name__ == '__main__':
    main()

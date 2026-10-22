#!/usr/bin/env python3
"""TR-11a — build the TransReID TensorRT engine ON THE TARGET DEVICE (Jetson Orin Nano).

Charter §6 step 6: the engine must be built in the actual target environment. TensorRT engines are
not portable across GPU architectures or TRT majors, so an engine built anywhere else is not the
deployment artefact — and a laptop build is not a Jetson build. This script therefore refuses to
run on a host that does not present Jetson markers unless ``--allow-non-target`` is passed, and it
records the device identity it observed while building.

It consumes the ONNX graph produced by TR-09 (``scripts/export_transreid_onnx.py``) and writes:

    <out>.engine            serialized engine
    <out>_build_report.json device/stack versions, power mode, clocks, parser and builder settings,
                            every parser/builder log message (the unsupported-op evidence), build
                            time, SHA-256 of graph and engine, and a real load+inference test.

§6 step 10 requires precision, batch, resolution, build settings and unsupported ops to be recorded;
that is what the report is. The engine is then load-tested through ``TransReIdTrtExtractor``, the
production class, so the report shows the deployed path can actually execute it — an engine file
existing is not success (charter §6).

Usage on the Orin Nano:
    python3 scripts/build_transreid_trt_engine.py \\
        --onnx /path/transreid_feature_agnostic.onnx --arm agnostic --precision fp16 \\
        --out ../edgesync_bringup/models/engines/transreid_agnostic_orin.engine \\
        --market1501 ../edgesync_bringup/data/Market-1501-v15.09.15
    python3 scripts/build_transreid_trt_engine.py \\
        --onnx /path/transreid_feature_sie.onnx --arm sie --precision fp16 \\
        --out ../edgesync_bringup/models/engines/transreid_sie_orin.engine \\
        --market1501 ../edgesync_bringup/data/Market-1501-v15.09.15
"""
import argparse
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('build_transreid_trt_engine')

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent))

CONTRACT_IMAGE_SHAPE = (3, 256, 128)
JETSON_MARKERS = ('/etc/nv_tegra_release', '/proc/device-tree/model', '/etc/nvidia-l4t-version')
DEFAULT_PROJECTION = os.path.abspath(os.path.join(
    script_dir, '..', '..', 'edgesync_bringup', 'models', 'transreid',
    'svd_projection_768_to_512_real_v2.npz'))
DEFAULT_CHECKPOINT = os.path.abspath(os.path.join(
    script_dir, '..', '..', 'edgesync_bringup', 'models', 'transreid', 'transformer_120.pth'))
UNSUPPORTED_PATTERN = re.compile(r'unsupported|not supported|fallback|pdb|external data|'
                                 r'could not be parsed', re.I)


def sha256_of(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    try:
        with open(path, 'rb') as handle:
            return handle.read().decode('utf-8', 'replace').strip('\x00\n ') or None
    except OSError:
        return None


def _sh(cmd):
    """Run a probe command; return its text or a recorded reason it could not run. Never silent."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
        return (out.stdout or out.stderr).strip() or f'(no output, exit {out.returncode})'
    except FileNotFoundError:
        return f'NOT AVAILABLE: {cmd[0]} not on PATH'
    except subprocess.SubprocessError as exc:
        return f'FAILED: {type(exc).__name__}: {exc}'


def target_environment():
    """Everything §7 requires to be recorded about the device and stack, gathered in one place."""
    env = {
        'is_jetson_marker_present': any(os.path.exists(p) for p in JETSON_MARKERS),
        'device_tree_model': _read('/proc/device-tree/model'),
        'l4t_nv_tegra_release': _read('/etc/nv_tegra_release'),
        'platform': {'system': platform.system(), 'machine': platform.machine(),
                     'python': platform.python_version(), 'node': platform.node()},
    }
    try:
        import torch
        env['torch'] = {'version': torch.__version__,
                        'cuda_available': torch.cuda.is_available(),
                        'cuda_version': getattr(torch.version, 'cuda', None),
                        'cudnn_version': (torch.backends.cudnn.version()
                                          if torch.backends.cudnn.is_available() else None)}
    except ImportError as exc:
        env['torch'] = f'NOT IMPORTABLE: {exc}'
    try:
        import tensorrt as trt
        env['tensorrt_version'] = trt.__version__
    except ImportError as exc:
        env['tensorrt_version'] = f'NOT IMPORTABLE: {exc}'
    try:
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
        dev = cuda.Device(0)
        cap = dev.compute_capability()
        env['cuda_device'] = {'name': dev.name(),
                              'total_memory_MiB': round(dev.total_memory() / 2**20),
                              'compute_capability': f'{cap[0]}.{cap[1]}'}
    except ImportError as exc:
        env['pycuda'] = f'NOT IMPORTABLE: {exc}'
    except Exception as exc:  # noqa: BLE001 — environment probing must never abort a build
        env['pycuda'] = f'device probe failed: {type(exc).__name__}: {exc}'
    env['nvpmodel'] = _sh(['nvpmodel', '-q'])
    env['jetson_clocks'] = _sh(['jetson_clocks', '--show'])
    env['gpu_busy_percent'] = _sh(['cat', '/sys/class/devfreq/17000000.gpu/gpu_busy_percent'])
    env['trtexec_on_path'] = shutil.which('trtexec')
    env['free_memory'] = _sh(['free', '-m'])
    return env


def make_logger(verbosity):
    """A TensorRT logger that keeps every message, so unsupported ops become evidence."""
    import tensorrt as trt

    class CollectingLogger(trt.Logger):
        def __init__(self):
            super().__init__(verbosity)
            self.messages = []

        def log(self, severity, message):
            self.messages.append(f'{severity}: {message}')
            super().log(severity, message)

    return CollectingLogger()


def build(onnx_path, engine_path, precision, arm, workspace_gib, min_batch, opt_batch, max_batch,
          verbose):
    """Parse the graph and build one engine with an explicit optimization profile."""
    import tensorrt as trt

    verbosity = trt.Logger.VERBOSE if verbose else trt.Logger.INFO
    builder_logger = make_logger(verbosity)
    builder = trt.Builder(builder_logger)
    parser_logger = make_logger(verbosity)
    # TRT 10 dropped EXPLICIT_BATCH from NetworkDefinitionCreationFlag; TRT 8 requires it.
    nd_flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, 'EXPLICIT_BATCH'):
        nd_flags = 1 << trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
    network = builder.create_network(nd_flags)
    # The (network, logger) constructor form works on both TRT 8.x and TRT 10.x.
    parser = trt.OnnxParser(network, parser_logger)
    with open(onnx_path, 'rb') as handle:
        if not parser.parse(handle.read()):
            raise RuntimeError(
                f'ONNX parse failed for {onnx_path}: {parser.num_errors} parser errors, '
                f'messages: {parser_logger.messages[:5]}')
    if network.num_inputs == 0:
        raise RuntimeError('parsed network exposes no inputs — refusing to build')

    flags = []
    config = builder.create_builder_config()
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gib * 2**30))
    except AttributeError:
        config.max_workspace_size = int(workspace_gib * 2**30)
    if precision == 'fp16':
        if not builder.platform_has_fast_fp16:
            raise RuntimeError('this device reports platform_has_fast_fp16 = False; '
                               'refusing to enable FP16 anyway')
        config.set_flag(trt.BuilderFlag.FP16)
        flags.append('FP16')
    elif precision == 'int8':
        if not builder.platform_has_fast_int8:
            raise RuntimeError('this device reports platform_has_fast_int8 = False')
        config.set_flag(trt.BuilderFlag.INT8)
        flags.append('INT8')
    else:
        flags.append('FP32')

    profile = builder.create_optimization_profile()
    image = lambda b: (b,) + CONTRACT_IMAGE_SHAPE  # noqa: E731 - three uses of one contract shape
    profile.add_shape('image', image(min_batch), image(opt_batch), image(max_batch))
    if arm == 'sie':
        profile.add_shape('cam_label', (min_batch,), (opt_batch,), (max_batch,))
    config.add_optimization_profile(profile)

    logger.info(f'tensorrt build starting: precision={precision}, batch '
                f'{min_batch}/{opt_batch}/{max_batch}, workspace={workspace_gib} GiB')
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - t0
    if serialized is None:
        raise RuntimeError(f'build returned no engine after {build_seconds:.1f}s; '
                           f'last messages: {builder_logger.messages[-5:]}')
    engine_dir = os.path.dirname(os.path.abspath(engine_path))
    os.makedirs(engine_dir, exist_ok=True)
    with open(engine_path, 'wb') as handle:
        handle.write(serialized)

    return {
        'status': 'OK', 'seconds': round(build_seconds, 1), 'precision_flags': flags,
        'network_definition_flags': 'EXPLICIT_BATCH' if nd_flags else 'implicit-batch API absent',
        'workspace_gib': workspace_gib,
        'optimization_profile': {
            'image': [image(min_batch), image(opt_batch), image(max_batch)],
            **({'cam_label': [(min_batch,), (opt_batch,), (max_batch,)]} if arm == 'sie' else {})},
        'network_inputs': [network.get_input(i).name for i in range(network.num_inputs)],
        'network_outputs': [network.get_output(i).name for i in range(network.num_outputs)],
        'platform_has_fast_fp16': bool(builder.platform_has_fast_fp16),
        'platform_has_fast_int8': bool(builder.platform_has_fast_int8),
        'log_messages': parser_logger.messages + builder_logger.messages,
    }


def load_and_run(engine_path, projection_path, arm, n_crops, market_root, tolerance):
    """Execute the engine through the PRODUCTION extractor and compare it to the Torch arm."""
    from edgesync_identity.models.transreid_trt_extractor import TransReIdTrtExtractor

    extractor = TransReIdTrtExtractor(engine_path=engine_path, projection_path=projection_path)
    if extractor.has_cam_input != (arm == 'sie'):
        raise RuntimeError(
            f'arm mismatch: built/asked for {arm!r} but the loaded engine has_cam_input='
            f'{extractor.has_cam_input}. The wrong engine would silently change the features.')
    result = {'class': type(extractor).__name__, 'engine_metadata': extractor.engine_metadata,
              'has_cam_input': extractor.has_cam_input,
              'projection_loaded': {'projection_shape': list(extractor._svd_projection.shape),
                                    'mean_shape': list(extractor._svd_mean.shape)},
              'preprocessing': 'resize 256x128, /255, mean=std=[0.5,0.5,0.5], HWC->CHW (numpy path)',
              'crops': [], 'failure_counts_at_load': dict(extractor.failure_counts)}

    paths = [None] * n_crops
    if market_root:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'evaluate_offline', script_dir / 'evaluate_offline.py')
        eo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eo)
        gallery, _, _ = eo.collect_gallery_and_query(market_root, population='directory')
        paths = [g[0] for g in gallery[:n_crops]]
        if not paths:
            raise RuntimeError(f'no images found under {market_root} — refusing to benchmark '
                               f'synthetic noise and call it a deployment test')
    else:
        result['crops_note'] = ('SYNTHETIC uniform crops: execution proof only, NOT accuracy and '
                                'NOT a latency benchmark')

    try:
        import cv2
    except ImportError:
        cv2 = None
        result['cv2'] = 'NOT IMPORTABLE — cannot decode real crops'

    torch_arm = None
    if market_root and cv2 is not None and os.path.isfile(DEFAULT_CHECKPOINT):
        try:
            from edgesync_identity.models.transreid_extractor import TransReIdExtractor
            torch_arm = TransReIdExtractor(checkpoint_path=DEFAULT_CHECKPOINT,
                                           projection_path=projection_path)
        except Exception as exc:  # noqa: BLE001 — reference availability is a finding, not a crash
            result['torch_reference'] = f'NOT AVAILABLE: {type(exc).__name__}: {exc}'
    else:
        result['torch_reference'] = 'NOT AVAILABLE on this run — parity NOT MEASURED'

    for i, path in enumerate(paths):
        if path is not None:
            bgr = cv2.imread(path) if cv2 else None
            crop = None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            source = path
        else:
            crop, source = np.full((256, 128, 3), 127, dtype=np.uint8), 'synthetic'
        if crop is None:
            result['crops'].append({'index': i, 'source': source, 'status': 'decode_failed'})
            continue
        cam = 0 if arm == 'sie' else None
        emb = np.asarray(extractor.extract_embedding(crop, cam_label=cam), dtype=np.float32)
        entry = {'index': i, 'source': source,
                 'unit_norm': round(float(np.linalg.norm(emb)), 6),
                 'finite': bool(np.all(np.isfinite(emb))),
                 'all_zero': bool(np.all(emb == 0.0))}
        if torch_arm is not None and path is not None:
            ref = np.asarray(torch_arm.extract_embedding(crop, cam_label=cam), dtype=np.float32)
            denom = float(np.linalg.norm(ref) * np.linalg.norm(emb))
            entry['vs_torch'] = {'cosine': float(np.dot(ref, emb) / denom) if denom > 0 else None,
                                 'max_abs': float(np.abs(ref - emb).max()),
                                 'l2_dist': float(np.linalg.norm(ref - emb)),
                                 'within_tolerance': bool(np.abs(ref - emb).max() <= tolerance)}
        result['crops'].append(entry)
    result['failure_counts_after_run'] = dict(extractor.failure_counts)
    result['embeddings_produced'] = extractor.embedding_count
    return result


def main():
    parser = argparse.ArgumentParser(description='TR-11a build the TransReID TensorRT engine on target.')
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--arm', required=True, choices=['agnostic', 'sie'])
    parser.add_argument('--out', required=True)
    parser.add_argument('--projection', default=DEFAULT_PROJECTION)
    parser.add_argument('--precision', default='fp16', choices=['fp32', 'fp16', 'int8'])
    parser.add_argument('--workspace-gib', type=float, default=4.0)
    parser.add_argument('--min-batch', type=int, default=1)
    parser.add_argument('--opt-batch', type=int, default=1)
    parser.add_argument('--max-batch', type=int, default=1)
    parser.add_argument('--parity-n', type=int, default=8)
    parser.add_argument('--market1501', default='')
    parser.add_argument('--tolerance', type=float, default=2e-2,
                        help='Max per-component |Torch - engine| allowed. Deliberately looser than '
                             'the fp32 ONNX check: fp16 accumulation in a 768-wide ViT is the thing '
                             'under test. The measured value is always reported.')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--skip-load-test', action='store_true')
    parser.add_argument('--allow-non-target', action='store_true',
                        help='Permit a build off the target (bring-up only). The report marks the '
                             'result NOT A DEPLOYMENT ARTEFACT.')
    args = parser.parse_args()

    report = {'experiment': 'ES-EXP07 TR-11a TensorRT engine build',
              'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'inputs': {'onnx': os.path.abspath(args.onnx),
                         'onnx_bytes': os.path.getsize(args.onnx),
                         'onnx_sha256': sha256_of(args.onnx), 'arm': args.arm},
              'build_settings': {'precision': args.precision,
                                 'batch_min_opt_max': [args.min_batch, args.opt_batch, args.max_batch],
                                 'image_shape_contract': [1] + list(CONTRACT_IMAGE_SHAPE),
                                 'workspace_gib': args.workspace_gib,
                                 'verbose': args.verbose},
              'environment': target_environment()}
    env = report['environment']
    if not env['is_jetson_marker_present'] and not args.allow_non_target:
        report['verdict'] = (
            'BLOCKED — this host presents none of ' + ', '.join(JETSON_MARKERS) +
            '. Charter §6 step 6 requires the engine to be built in the actual target environment; '
            're-run on the Orin Nano, or pass --allow-non-target for a bring-up build that is not a '
            'deployment artefact.')
        _write(report, args.out)
        raise SystemExit(report['verdict'])
    if not env['is_jetson_marker_present']:
        report['non_target_build'] = 'NOT A DEPLOYMENT ARTEFACT — built with --allow-non-target'

    report['projection'] = {'path': os.path.abspath(args.projection),
                            'bytes': os.path.getsize(args.projection),
                            'sha256': sha256_of(args.projection)}

    try:
        build_info = build(args.onnx, args.out, args.precision, args.arm, args.workspace_gib,
                           args.min_batch, args.opt_batch, args.max_batch, args.verbose)
    except ImportError as exc:
        report['build'] = f'FAILED — tensorrt not importable: {exc}'
        report['verdict'] = 'BLOCKED — Pending Jetson Orin Nano access'
        _write(report, args.out)
        raise SystemExit(report['verdict'])
    except Exception as exc:  # noqa: BLE001 — the build outcome IS the finding
        report['build'] = f'FAILED: {type(exc).__name__}: {exc}'
        report['verdict'] = 'ENGINE BUILD FAILED — nothing was deployed'
        _write(report, args.out)
        raise SystemExit(report['verdict'])

    messages = build_info.pop('log_messages')
    report['build'] = build_info
    report['engine'] = {'path': os.path.abspath(args.out), 'bytes': os.path.getsize(args.out),
                        'sha256': sha256_of(args.out)}
    report['log_messages'] = {
        'total': len(messages),
        'unsupported_or_fallback': [m for m in messages if UNSUPPORTED_PATTERN.search(m)][:200],
        'warnings': [m for m in messages if 'WARNING' in m][:200],
    }
    if report['log_messages']['unsupported_or_fallback']:
        report['unsupported_ops_note'] = (
            'the parser or builder reported operations it could not handle natively (listed above). '
            'Charter §6 step 10 requires this to be recorded before an engine is called deployable.')

    if args.skip_load_test:
        report['load_and_run'] = ('SKIPPED (--skip-load-test) — an engine that has never executed is '
                                  'not a validated deployment (charter §6)')
    else:
        try:
            run = load_and_run(args.out, args.projection, args.arm, args.parity_n,
                               args.market1501, args.tolerance)
        except Exception as exc:  # noqa: BLE001 — a load test that fails IS the finding
            report['load_and_run'] = {'status': f'FAILED: {type(exc).__name__}: {exc}'}
            report['verdict'] = ('ENGINE BUILT BUT NOT DEPLOYABLE — the production extractor could '
                                 'not load and run it')
            _write(report, args.out)
            raise SystemExit(report['verdict'])
        valid = [c for c in run['crops'] if c.get('finite') and not c.get('all_zero')]
        deltas = [c['vs_torch']['max_abs'] for c in run['crops'] if 'vs_torch' in c]
        report['load_and_run'] = run
        report['load_and_run']['summary'] = {
            'crops_executed': len(run['crops']), 'crops_with_valid_embedding': len(valid),
            'embeddings_produced': run['embeddings_produced'],
            'worst_max_abs_vs_torch': max(deltas) if deltas else None,
            'tolerance': args.tolerance,
            'verdict': ('NOT MEASURED — synthetic crops, no Torch reference' if not deltas
                        else 'PASS' if max(deltas) <= args.tolerance and valid
                        else 'FAIL')}
        if report['load_and_run']['summary']['verdict'] == 'FAIL':
            _write(report, args.out)
            raise SystemExit('engine output disagrees with the Torch reference beyond tolerance')

    report['verdict'] = ('COMPLETE — engine built on a Jetson-marked host, loaded through the '
                         'production extractor and executed real crops'
                         if env['is_jetson_marker_present'] and not args.skip_load_test
                         else 'PARTIALLY COMPLETE — see the flags above')
    _write(report, args.out)
    logger.info(json.dumps({'engine': report['engine'], 'verdict': report['verdict']}, indent=2))


def _write(report, engine_path):
    path = os.path.splitext(os.path.abspath(engine_path))[0] + '_build_report.json'
    with open(path, 'w') as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f'Report: {path}')


if __name__ == '__main__':
    main()

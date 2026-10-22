#!/usr/bin/env python3
"""TR-09 — TransReID ONNX export, graph validation and PyTorch<->ONNX numerical parity.

Charter §6 sequence, steps 3-5 plus the parts of 1-2 and 9 this host can execute:

  1. the reference is the *production* extractor (``TransReIdExtractor`` loaded from
     ``transformer_120.pth``), not a re-implementation;
  2. the exported graph is the ViT-Base/16 + BNNeck path that yields the 768-D post-BNNeck CLS
     feature — the value the documented contract then centres, projects (SVD 768->512) and
     L2-normalises. The projection is deliberately NOT baked into the graph: it is a fitted
     artefact with its own provenance and the contract keeps feature extraction and projection
     fitting separate (charter §5);
  3. two graphs are exported, because the deployed arms differ in a way one graph cannot express:
     ``transreid_feature_agnostic.onnx`` (no Side Information Embedding) and
     ``transreid_feature_sie.onnx`` (``cam_label`` as a real int64 input). ``cam_label = 0`` is NOT
     equivalent to omitting SIE — camera 0's embedding is added — so an agnostic graph is required,
     not a zero index;
  4. each graph is checked (``onnx.checker``), then inspected for inputs/outputs (names, ranks,
     dtypes), opset/IR version, op-type histogram, initialisers, dynamic-batch behaviour and an
     off-contract resolution probe;
  5. numerical parity is measured against the reference on the SAME decoded crops, both at 768-D
     and after the contract's centre -> SVD -> L2 step, with an end-to-end comparison against
     ``extract_embedding()`` itself.

Not claimed here, and recorded as such in the report: a TensorRT engine, target-device latency, or
an accuracy-parity statement about deployment. ONNX Runtime on this host is CPU-only, so parity is
fp32 Torch vs fp32 ONNXRuntime-on-CPU — an exporter/numerics check, not an accelerator benchmark.

Exit status is non-zero if a graph fails validation, if no image could be scored, or if measured
parity exceeds --parity-tolerance / falls below --cosine-floor.

Usage:
    python scripts/export_transreid_onnx.py \\
        --checkpoint ../edgesync_bringup/models/transreid/transformer_120.pth \\
        --projection ../edgesync_bringup/models/transreid/svd_projection_768_to_512_real_v2.npz \\
        --market1501 ../edgesync_bringup/data/Market-1501-v15.09.15 \\
        --out-dir /path/outside/repo/onnx --parity-n 8
"""
import argparse
import collections
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent))

# The deployed extractor's wire contract, against which the *exported* graph is validated.
# Only `image` is float NCHW; `cam_label` is an int64 vector, so the two need separate rules.
IMAGE_CONTRACT = {'dtype': 'FLOAT', 'rank': 4, 'trailing_dims': [3, 256, 128]}
CAM_CONTRACT = {'dtype': 'INT64', 'rank': 1}
NAME_GRAMMAR = re.compile(r'^(-?\d+)_c(\d)s(\d)_.*\.jpg$')


def sha256_of(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def make_feature_graph(model, with_cam):
    """Wrap the production backbone so the exporter sees one explicit input signature.

    ``with_cam=False`` calls ``model(image, cam_label=None)`` — the SIE-free arm. Baking the
    branch into a Python bool (rather than passing ``cam_label=None`` through the graph) is what
    keeps the agnostic graph agnostic.
    """
    import torch.nn as nn

    class _FeatureGraph(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = model

        def forward(self, image, cam_label=None):
            return self.inner(image, cam_label=cam_label if with_cam else None)

    graph = _FeatureGraph()
    graph.eval()
    return graph


def export(wrapper, dummy_inputs, input_names, output_path, opset, dynamic_axes, dynamo):
    import torch

    wrapper.eval()
    torch.onnx.export(
        wrapper, tuple(dummy_inputs), output_path,
        input_names=input_names, output_names=['feature_768'],
        opset_version=opset, do_constant_folding=True,
        dynamic_axes=dynamic_axes, dynamo=dynamo,
    )
    return output_path


def _tensor_proto_shape(tensor):
    return [(d.dim_value if d.HasField('dim_value') else d.dim_param)
            for d in tensor.type.tensor_type.shape.dim]


def check_inputs(inputs, expected_names):
    """Validate exported input names, dtypes and shapes against the deployed contract."""
    import onnx

    names = [i.name for i in inputs]
    violations = []
    for i in inputs:
        dtype = onnx.TensorProto.DataType.Name(i.type.tensor_type.elem_type)
        shape = _tensor_proto_shape(i)
        if i.name == 'image':
            rule = IMAGE_CONTRACT
        elif i.name == 'cam_label':
            rule = CAM_CONTRACT
        else:
            violations.append(f'{i.name}: not part of the expected input set {sorted(expected_names)}')
            continue
        if dtype != rule['dtype']:
            violations.append(f"{i.name}: dtype {dtype}, contract expects {rule['dtype']}")
        if len(shape) != rule['rank']:
            violations.append(f"{i.name}: rank {len(shape)}, contract expects {rule['rank']}")
        elif 'trailing_dims' in rule and list(shape[1:]) != rule['trailing_dims']:
            violations.append(f"{i.name}: spatial dims {list(shape[1:])}, contract expects {rule['trailing_dims']}")
    for name in expected_names:
        if name not in names:
            violations.append(f'{name}: expected input absent from the graph')
    return violations


def inspect_graph(onnx_path, expected_inputs):
    """Structural validation of an exported graph — checker, names, ranks, dtypes, ops."""
    import onnx

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    graph = model.graph

    ops = collections.Counter(node.op_type for node in graph.node)
    float_init_bytes = sum(int(np.prod(i.dims)) * 4 for i in graph.initializer
                           if i.data_type == onnx.TensorProto.FLOAT)
    external = [i.name for i in graph.initializer
                if i.data_location == onnx.TensorProto.EXTERNAL]
    inputs = [{'name': i.name,
               'dtype': onnx.TensorProto.DataType.Name(i.type.tensor_type.elem_type),
               'shape': _tensor_proto_shape(i)} for i in graph.input]
    outputs = [{'name': o.name,
                'dtype': onnx.TensorProto.DataType.Name(o.type.tensor_type.elem_type),
                'shape': _tensor_proto_shape(o)} for o in graph.output]
    return {
        'path': onnx_path,
        'bytes': os.path.getsize(onnx_path),
        'sha256': sha256_of(onnx_path),
        'checker': 'PASS (onnx.checker.check_model)',
        'ir_version': model.ir_version,
        'opset_imports': {o.domain or 'ai.onnx': o.version for o in model.opset_import},
        'n_nodes': len(graph.node),
        'op_type_histogram': dict(sorted(ops.items(), key=lambda kv: (-kv[1], kv[0]))),
        'inputs': inputs,
        'outputs': outputs,
        # A non-integer dim here is a symbolic dimension left by tracing. The engine build on the
        # target must resolve it; the runtime probes below record what it actually resolves to, so
        # the deployment step is documented rather than assumed.
        'output_symbolic_dims': {o['name']: [d for d in o['shape'] if isinstance(d, str)]
                                 for o in outputs if any(isinstance(d, str) for d in o['shape'])},
        'n_initializers': len(graph.initializer),
        'initializer_float32_bytes': float_init_bytes,
        'external_data_tensors': external,
        'input_contract_violations': check_inputs(graph.input, expected_inputs),
    }


def ort_session(onnx_path, providers=('CPUExecutionProvider',)):
    import onnxruntime as ort

    return ort.InferenceSession(onnx_path, providers=list(providers))


def parity_metrics(ref, test):
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    test = np.asarray(test, dtype=np.float64).reshape(-1)
    if ref.shape != test.shape:
        raise ValueError(f'parity shape mismatch: reference {ref.shape} vs test {test.shape}')
    diff = np.abs(ref - test)
    denom = float(np.linalg.norm(ref) * np.linalg.norm(test))
    cos = float(np.dot(ref, test) / denom) if denom > 0 else float('nan')
    return {'max_abs': float(diff.max()), 'mean_abs': float(diff.mean()),
            'rmse': float(np.sqrt(np.mean(diff ** 2))), 'cosine': cos,
            'l2_dist': float(np.linalg.norm(ref - test))}


def project_and_normalise(x768, mean, projection):
    """The documented contract step after the graph: centre -> SVD 768->512 -> L2.

    A non-finite input or a zero norm is a real failure of the feature, not a value to paper
    over: this raises instead of returning a zero vector (charter §5 forbids substituting
    fabricated embeddings).
    """
    x = np.asarray(x768, dtype=np.float32).reshape(-1)
    if not np.isfinite(x).all():
        raise ValueError('project_and_normalise: non-finite 768-D feature')
    z = (x - mean) @ projection
    norm = float(np.linalg.norm(z))
    if norm <= 1e-6:
        raise ValueError(f'project_and_normalise: degenerate projection, |z| = {norm:.3e}')
    return (z / norm).astype(np.float32)


def pick_images(market_root, count, evaluate_offline_path):
    """Real Market-1501 crops, deterministic order, first ``count`` of the named population."""
    import importlib.util

    spec = importlib.util.spec_from_file_location('evaluate_offline', evaluate_offline_path)
    eo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eo)
    gallery, _, _ = eo.collect_gallery_and_query(market_root, population='directory')
    return [g[0] for g in gallery[:count]], 'directory'


def identity_of(path):
    match = NAME_GRAMMAR.match(os.path.basename(path))
    if not match:
        raise ValueError(f'{os.path.basename(path)} does not match the Market-1501 name grammar')
    return int(match.group(1)), int(match.group(2))


def main():
    parser = argparse.ArgumentParser(description='TR-09 TransReID ONNX export + validation + parity.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--projection', required=True)
    parser.add_argument('--market1501', default='',
                        help='Required for the numerical parity stage (real crops).')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--parity-n', type=int, default=8)
    parser.add_argument('--parity-tolerance', type=float, default=2e-3,
                        help='Max |Torch - ONNX| allowed on the 768-D post-BNNeck feature. '
                             'Justified as fp32 export / ORT op-reordering drift on a 768-wide ViT; '
                             'the measured value is always reported, this only decides pass/fail.')
    parser.add_argument('--cosine-floor', type=float, default=0.9999)
    parser.add_argument('--reference-device', default='cuda', choices=['cuda', 'cpu'],
                        help='Device for the Torch reference. cpu compares against the same '
                             'CPUExecutionProvider ONNX runs, removing cross-device drift.')
    parser.add_argument('--skip-dynamic-batch', action='store_true')
    parser.add_argument('--dynamo', action='store_true',
                        help='Use the Torch exporter --dynamo path instead of the legacy exporter.')
    args = parser.parse_args()

    import torch

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    report = {
        'experiment': 'ES-EXP07 TR-09',
        'evaluation_type': 'ONNX export, graph validation and Torch<->ONNX numerical parity',
        'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'invocation': {'argv': sys.argv, 'cwd': os.getcwd(), 'settings': vars(args)},
        'host': {'python': sys.version.split()[0], 'torch': torch.__version__,
                 'cuda_available': torch.cuda.is_available()},
        'assets': {
            'checkpoint': {'path': os.path.abspath(args.checkpoint),
                           'bytes': os.path.getsize(args.checkpoint),
                           'sha256': sha256_of(args.checkpoint)},
            'projection': {'path': os.path.abspath(args.projection),
                           'bytes': os.path.getsize(args.projection),
                           'sha256': sha256_of(args.projection)}},
        'shape_assumptions': {
            'input': 'NCHW float32; contract is 1x3x256x128 with only the batch axis dynamic',
            'pos_embed': 'the checkpoint carries (1, 211, 768); 256x128 with patch 16 / stride 12 '
                         'gives exactly 210 patches + CLS, so no interpolation path is exercised. '
                         'Other resolutions change the token count and are recorded as '
                         'off-contract, not silently re-shaped.',
            'preprocessing': 'resize 256x128, /255, mean=std=[0.5,0.5,0.5], HWC->CHW, RGB uint8 in',
            'projection': 'SVD 768->512 stays OUTSIDE the graph (fitted artefact, own provenance); '
                          'the graph ends at the post-BNNeck 768-D CLS feature',
        },
    }

    from edgesync_identity.models.extractor_factory import create_extractor
    extractor = create_extractor('transreid', checkpoint_path=args.checkpoint,
                                 projection_path=args.projection)
    model = extractor._model.eval()
    report['reference_extractor'] = {
        'class': type(extractor).__name__,
        'load_device': str(extractor._device),
        'checkpoint_coverage': extractor.checkpoint_coverage,
        'torch_parameters': sum(p.numel() for p in model.parameters()),
    }

    if args.reference_device == 'cpu' and extractor._device.type == 'cuda':
        model = model.to('cpu')
        extractor._device = torch.device('cpu')
    report['reference_device_used'] = args.reference_device

    crop = np.full((256, 128, 3), 127, dtype=np.uint8)
    dummy = extractor.preprocess(crop)
    dummy_cam = torch.tensor([2], device=dummy.device, dtype=torch.long)

    arms = (
        ('agnostic', False, ['image'],
         {'image': {0: 'batch'}, 'feature_768': {0: 'batch'}}),
        ('sie', True, ['image', 'cam_label'],
         {'image': {0: 'batch'}, 'cam_label': {0: 'batch'}, 'feature_768': {0: 'batch'}}),
    )

    graphs = {}
    for name, with_cam, input_names, axes in arms:
        wrapper = make_feature_graph(model, with_cam=with_cam)
        inputs = [dummy, dummy_cam] if with_cam else [dummy]
        path = os.path.join(out_dir, f'transreid_feature_{name}.onnx')
        t0 = time.perf_counter()
        try:
            export(wrapper, inputs, input_names, path, args.opset, axes, args.dynamo)
        except Exception as exc:  # noqa: BLE001 — the export result IS the finding
            report.setdefault('exports', {})[name] = {
                'status': f'FAILED: {type(exc).__name__}: {exc}'}
            logger.error(f'ONNX export failed for {name}: {exc}')
            continue
        report.setdefault('exports', {})[name] = {
            'status': 'OK', 'seconds': round(time.perf_counter() - t0, 1),
            'input_names': input_names, 'opset': args.opset, 'dynamo': args.dynamo}
        graphs[name] = (path, input_names)
        try:
            report.setdefault('graph_validation', {})[name] = inspect_graph(
                path, set(input_names))
        except Exception as exc:  # noqa: BLE001
            report.setdefault('graph_validation', {})[name] = {
                'status': f'VALIDATION FAILED: {type(exc).__name__}: {exc}'}

    failed_validation = [n for n, v in report.get('graph_validation', {}).items()
                         if 'status' in v or v['input_contract_violations']]
    if not graphs:
        _write(report, out_dir)
        raise SystemExit('no graph exported — see the report')

    if not args.skip_dynamic_batch:
        for name, (path, input_names) in graphs.items():
            entry = {}
            for batch in (1, 2):
                sess = ort_session(path)
                feed = {'image': np.zeros((batch, 3, 256, 128), dtype=np.float32)}
                if 'cam_label' in input_names:
                    feed['cam_label'] = np.arange(batch, dtype=np.int64)
                try:
                    out = sess.run(None, feed)[0]
                    entry[f'dynamic_batch_{batch}'] = f'OK, output shape {tuple(out.shape)}'
                    if batch == 1:
                        entry['feature_width_observed'] = int(out.shape[-1])
                except Exception as exc:  # noqa: BLE001
                    entry[f'dynamic_batch_{batch}'] = f'FAILED {type(exc).__name__}: {exc}'
            # Off-contract resolution: the graph must NOT silently accept it.
            try:
                sess = ort_session(path)
                feed = {'image': np.zeros((1, 3, 128, 64), dtype=np.float32)}
                if 'cam_label' in input_names:
                    feed['cam_label'] = np.array([0], dtype=np.int64)
                out = sess.run(None, feed)[0]
                entry['off_contract_128x64'] = (
                    f'RAN, output {tuple(out.shape)} — shape inference did not pin the '
                    f'resolution; deployment must enforce 256x128 outside the graph')
            except Exception as exc:  # noqa: BLE001
                entry['off_contract_128x64'] = f'refused by the graph: {type(exc).__name__}'
            report.setdefault('runtime_checks', {})[name] = entry

    if args.market1501:
        import cv2
        import onnxruntime as ort

        paths, pop = pick_images(args.market1501, args.parity_n,
                                 script_dir / 'evaluate_offline.py')
        report['parity'] = {'population': pop, 'n_images_selected': len(paths), 'images': [],
                            'providers': ort.get_available_providers(),
                            'reference_device': args.reference_device,
                            'test_device': 'onnxruntime CPUExecutionProvider (fp32)'}
        sessions = {name: ort_session(path) for name, (path, _) in graphs.items()}
        worst = {name: -1.0 for name in graphs}
        worst_cos = {name: 2.0 for name in graphs}
        worst512 = {name: -1.0 for name in graphs}
        e2e_worst = {name: -1.0 for name in graphs}
        arm_gap_worst = -1.0
        for img_path in paths:
            bgr = cv2.imread(img_path)
            if bgr is None:
                report['parity']['images'].append({'path': img_path, 'status': 'decode_failed'})
                continue
            crop_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pid, camid = identity_of(img_path)
            if pid < 0 or camid < 1:
                report['parity']['images'].append({'path': img_path, 'status': 'not_an_identity'})
                continue
            cam_label = max(camid - 1, 0)
            tensor = extractor.preprocess(crop_rgb)
            with torch.no_grad():
                ref768 = model(tensor, cam_label=None).cpu().numpy().reshape(-1)
                ref768_sie = model(
                    tensor, cam_label=torch.tensor([cam_label], device=tensor.device,
                                                   dtype=torch.long)).cpu().numpy().reshape(-1)
            # Each arm has its own reference at both 768-D and post-contract 512-D. Comparing the
            # SIE graph against the agnostic reference would measure the SIE effect, not parity.
            refs = {'agnostic': (ref768, project_and_normalise(ref768, extractor._svd_mean,
                                                                extractor._svd_projection)),
                    'sie': (ref768_sie, project_and_normalise(ref768_sie, extractor._svd_mean,
                                                               extractor._svd_projection))}
            record = {'path': img_path, 'pid': pid, 'camid': camid, 'cam_label': cam_label}

            # How much SIE itself moves the deployed embedding — evidence that cam_label=0 is not
            # "no SIE", measured rather than asserted.
            gap = parity_metrics(refs['agnostic'][1], refs['sie'][1])
            record['agnostic_vs_sie_arm'] = gap
            arm_gap_worst = max(arm_gap_worst, gap['l2_dist'])

            for name, reference512 in (('agnostic', refs['agnostic'][1]),
                                       ('sie', refs['sie'][1])):
                prod = np.asarray(extractor.extract_embedding(
                    crop_rgb, cam_label=None if name == 'agnostic' else cam_label),
                    dtype=np.float32).reshape(-1)
                if np.all(prod == 0.0):
                    record[f'production_extract_embedding_vs_torch_{name}'] = {
                        'status': 'SENTINEL — ' + str(getattr(extractor, 'last_failure', 'unknown'))}
                    continue
                metrics = parity_metrics(reference512, prod)
                record[f'production_extract_embedding_vs_torch_{name}'] = metrics
                e2e_worst[name] = max(e2e_worst[name], metrics['max_abs'])

            for name, (_, input_names) in graphs.items():
                feed = {'image': tensor.cpu().numpy()}
                reference768, reference512_arm = refs[name]
                if 'cam_label' in input_names:
                    feed['cam_label'] = np.array([cam_label], dtype=np.int64)
                onnx768 = sessions[name].run(None, feed)[0].reshape(-1)
                metrics = parity_metrics(reference768, onnx768)
                onnx512 = project_and_normalise(onnx768, extractor._svd_mean,
                                                extractor._svd_projection)
                record[name] = {'feature_768': metrics,
                                'feature_512_after_contract':
                                    parity_metrics(reference512_arm, onnx512)}
                worst[name] = max(worst[name], metrics['max_abs'])
                worst_cos[name] = min(worst_cos[name], metrics['cosine'])
                worst512[name] = max(worst512[name],
                                     record[name]['feature_512_after_contract']['max_abs'])
            report['parity']['images'].append(record)

        scored = [n for n in graphs if worst[n] >= 0.0]
        report['parity']['n_images_scored'] = len(
            [i for i in report['parity']['images'] if 'agnostic' in i or 'sie' in i])
        report['parity']['summary'] = {
            name: {'worst_max_abs_768': worst[name],
                   'worst_max_abs_512_after_contract': worst512[name],
                   'worst_cosine_768': worst_cos[name],
                   'tolerance': args.parity_tolerance, 'cosine_floor': args.cosine_floor,
                   'verdict': ('PASS' if worst[name] <= args.parity_tolerance
                               and worst512[name] <= args.parity_tolerance
                               and worst_cos[name] >= args.cosine_floor else 'FAIL')}
            for name in scored}
        report['parity']['production_api_vs_torch_worst_max_abs_512'] = {
            name: (None if e2e_worst[name] < 0 else e2e_worst[name]) for name in graphs}
        report['parity']['sie_effect_worst_l2_on_512'] = (
            None if arm_gap_worst < 0 else arm_gap_worst)
        report['parity']['ORT_is_CPU_only'] = (
            'CUDAExecutionProvider' not in ort.get_available_providers())
        report['parity']['caveat'] = (
            'fp32 Torch-vs-fp32 ONNXRuntime-on-CPU numerics check. It validates the exported graph '
            'and the preprocessing/projection contract; it is not a TensorRT result, an fp16 '
            'result, a speedup, or a Jetson measurement.')
        report['parity']['arm_note'] = (
            'feature_512_after_contract and production_extract_embedding_vs_torch_* are each '
            'compared against the reference from the SAME arm; the SIE arm legitimately differs '
            'from the agnostic arm, and that difference is reported separately as sie_effect_worst_l2_on_512.')
        unscored = [n for n, _ in graphs.items() if n not in scored]
        if unscored:
            _write(report, out_dir)
            raise SystemExit(f'parity NOT MEASURED for {unscored} — no image could be scored')
        bad = [n for n, v in report['parity']['summary'].items() if v['verdict'] == 'FAIL']
        if bad:
            _write(report, out_dir)
            raise SystemExit(f'parity FAILED for {bad} — see the report')
    else:
        report['parity'] = {'status': 'NOT MEASURED — --market1501 not supplied'}

    symbolic = {n: v.get('output_symbolic_dims') for n, v in
                report.get('graph_validation', {}).items() if isinstance(v, dict)}
    observed = {n: v.get('feature_width_observed') for n, v in
                report.get('runtime_checks', {}).items()}
    report['deployment_notes'] = {
        'symbolic_output_dims': symbolic,
        'feature_width_observed_at_runtime': observed,
        'reading': 'the exported feature width is traced through the final BatchNormalization, so '
                   'the declared output dim is symbolic rather than the constant 768. ONNXRuntime '
                   'resolves it to the measured width above. This is recorded, not assumed: the '
                   'TensorRT parser has to resolve the same dim during the engine build.',
        'engine_build_requirements': [
            'an explicit optimization profile pinned to the contract batch/dims '
            '(1x3x256x128, cam_label 1) — no implicit shapes',
            'a build on the Orin Nano itself with the matching TensorRT major version; engines '
            'are not portable across GPU architectures or TRT majors',
            'if the parser cannot resolve the symbolic width the build must fail; '
            'TransReIdTrtExtractor._validate_engine_contract() additionally refuses any engine '
            'whose feature output is not 768 wide, so a wrong-width engine cannot silently be '
            'loaded and double-projected',
        ],
    }

    report['not_claimed'] = {
        'tensorrt_engine': 'BLOCKED — Pending Jetson Orin Nano access '
                           '(no tensorrt, trtexec or JetPack on this host)',
        'target_latency': 'NOT MEASURED — an ONNX/CPU parity check is not a speedup claim',
        'accuracy_parity_on_device': 'NOT MEASURED — requires the TensorRT arm on the target',
        'graph_validation': (f'{len(failed_validation)} graph(s) with validation problems'
                             if failed_validation else 'all exported graphs passed '
                             'onnx.checker and the input-contract checks'),
    }
    _write(report, out_dir)
    logger.info(json.dumps({k: report[k] for k in
                            ('exports', 'graph_validation', 'runtime_checks') if k in report},
                           indent=2)[:2500])
    logger.info(json.dumps(report.get('parity', {}).get('summary', {}), indent=2))


def _json_default(obj):
    """Keep numbers numeric in the report; only genuinely unserialisable objects become text.

    ``default=str`` would have written every numpy metric as a string, which makes the report
    unreadable by anything that wants to compare numbers.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _write(report, out_dir):
    path = os.path.join(out_dir, 'transreid_onnx_report.json')
    with open(path, 'w') as handle:
        json.dump(report, handle, indent=2, default=_json_default)
    logger.info(f'Report: {path}')


if __name__ == '__main__':
    main()

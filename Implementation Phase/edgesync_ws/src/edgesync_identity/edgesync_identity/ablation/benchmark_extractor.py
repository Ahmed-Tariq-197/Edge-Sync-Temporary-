"""Benchmark one Re-ID backbone: latency, engine size and (optionally) accuracy.

Example (on the Jetson, after sourcing the workspace):
  python3 -m edgesync_identity.ablation.benchmark_extractor --backbone mobilenetv2 \
      --engine-path engines/mobilenetv2_reid_fp16.engine --mode full \
      --query-dir data/market1501/query --gallery-dir data/market1501/bounding_box_test \
      --output results/mobilenetv2.json
"""
import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional

import numpy as np

from ..factory.extractor_factory import ExtractorFactory
from ..interfaces.base_identity_extractor import BaseIdentityExtractor
from .ablation_config import (
    BYTES_PER_MIB, EXIT_BAD_ARGUMENTS, EXIT_NO_HARDWARE, EXIT_OK, LATENCY_PERCENTILES,
    MEASURED_ITERATIONS, MS_PER_SECOND, PIXEL_RANGE_EXCLUSIVE_MAX, PROGRESS_LOG_INTERVAL,
    RESULT_SCHEMA_VERSION, SYNTHETIC_CROP_HEIGHT, SYNTHETIC_CROP_SEED, SYNTHETIC_CROP_WIDTH,
    WARMUP_ITERATIONS, AblationConfig,
)
from .market_dataset import ImageRecord, list_records, load_rgb
from .reid_metrics import ReidEvaluator

MODE_LATENCY = "latency"
MODE_ACCURACY = "accuracy"
MODE_FULL = "full"


@dataclass
class LatencyStats:
    samples: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_fps: float


def summarize_latency(durations_s: List[float]) -> LatencyStats:
    """Convert per-call durations (seconds) to millisecond statistics."""
    ms = np.asarray(durations_s, dtype=np.float64) * MS_PER_SECOND
    p50, p95, p99 = (float(np.percentile(ms, q)) for q in LATENCY_PERCENTILES)
    mean = float(ms.mean())
    return LatencyStats(
        samples=int(ms.size), mean_ms=mean, std_ms=float(ms.std()), min_ms=float(ms.min()),
        max_ms=float(ms.max()), p50_ms=p50, p95_ms=p95, p99_ms=p99,
        throughput_fps=MS_PER_SECOND / mean if mean > 0.0 else 0.0)


def measure_calls(call: Callable[[], object], warmup: int, iterations: int,
                  clock: Callable[[], float] = time.perf_counter) -> LatencyStats:
    """Time `call` after discarding `warmup` invocations."""
    for _ in range(warmup):
        call()
    durations: List[float] = []
    for _ in range(iterations):
        start = clock()
        call()
        durations.append(clock() - start)
    return summarize_latency(durations)


def synthetic_crop() -> np.ndarray:
    rng = np.random.default_rng(SYNTHETIC_CROP_SEED)
    return rng.integers(0, PIXEL_RANGE_EXCLUSIVE_MAX,
                        (SYNTHETIC_CROP_HEIGHT, SYNTHETIC_CROP_WIDTH, 3), dtype=np.uint8)


def benchmark_latency(extractor: BaseIdentityExtractor, crop: np.ndarray,
                      warmup: int, iterations: int) -> dict:
    result = {"end_to_end": asdict(measure_calls(
        lambda: extractor.extract_embedding(crop), warmup, iterations))}
    runner = getattr(extractor, "runner", None)
    if runner is not None and hasattr(extractor, "preprocess"):
        tensor = extractor.preprocess(crop)
        result["engine_only"] = asdict(measure_calls(lambda: runner.infer(tensor), warmup, iterations))
    return result


def embed_records(extractor: BaseIdentityExtractor, records: List[ImageRecord], tag: str) -> np.ndarray:
    embeddings = []
    for index, record in enumerate(records, start=1):
        embeddings.append(extractor.extract_embedding(load_rgb(record.path)))
        if index % PROGRESS_LOG_INTERVAL == 0:
            print(f"[{tag}] embedded {index}/{len(records)}", file=sys.stderr)
    return np.stack(embeddings).astype(np.float32)


def benchmark_accuracy(extractor: BaseIdentityExtractor, query_dir: str, gallery_dir: str,
                       config: AblationConfig, max_images: Optional[int],
                       embeddings_out: Optional[str]) -> dict:
    queries = list_records(query_dir, max_images)
    gallery = list_records(gallery_dir, max_images)
    q_emb = embed_records(extractor, queries, "query")
    g_emb = embed_records(extractor, gallery, "gallery")
    q_pid = np.array([r.person_id for r in queries])
    q_cam = np.array([r.camera_id for r in queries])
    g_pid = np.array([r.person_id for r in gallery])
    g_cam = np.array([r.camera_id for r in gallery])
    if embeddings_out:
        np.savez_compressed(embeddings_out, q_emb=q_emb, q_pid=q_pid, q_cam=q_cam,
                            g_emb=g_emb, g_pid=g_pid, g_cam=g_cam)
    return ReidEvaluator(config).evaluate(q_emb, q_pid, q_cam, g_emb, g_pid, g_cam).to_dict()


def build_parser() -> argparse.ArgumentParser:
    factory = ExtractorFactory()
    defaults = AblationConfig()
    parser = argparse.ArgumentParser(description="Benchmark a Re-ID backbone.")
    parser.add_argument("--backbone", required=True, choices=factory.available())
    parser.add_argument("--engine-path", default=None)
    parser.add_argument("--mode", choices=(MODE_LATENCY, MODE_ACCURACY, MODE_FULL), default=MODE_FULL)
    parser.add_argument("--query-dir", default=None)
    parser.add_argument("--gallery-dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERATIONS)
    parser.add_argument("--iterations", type=int, default=MEASURED_ITERATIONS)
    parser.add_argument("--threshold", type=float, default=defaults.similarity_threshold)
    parser.add_argument("--target-tar", type=float, default=defaults.target_true_accept_rate)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--embeddings-out", default=None)
    parser.add_argument("--allow-mock", action="store_true",
                        help="Permit mock embeddings (harness dry run only; results are not valid).")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    needs_data = args.mode in (MODE_ACCURACY, MODE_FULL)
    if needs_data and not (args.query_dir and args.gallery_dir):
        print("--query-dir and --gallery-dir are required for accuracy/full mode", file=sys.stderr)
        return EXIT_BAD_ARGUMENTS

    extractor = ExtractorFactory().create(args.backbone, args.engine_path)
    hardware_backed = bool(getattr(extractor, "is_hardware_backed", False))
    if not hardware_backed and not args.allow_mock:
        print("No TensorRT runtime/engine attached; refusing to benchmark mock embeddings. "
              "Pass --allow-mock for a harness dry run.", file=sys.stderr)
        return EXIT_NO_HARDWARE

    config = AblationConfig(similarity_threshold=args.threshold,
                            target_true_accept_rate=args.target_tar)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "label": args.label or args.backbone,
        "backbone": args.backbone,
        "engine_path": args.engine_path,
        "engine_size_mib": (os.path.getsize(args.engine_path) / BYTES_PER_MIB
                            if args.engine_path and os.path.isfile(args.engine_path) else None),
        "hardware_backed": hardware_backed,
        "threshold": config.similarity_threshold,
        "target_tar": config.target_true_accept_rate,
        "robot_a_cameras": list(config.robot_a_cameras),
        "robot_b_cameras": list(config.robot_b_cameras),
    }
    if args.mode in (MODE_LATENCY, MODE_FULL):
        result["latency"] = benchmark_latency(extractor, synthetic_crop(), args.warmup, args.iterations)
    if needs_data:
        result["accuracy"] = benchmark_accuracy(extractor, args.query_dir, args.gallery_dir, config,
                                                args.max_images, args.embeddings_out)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"Wrote {args.output}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

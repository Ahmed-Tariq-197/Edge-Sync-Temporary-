#!/usr/bin/env bash
# Run both Task 6 arms on the Jetson with identical settings, then compare.
# Required environment: OSNET_ENGINE, MOBILENETV2_ENGINE, QUERY_DIR, GALLERY_DIR
# Optional: OUT_DIR (default docs/ablation/task6_results), THRESHOLD (default 0.70)
# Prerequisite: workspace built and sourced so `edgesync_identity` is importable.
set -euo pipefail

NVPMODEL_MODE=0
MODULE_BENCH="edgesync_identity.ablation.benchmark_extractor"
MODULE_COMPARE="edgesync_identity.ablation.compare_results"
OUT_DIR="${OUT_DIR:-docs/ablation/task6_results}"
THRESHOLD="${THRESHOLD:-0.70}"
: "${OSNET_ENGINE:?set OSNET_ENGINE}"
: "${MOBILENETV2_ENGINE:?set MOBILENETV2_ENGINE}"
: "${QUERY_DIR:?set QUERY_DIR}"
: "${GALLERY_DIR:?set GALLERY_DIR}"

mkdir -p "$OUT_DIR"
sudo nvpmodel -m "$NVPMODEL_MODE"
sudo jetson_clocks
sudo tegrastats --interval 1000 --logfile "$OUT_DIR/tegrastats.log" --start
trap 'sudo tegrastats --stop' EXIT

run_arm() {
  local backbone="$1" engine="$2"
  python3 -m "$MODULE_BENCH" --backbone "$backbone" --engine-path "$engine" --mode full \
    --query-dir "$QUERY_DIR" --gallery-dir "$GALLERY_DIR" --threshold "$THRESHOLD" \
    --embeddings-out "$OUT_DIR/$backbone.embeddings.npz" --output "$OUT_DIR/$backbone.json"
}

run_arm osnet "$OSNET_ENGINE"
run_arm mobilenetv2 "$MOBILENETV2_ENGINE"

python3 -m "$MODULE_COMPARE" --baseline "$OUT_DIR/osnet.json" --candidate "$OUT_DIR/mobilenetv2.json" \
  --json-out "$OUT_DIR/comparison.json" --markdown-out "$OUT_DIR/comparison.md"

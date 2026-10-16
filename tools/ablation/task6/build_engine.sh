#!/usr/bin/env bash
# Build an FP16 TensorRT engine from an ONNX file (run on the Jetson Orin Nano).
# Use the same precision flags for every backbone so only the network differs.
set -euo pipefail

TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
WORKSPACE_MIB=1024
EXIT_USAGE=2

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <model.onnx> <model.engine>" >&2
  exit "$EXIT_USAGE"
fi

"$TRTEXEC" --onnx="$1" --saveEngine="$2" --fp16 --memPoolSize=workspace:"$WORKSPACE_MIB"

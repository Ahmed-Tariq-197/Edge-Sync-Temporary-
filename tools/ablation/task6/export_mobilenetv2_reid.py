#!/usr/bin/env python3
"""Export the MobileNetV2 Re-ID embedder to a static-shape ONNX file and verify parity.

Example:
  python tools/ablation/task6/export_mobilenetv2_reid.py \
      --checkpoint checkpoints/mobilenetv2_reid.pt --output engines/mobilenetv2_reid.onnx
"""
import argparse
import sys

import numpy as np
import torch

from mobilenetv2_reid_model import (
    EMBEDDING_DIM, INFERENCE_BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH,
    MobileNetV2Embedder,
)

ONNX_OPSET = 13
PARITY_TOLERANCE = 1e-4
INPUT_NAME = "input"
OUTPUT_NAME = "embedding"
EXIT_PARITY_FAILED = 3
EXIT_NO_CHECKPOINT = 2
PARITY_SEED = 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Export MobileNetV2 Re-ID embedder to ONNX.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-untrained", action="store_true",
                        help="Export without a trained head (shape/latency checks only).")
    args = parser.parse_args()

    if args.checkpoint is None and not args.allow_untrained:
        print("A trained --checkpoint is required (or pass --allow-untrained).", file=sys.stderr)
        return EXIT_NO_CHECKPOINT

    model = MobileNetV2Embedder(pretrained=args.checkpoint is None)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["embedder"])
    model.eval()

    torch.manual_seed(PARITY_SEED)
    dummy = torch.randn(INFERENCE_BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    torch.onnx.export(model, dummy, args.output, opset_version=ONNX_OPSET, do_constant_folding=True,
                      input_names=[INPUT_NAME], output_names=[OUTPUT_NAME])

    import onnxruntime as ort
    session = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    onnx_out = session.run([OUTPUT_NAME], {INPUT_NAME: dummy.numpy()})[0]
    with torch.no_grad():
        torch_out = model(dummy).numpy()
    max_diff = float(np.abs(onnx_out - torch_out).max())
    print(f"Exported {args.output}; embedding dim {onnx_out.shape[-1]}; max |torch-onnx| = {max_diff:.2e}")
    if onnx_out.shape[-1] != EMBEDDING_DIM or max_diff > PARITY_TOLERANCE:
        return EXIT_PARITY_FAILED
    return 0


if __name__ == "__main__":
    sys.exit(main())

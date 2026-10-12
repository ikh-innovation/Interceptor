#!/usr/bin/env python3
"""Build or verify the project's TensorRT model on the destination Jetson."""

import argparse
import os
from pathlib import Path


def main():
    default = Path(__file__).resolve().parents[1] / "runs/detect/v0/weights/best.pt"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=default, help="trained .pt weights")
    parser.add_argument("--workspace", type=float, default=2.0, help="export workspace limit in GiB")
    parser.add_argument("--verify-only", action="store_true", help="warm up the existing .engine without rebuilding")
    args = parser.parse_args()
    source = args.model.expanduser().resolve()
    if source.suffix != ".pt" or not source.is_file():
        parser.error(f"trained .pt weights not found: {source}")
    if args.workspace <= 0:
        parser.error("workspace must be positive")

    os.environ["YOLO_AUTOINSTALL"] = "false"
    import numpy as np
    import torch
    import tensorrt
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; install the PyTorch build matching this JetPack.")
    model = YOLO(str(source))
    engine = source.with_suffix(".engine")
    if not args.verify_only:
        engine = Path(model.export(format="engine", imgsz=640, half=True, device=0,
                                   batch=1, dynamic=False, workspace=args.workspace))
    if not engine.is_file():
        raise SystemExit(f"Engine missing: {engine}; run without --verify-only to build it.")
    optimized = YOLO(str(engine), task="detect")
    optimized.predict(np.zeros((640, 640, 3), np.uint8), imgsz=640, verbose=False)
    if optimized.names != model.names:
        raise SystemExit(f"Class labels differ: source={model.names}, engine={optimized.names}")
    print(f"Verified {engine}\nGPU: {torch.cuda.get_device_name(0)}; TensorRT: {tensorrt.__version__}; classes: {optimized.names}")


if __name__ == "__main__":
    main()

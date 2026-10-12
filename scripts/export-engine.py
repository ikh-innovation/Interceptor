#!/usr/bin/env python3
"""Build or verify the project's TensorRT model on the destination Jetson."""

import argparse
import os
from pathlib import Path
import shutil
import tempfile


def main():
    default = Path(__file__).resolve().parents[1] / "runs/detect/v0/weights/best.pt"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=default, help="trained .pt weights")
    parser.add_argument("--workspace", type=float, default=2.0, help="export workspace limit in GiB")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="engine input size; other than 640 it is written as <name>_<imgsz>.engine")
    parser.add_argument("--verify-only", action="store_true", help="warm up the existing .engine without rebuilding")
    args = parser.parse_args()
    source = args.model.expanduser().resolve()
    if source.suffix != ".pt" or not source.is_file():
        parser.error(f"trained .pt weights not found: {source}")
    if args.workspace <= 0:
        parser.error("workspace must be positive")
    if args.imgsz <= 0 or args.imgsz % 32:
        parser.error("imgsz must be a positive multiple of 32")

    os.environ["YOLO_AUTOINSTALL"] = "false"
    import numpy as np
    import torch
    import tensorrt
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; install the PyTorch build matching this JetPack.")
    model = YOLO(str(source))
    stem = source.stem if args.imgsz == 640 else f"{source.stem}_{args.imgsz}"
    engine = source.with_name(stem + ".engine")
    if not args.verify_only:
        # Ultralytics names the engine after the .pt, so other sizes are built from a
        # renamed copy and never overwrite the 640 production engine.
        with tempfile.TemporaryDirectory(dir=source.parent) as tmp:
            work = Path(tmp) / (stem + ".pt")
            shutil.copy2(source, work)
            built = Path(YOLO(str(work)).export(format="engine", imgsz=args.imgsz, half=True, device=0,
                                                batch=1, dynamic=False, workspace=args.workspace))
            shutil.move(str(built), engine)
    if not engine.is_file():
        raise SystemExit(f"Engine missing: {engine}; run without --verify-only to build it.")
    optimized = YOLO(str(engine), task="detect")
    optimized.predict(np.zeros((args.imgsz, args.imgsz, 3), np.uint8), imgsz=args.imgsz, verbose=False)
    if optimized.names != model.names:
        raise SystemExit(f"Class labels differ: source={model.names}, engine={optimized.names}")
    print(f"Verified {engine}\nGPU: {torch.cuda.get_device_name(0)}; TensorRT: {tensorrt.__version__}; classes: {optimized.names}")


if __name__ == "__main__":
    main()

"""
export_rasp_model.py  –  Compile best.pt → Hailo .hef for the Pi 5 + Hailo AI HAT
==================================================================================
Plan B (tracker_skip_final_rasp.py) runs detection on a Hailo-8/8L NPU, which
executes a **compiled .hef**, not a PyTorch/ONNX file.  This script produces that
.hef in the standard two-stage flow:

    best.pt  ──(Ultralytics)──►  best.onnx  ──(Hailo DFC)──►  best.hef

┌─ WHERE THIS RUNS ─────────────────────────────────────────────────────────────┐
│  * STAGE 1 (pt → onnx)   : anywhere with Ultralytics (this dev box is fine).    │
│  * STAGE 2 (onnx → hef)  : an **x86_64 Linux** machine with the Hailo Dataflow  │
│                            Compiler (DFC) + a calibration image set.            │
│                            It CANNOT run on the Raspberry Pi — the Pi only runs  │
│                            HailoRT inference on the finished .hef.               │
└─────────────────────────────────────────────────────────────────────────────┘

Install (x86 box), matching your HailoRT version on the Pi:
    pip install hailo-dataflow-compiler        # from the Hailo Developer Zone
    pip install ultralytics onnx onnxruntime

Then:
    # Stage 1 only (produces best.onnx):
    python export_rasp_model.py --stage onnx --weights runs/detect/v0/weights/best.pt

    # Stage 2 (needs calibration images):
    python export_rasp_model.py --stage hef  --onnx runs/detect/v0/weights/best.onnx \
        --calib data/videos_drones/calib_frames --arch hailo8l

Copy the resulting best.hef to the Pi at runs/detect/v0/weights/best.hef and the
tracker auto-detects it.

Key choices explained
---------------------
* imgsz = 640, opset = 11 — the Hailo model zoo YOLO flows are validated at 640;
  the NPU input size is FIXED here for good, so the tracker gets its "zoom" from
  cropping/tiling, not from imgsz (see tracker_skip_final_rasp.py header).
* INT8 quantisation — the NPU is integer-only, so the DFC quantises using the
  CALIBRATION IMAGES you supply.  Use REAL sky/drone frames from your own footage
  (extract a few hundred with --dump-calib) so the quantised model keeps the
  small, low-contrast drones Plan A cares about.  Generic COCO calib images will
  hurt far-drone recall.
* On-chip NMS — we bake NMS into the .hef (meta_arch + nms config) so HailoRT
  returns decoded boxes; HailoBackend in the tracker expects exactly this.
"""

import argparse
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1 — PyTorch checkpoint → ONNX
# ──────────────────────────────────────────────────────────────────────────────

def export_onnx(weights, imgsz=640, opset=11):
    from ultralytics import YOLO
    model = YOLO(str(weights))
    path = model.export(format="onnx", imgsz=imgsz, opset=opset)
    print(f"[ONNX] wrote {path}  (imgsz={imgsz}, opset={opset})")
    print("[ONNX] note the OUTPUT node name(s) printed above — you need them as "
          "end_node_names for stage 2 if the DFC can't infer them.")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Helper — dump calibration frames from your own footage
# ──────────────────────────────────────────────────────────────────────────────

def dump_calib(video, out_dir, n=300, imgsz=640):
    """Extract ~n evenly-spaced letterboxed frames as INT8 calibration images.
    Use YOUR sky/drone footage so quantisation preserves faint far targets."""
    import cv2
    import numpy as np
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or (n * 5)
    step  = max(1, total // n)
    saved = i = 0
    while saved < n:
        ret, frame = cap.read()
        if not ret:
            break
        if i % step == 0:
            h, w = frame.shape[:2]
            s = min(imgsz / w, imgsz / h)
            nw, nh = int(w * s), int(h * s)
            canvas = np.full((imgsz, imgsz, 3), 114, np.uint8)
            rz = cv2.resize(frame, (nw, nh))
            canvas[(imgsz - nh) // 2:(imgsz - nh) // 2 + nh,
                   (imgsz - nw) // 2:(imgsz - nw) // 2 + nw] = rz
            cv2.imwrite(str(out / f"calib_{saved:04d}.png"), canvas)
            saved += 1
        i += 1
    cap.release()
    print(f"[CALIB] wrote {saved} frames -> {out.resolve()}")


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2 — ONNX → HEF via the Hailo Dataflow Compiler
# ──────────────────────────────────────────────────────────────────────────────

def export_hef(onnx, calib_dir, arch="hailo8l", imgsz=640,
               end_nodes=None, meta_arch="yolov8", out_hef=None):
    """Translate → optimise (quantise) → compile.

    `meta_arch` selects the NMS post-process template. YOLO26/v8-style detect
    heads use the "yolov8" meta-arch in the Hailo model zoo; if your export uses
    a different head, set it accordingly (see Hailo model-zoo nms configs).
    """
    import numpy as np
    try:
        import cv2
        from hailo_sdk_client import ClientRunner
    except ImportError as e:
        raise SystemExit(
            f"Stage 2 needs the Hailo Dataflow Compiler + OpenCV on an x86 box "
            f"({e}). Install hailo-dataflow-compiler from the Hailo Developer Zone.")

    onnx = Path(onnx)
    out_hef = Path(out_hef) if out_hef else onnx.with_suffix(".hef")
    runner = ClientRunner(hw_arch=arch)

    print(f"[HEF] translate {onnx.name}  (arch={arch})")
    runner.translate_onnx_model(
        str(onnx),
        onnx.stem,
        end_node_names=end_nodes,                    # None -> DFC auto-detects
        net_input_shapes={"images": [1, 3, imgsz, imgsz]},
    )

    # Bake NMS into the graph so HailoRT returns decoded boxes (the tracker's
    # HailoBackend expects on-chip NMS output).
    runner.load_model_script(
        f'nms_postprocess(meta_arch={meta_arch}, engine=cpu)\n')

    # ── calibration set → INT8 quantisation ──────────────────────────────────
    imgs = sorted(Path(calib_dir).glob("*.png")) + sorted(Path(calib_dir).glob("*.jpg"))
    if not imgs:
        raise SystemExit(f"No calibration images in {calib_dir} — run --dump-calib first.")
    calib = np.stack([cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
                      for p in imgs]).astype(np.float32)
    print(f"[HEF] optimise/quantise with {len(imgs)} calibration frames")
    runner.optimize(calib)

    print("[HEF] compile …")
    compiled = runner.compile()
    out_hef.write_bytes(compiled)
    print(f"[HEF] wrote {out_hef.resolve()}  — copy this to the Pi.")
    return out_hef


# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Compile best.pt -> Hailo .hef")
    ap.add_argument("--stage", choices=["onnx", "hef", "all"], default="all")
    ap.add_argument("--weights", default="runs/detect/v0/weights/best.pt")
    ap.add_argument("--onnx",    default="runs/detect/v0/weights/best.onnx")
    ap.add_argument("--calib",   default="data/calib_frames")
    ap.add_argument("--arch",    default="hailo8l", choices=["hailo8", "hailo8l", "hailo15h"])
    ap.add_argument("--imgsz",   type=int, default=640)
    ap.add_argument("--meta-arch", default="yolov8")
    ap.add_argument("--end-nodes", nargs="*", default=None,
                    help="ONNX output node names (omit to let the DFC auto-detect)")
    ap.add_argument("--dump-calib", metavar="VIDEO", default=None,
                    help="Extract calibration frames from VIDEO into --calib and exit")
    ap.add_argument("--n-calib", type=int, default=300)
    args = ap.parse_args()

    if args.dump_calib:
        dump_calib(args.dump_calib, args.calib, n=args.n_calib, imgsz=args.imgsz)
        return

    onnx_path = args.onnx
    if args.stage in ("onnx", "all"):
        onnx_path = export_onnx(args.weights, imgsz=args.imgsz)
    if args.stage in ("hef", "all"):
        export_hef(onnx_path, args.calib, arch=args.arch, imgsz=args.imgsz,
                   end_nodes=args.end_nodes, meta_arch=args.meta_arch)


if __name__ == "__main__":
    main()

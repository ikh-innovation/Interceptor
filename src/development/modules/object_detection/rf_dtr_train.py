import sys
import os
import glob
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTHONFAULTHANDLER",       "1")   # better crash traces
os.environ.setdefault("CUDA_LAUNCH_BLOCKING",     "0")   # async CUDA (faster)
os.environ.setdefault("TOKENIZERS_PARALLELISM",   "false") # prevent HF deadlocks

# Tell PyTorch / PTL to use "spawn" instead of "fork" for subprocesses.
# "fork" is the default on Linux but unsupported on Windows; "spawn" is
# required here and is safe because PTL's own _wrapping_function is the
# entry-point for spawned children (no __main__ guard needed inside PTL).
os.environ.setdefault("PL_DISABLE_FORK", "1")

def _find_project_root(marker: str = "Interceptor") -> str:
    current = os.path.abspath(__file__)
    while True:
        parent = os.path.dirname(current)
        if os.path.basename(current) == marker:
            return current
        if parent == current:
            return os.getcwd()
        current = parent

_project_root = _find_project_root("Interceptor")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


import torch
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from rfdetr import RFDETRSmall
import supervision as sv
from PIL import Image as _Image
from src.development.modules.object_detection.config import training_config

def find_resume_checkpoint(output_dir: str):
    """
    Locate a PTL .ckpt file suitable for resuming training with trainer.fit().

    PTL checkpoints contain optimizer states, epoch counters, and scheduler
    state -- they are NOT loadable by RFDETRSmall.from_checkpoint().

    Files written by RF-DETR's build_trainer():
        last.ckpt               -- overwritten every epoch  <- preferred
        checkpoint_{epoch}.ckpt -- periodic archive copies

    Returns absolute path or None.
    """
    if not os.path.isdir(output_dir):
        return None

    candidates = (
        glob.glob(os.path.join(output_dir, "*.ckpt")) +
        glob.glob(os.path.join(output_dir, "*.pt"))
    )
    if not candidates:
        return None

    last = os.path.join(output_dir, "last.ckpt")
    if os.path.isfile(last):
        print(f"[Checkpoint/resume] {last}")
        return last

    latest = max(candidates, key=os.path.getmtime)
    print(f"[Checkpoint/resume] {latest}")
    return latest


def find_best_weights(output_dir: str):
    """
    Locate the best RF-DETR .pth weights file for inference / evaluation.

    RF-DETR's weight-saver callback writes these at the end of training:
        checkpoint_best_total.pth    -- best of (regular, EMA)  <- preferred
        checkpoint_best_ema.pth      -- best EMA weights
        checkpoint_best_regular.pth  -- best non-EMA weights

    These are RF-DETR-format files loadable by RFDETRSmall.from_checkpoint().
    They do NOT contain PTL optimizer state and cannot be used with trainer.fit().

    Returns absolute path or None.
    """
    if not os.path.isdir(output_dir):
        return None

    for name in [
        "checkpoint_best_total.pth",
        "checkpoint_best_ema.pth",
        "checkpoint_best_regular.pth",
    ]:
        path = os.path.join(output_dir, name)
        if os.path.isfile(path):
            print(f"[Checkpoint/weights] {path}")
            return path

    # Fallback: any .pth that is not a pretrain weights file
    pth_files = [
        p for p in glob.glob(os.path.join(output_dir, "*.pth"))
        if "pretrain" not in os.path.basename(p).lower()
        and "rf-detr" not in os.path.basename(p).lower()
    ]
    if pth_files:
        latest = max(pth_files, key=os.path.getmtime)
        print(f"[Checkpoint/weights] fallback: {latest}")
        return latest

    return None


# Alias so existing call-sites keep working
def find_latest_checkpoint(output_dir: str):
    return find_resume_checkpoint(output_dir)


def _read_completed_epochs(ckpt_path: str):
    """
    Return the number of *completed* epochs stored in a PTL checkpoint,
    or None if the file cannot be read.

    PTL stores `epoch` as a 0-based index of the last completed epoch,
    so completed_epochs = epoch + 1.
    """
    # Try weights_only=True first (safe, no FutureWarning on PyTorch >= 2.4)
    for weights_only in (True, False):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=weights_only)
            raw = ckpt.get("epoch")
            if raw is None:
                return None
            return int(raw) + 1   # PTL epoch is 0-indexed
        except Exception as exc:
            if weights_only:
                continue          # retry with weights_only=False
            print(f"[Checkpoint] Could not read epoch from {ckpt_path}: {exc}")
            return None


# ---------------------------------------------------------------------------
# Metrics plot
# ---------------------------------------------------------------------------

def plot_metrics(output_dir: str):
    """
    Parse RF-DETR's metrics.csv (written directly to output_dir by
    CSVLogger(name='', version='')) and save a training plot.
    """
    import csv

    metrics_file = os.path.join(output_dir, "metrics.csv")
    if not os.path.isfile(metrics_file):
        found = glob.glob(os.path.join(output_dir, "**", "metrics.csv"), recursive=True)
        if not found:
            print("[Metrics] No metrics.csv found — skipping plot.")
            return
        metrics_file = found[0]

    print(f"[Metrics] Reading {metrics_file}")

    epochs, train_loss, val_loss, map_vals, map50_vals = [], [], [], [], []

    def _safe_float(row, key):
        val = row.get(key, "").strip()
        return float(val) if val else float("nan")

    with open(metrics_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = row.get("epoch", "").strip()
            if not ep:
                continue
            epochs.append(float(ep))
            train_loss.append(_safe_float(row, "train_loss"))
            val_loss.append(_safe_float(row, "val_loss"))
            map_vals.append(_safe_float(row, "map"))
            map50_vals.append(_safe_float(row, "map_50"))

    if not epochs:
        print("[Metrics] metrics.csv has no data rows — skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("RF-DETR Small — Training Metrics", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.set_title("Loss");  ax.set_xlabel("Epoch");  ax.set_ylabel("Loss")
    ax.plot(epochs, train_loss, label="Train loss", color="#2196F3")
    ax.plot(epochs, val_loss,   label="Val loss",   color="#FF5722")
    ax.legend();  ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.set_title("mAP");  ax.set_xlabel("Epoch");  ax.set_ylabel("mAP")
    ax.plot(epochs, map_vals,   label="mAP@[.5:.95]", color="#4CAF50")
    ax.plot(epochs, map50_vals, label="mAP@0.50",      color="#9C27B0")
    ax.legend();  ax.grid(True, alpha=0.3);  ax.set_ylim(0, 1)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "metrics.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Metrics] Plot saved to {plot_path}")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

_NUM_WORKERS = min(4, (os.cpu_count() or 1))

class RFDTRTrainer:
    """
    RF-DETR Small trainer using the library's native .train() API.

    Worker strategy
    ---------------
    num_workers=4, persistent_workers=True, pin_memory=True
      • Workers are spawned once at the start of training and reused across
        epochs — avoids the per-epoch spawn cost that caused hangs.
      • pin_memory speeds up CPU→GPU transfers on CUDA.
      • prefetch_factor=2 keeps 2 batches pre-loaded per worker so the GPU
        never starves waiting for data.
    This requires the `if __name__ == '__main__':` guard (see bottom of file).

    Resume behaviour
    ----------------
    Pass resume=<abs_path> to TrainConfig; PTL handles all epoch/step
    offset bookkeeping automatically.
    """

    def __init__(self, num_classes: int, device, resolution: int = 640):
        if isinstance(device, int):
            device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        self.device     = device
        self.resolution = resolution

        print(f"[GPU] torch.cuda.is_available() = {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"[GPU] {props.name}  |  VRAM: {props.total_memory / 1e9:.1f} GB")

        self.model = RFDETRSmall(
            num_classes      = num_classes,
            pretrain_weights = "rf-detr-small.pth",
            device           = device,
            resolution       = resolution,
        )
        print(
            f"[RFDTRTrainer] device={device} | "
            f"resolution={resolution} | "
            f"num_classes={num_classes} | "
            f"num_workers={_NUM_WORKERS}"
        )

    def train(
        self,
        dataset_dir:      str,
        epochs:           int   = 150,
        batch_size:       int   = 8,
        grad_accum_steps: int   = 4,
        lr:               float = 1e-3,
        lr_encoder:       float = 1e-4,
        weight_decay:     float = 1e-4,
        name:             str   = "rf_dtr_run",
    ):
        output_dir = os.path.join("runs", name)
        os.makedirs(output_dir, exist_ok=True)

        # ---- Checkpoint resume ----------------------------------------
        resume_ckpt = find_latest_checkpoint(output_dir)

        if resume_ckpt:
            completed = _read_completed_epochs(resume_ckpt)
            if completed is not None:
                if completed >= epochs:
                    print(f"[Resume] Already completed {completed}/{epochs} epochs — nothing to do.")
                    return
                print(f"[Resume] Resuming from {resume_ckpt} (completed {completed}/{epochs} epochs)")
            else:
                print(f"[Resume] Resuming from {resume_ckpt} (epoch count unreadable)")
        else:
            print("[Resume] No checkpoint found — starting fresh.")

        print(
            f"\n[Train] epochs={epochs} | batch={batch_size} | "
            f"accum={grad_accum_steps} | "
            f"effective_batch={batch_size * grad_accum_steps} | "
            f"lr={lr} | lr_encoder={lr_encoder} | wd={weight_decay} | "
            f"resume={resume_ckpt is not None}\n"
        )

        train_kwargs = dict(
            dataset_dir         = dataset_dir,
            epochs              = epochs,
            batch_size          = batch_size,
            grad_accum_steps    = grad_accum_steps,
            lr                  = lr,
            lr_encoder          = lr_encoder,
            weight_decay        = weight_decay,
            output_dir          = output_dir,
            device              = self.device,
            num_workers         = _NUM_WORKERS,
            persistent_workers  = True,   # keep workers alive across epochs
            pin_memory          = True,   # faster CPU→GPU transfer
            prefetch_factor     = 2,      # pre-load 2 batches per worker
            eval_interval       = 5,      # validate every 5 epochs, not every 1
        )

        if resume_ckpt:
            train_kwargs["resume"] = os.path.abspath(resume_ckpt)

        self.model.train(**train_kwargs)
        plot_metrics(output_dir)

    def validate(
        self,
        dataset_dir:    str,
        split:          str   = "test",   # "test" or "valid"
        threshold:      float = 0.5,
        batch_size:     int   = 8,
        checkpoint_path: str | None = None,
    ) -> dict:
        """
        COCO-metric evaluation using supervision's MeanAveragePrecision.

        RFDETRSmall.evaluate() was removed in the current RF-DETR release.
        This replacement runs model.predict() over every image in the chosen
        split and computes mAP@[.5:.95], mAP@50, and mAP@75 via the
        supervision metrics stack (already a hard dependency of RF-DETR).

        Parameters
        ----------
        dataset_dir     : root folder that contains train/ valid/ test/ subdirs
        split           : which sub-split to evaluate ("test" or "valid")
        threshold       : confidence threshold passed to model.predict()
        batch_size      : number of images per predict() call
        checkpoint_path : optional path to a .pth checkpoint to load before
                          evaluation; if None the model's current weights are used

        Returns
        -------
        dict with keys: map50_95, map50, map75
        """

        if checkpoint_path is not None:
            if checkpoint_path.endswith(".ckpt"):
                raise ValueError(
                    f"[Validate] '{os.path.basename(checkpoint_path)}' is a PTL "
                    "training checkpoint — it cannot be loaded for inference.\n"
                    "Pass a RF-DETR .pth file instead "
                    "(checkpoint_best_total.pth / checkpoint_best_ema.pth / "
                    "checkpoint_best_regular.pth).\n"
                    f"These are written to: {os.path.dirname(checkpoint_path)}"
                )
            print(f"[Validate] Loading weights: {checkpoint_path}")
            self.model = RFDETRSmall.from_checkpoint(
                checkpoint_path,
                num_classes = self.model.model_config.num_classes,
                device      = self.device,
                resolution  = self.resolution,
            )
        else:
            print("[Validate] Using current in-memory model weights.")

        split_dir = os.path.join(dataset_dir, split)

        if not os.path.isdir(split_dir):
            alt = "valid" if split == "test" else "test"
            print(f"[Validate] '{split}' split not found — trying '{alt}'")
            split_dir = os.path.join(dataset_dir, alt)

        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"[Validate] Split directory not found: {split_dir}"
            )

        images_dir = (
            os.path.join(split_dir, "images")
            if os.path.isdir(os.path.join(split_dir, "images"))
            else split_dir
        )
        ann_path = os.path.join(split_dir, "_annotations.coco.json")

        print(f"[Validate] Split : {split_dir}")

        if os.path.isfile(ann_path):
            print(f"[Validate] Format : COCO  ({ann_path})")
            dataset = sv.DetectionDataset.from_coco(
                images_directory_path = images_dir,
                annotations_path      = ann_path,
            )
        else:
            labels_dir = os.path.join(split_dir, "labels")
            if not os.path.isdir(labels_dir):
                raise FileNotFoundError(
                    f"[Validate] Neither COCO annotations ({ann_path}) nor "
                    f"YOLO labels directory ({labels_dir}) found. "
                    "Check your dataset layout."
                )

            yaml_path = None
            for candidate in [
                os.path.join(dataset_dir, "data.yaml"),
                os.path.join(os.path.dirname(dataset_dir), "data.yaml"),
                os.path.join(dataset_dir, "dataset.yaml"),
            ]:
                if os.path.isfile(candidate):
                    yaml_path = candidate
                    break

            if yaml_path is None:
                import tempfile, yaml as _yaml
                tmp = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".yaml", delete=False
                )
                _yaml.dump(
                    {"names": {i: str(i) for i in range(
                        self.model.model_config.num_classes
                    )}},
                    tmp,
                )
                tmp.close()
                yaml_path = tmp.name
                print(f"[Validate] No data.yaml found — using temp yaml: {yaml_path}")
            else:
                print(f"[Validate] Format : YOLO  (yaml={yaml_path})")

            dataset = sv.DetectionDataset.from_yolo(
                images_directory_path      = images_dir,
                annotations_directory_path = labels_dir,
                data_yaml_path             = yaml_path,
            )

        image_paths = sorted(dataset.image_paths)
        print(f"[Validate] {len(image_paths)} images to evaluate")

        map_metric = sv.metrics.MeanAveragePrecision(class_agnostic=True)

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_imgs  = [_Image.open(p).convert("RGB") for p in batch_paths]

            preds = self.model.predict(batch_imgs, threshold=threshold)
            if not isinstance(preds, list):
                preds = [preds]

            gts = [dataset.annotations[p] for p in batch_paths]

            map_metric.update(predictions=preds, targets=gts)

            done = min(i + batch_size, len(image_paths))
            print(f"\r[Validate] {done}/{len(image_paths)}", end="", flush=True)

        print()  # newline after progress

        result = map_metric.compute()

        metrics = {
            "map50_95" : round(result.map50_95, 4),
            "map50"    : round(result.map50,    4),
            "map75"    : round(result.map75,    4),
        }

        print("\n[Validate] Results")
        print(f"  mAP@[.5:.95] : {metrics['map50_95']:.4f}")
        print(f"  mAP@0.50     : {metrics['map50']:.4f}")
        print(f"  mAP@0.75     : {metrics['map75']:.4f}")

        return metrics

if __name__ == "__main__":

    params = training_config

    dataset_dir = os.path.abspath(params["yaml"])
    if os.path.isfile(dataset_dir):
        dataset_dir = os.path.dirname(dataset_dir)
    print(f"[Dataset] {dataset_dir}")

    batch_size       = 8
    grad_accum_steps = max(params["batch"] // batch_size, 1)

    trainer = RFDTRTrainer(
        num_classes = params["num_classes"],
        device      = params["device"],
        resolution  = params["imgsz"],
    )

    trainer.train(
        dataset_dir      = dataset_dir,
        epochs           = 150,
        batch_size       = batch_size,
        grad_accum_steps = grad_accum_steps,
        lr               = params.get("lr0",          1e-3),
        lr_encoder       = params.get("lr0",          1e-3) * 0.1,
        weight_decay     = params.get("weight_decay", 1e-4),
        name             = params["name"],
    )

    run_dir   = os.path.join("runs", params["name"])
    best_pth  = find_best_weights(run_dir)
    if best_pth is None:
        print(f"[Validate] No .pth weights found in {run_dir} — skipping eval.")
    else:
        metrics = trainer.validate(
            dataset_dir     = dataset_dir,
            split           = "test",
            threshold       = 0.1,
            batch_size      = 8,
            checkpoint_path = best_pth,
        )
        print(metrics)
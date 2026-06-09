import sys
sys.path.append('./')

import os
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
from torch.cuda.amp import GradScaler, autocast
from rfdetr import RFDETRSmall
from rfdetr.util.coco_utils import get_coco_api_from_dataset
from rfdetr.datasets import build_dataset, build_dataloader
from src.development.modules.object_detection.config import training_config

class RFDTRTrainer:
    """
    RF-DETR (Small) trainer that mirrors the YOLOTrainer API while adding:

      • Cosine-annealing LR with a short linear warm-up
      • AdamW with decoupled weight-decay (biases / norms excluded)
      • Mixed-precision training (AMP) via GradScaler
      • Gradient clipping to stabilise early training
      • EMA (exponential moving average) of model weights for a better
        validation checkpoint
    """

    def __init__(
        self,
        num_classes: int,
        device: int | str,
        pretrained: bool = True,
        ema_decay: float = 0.9999,
        grad_clip: float = 0.1,
    ):
        """
        Parameters
        ----------
        num_classes : int
            Number of object classes (excluding background).
        device : int | str
            CUDA device index (int) or ``"cpu"`` / ``"cuda:N"`` string.
        pretrained : bool
            Load COCO-pretrained backbone weights.
        ema_decay : float
            Decay factor for the EMA shadow model (0 = no EMA).
        grad_clip : float
            Max L2-norm for gradient clipping. Set to 0 to disable.
        """
        if isinstance(device, int):
            device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.grad_clip = grad_clip
        self.ema_decay = ema_decay

        # RF-DETR Small — lightweight DETR variant suited for edge / small-
        # dataset scenarios (~28 M params, ResNet-18 backbone by default).
        self.model = RFDETRSmall(
            num_classes=num_classes,
            pretrained=pretrained,
        ).to(self.device)

        # EMA shadow copy (kept on CPU to save VRAM; synced each step)
        self.ema_model = None
        if ema_decay > 0:
            import copy
            self.ema_model = copy.deepcopy(self.model).cpu()
            for p in self.ema_model.parameters():
                p.requires_grad_(False)

        # AMP scaler (no-op when device is CPU)
        self.scaler = GradScaler(enabled=self.device.type == "cuda")

        print(
            f"[RFDTRTrainer] device={self.device}  "
            f"ema_decay={ema_decay}  grad_clip={grad_clip}"
        )

    def _build_optimizer(
        self,
        lr0: float,
        weight_decay: float,
    ) -> AdamW:
        """
        AdamW with layer-wise parameter grouping:
          • Backbone parameters   → 0.1 × lr0  (fine-tune slowly)
          • Transformer/head      → lr0
          • Biases & LayerNorm    → no weight decay (standard practice)
        """
        decay_params, no_decay_params = [], []
        backbone_decay, backbone_no_decay = [], []

        no_decay_names = {"bias", "norm", "layernorm", "layer_norm"}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            is_no_decay = any(nd in name.lower() for nd in no_decay_names)
            is_backbone = name.startswith("backbone")

            if is_backbone:
                (backbone_no_decay if is_no_decay else backbone_decay).append(param)
            else:
                (no_decay_params if is_no_decay else decay_params).append(param)

        param_groups = [
            {"params": decay_params,          "lr": lr0,        "weight_decay": weight_decay},
            {"params": no_decay_params,        "lr": lr0,        "weight_decay": 0.0},
            {"params": backbone_decay,         "lr": lr0 * 0.1,  "weight_decay": weight_decay},
            {"params": backbone_no_decay,      "lr": lr0 * 0.1,  "weight_decay": 0.0},
        ]
        return AdamW(param_groups, lr=lr0, betas=(0.9, 0.999), eps=1e-8)

    def _build_scheduler(
        self,
        optimizer: AdamW,
        epochs: int,
        warmup_epochs: int,
        lrf: float,
    ) -> SequentialLR:
        """
        Two-phase LR schedule:
          1. Linear warm-up  : epoch 0 → warmup_epochs  (1e-6 → lr0)
          2. Cosine annealing: warmup_epochs → epochs    (lr0 → lr0 × lrf)

        ``lrf`` mirrors the YOLO convention: final LR = lr0 × lrf.
        """
        warmup = LinearLR(
            optimizer,
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=max(warmup_epochs, 1),
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(epochs - warmup_epochs, 1),
            eta_min=0.0,          # scheduler brings LR to ~0; lrf applied below
        )

        # Scale the minimum LR by lrf for each param group after cosine phase.
        # We do this by overriding eta_min per-group once cosine is attached.
        for group in optimizer.param_groups:
            group["eta_min"] = group["lr"] * lrf

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
        return scheduler

    @torch.no_grad()
    def _update_ema(self):
        if self.ema_model is None:
            return
        d = self.ema_decay
        for ema_p, model_p in zip(
            self.ema_model.parameters(), self.model.parameters()
        ):
            ema_p.data.mul_(d).add_(model_p.data.cpu(), alpha=1.0 - d)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader,
        val_loader,
        epochs: int,
        lr0: float = 1e-4,
        lrf: float = 0.01,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 3,
        name: str = "rf_dtr_run",
    ):
        """
        Train RF-DETR Small.

        Parameters
        ----------
        train_loader : DataLoader
            Yields ``(images, targets)`` batches in COCO format.
        val_loader : DataLoader
            Used for per-epoch validation loss tracking.
        epochs : int
            Total training epochs.
        lr0 : float
            Peak (initial post-warmup) learning rate.
        lrf : float
            Final LR as a fraction of lr0 (cosine decay target).
        weight_decay : float
            AdamW weight decay coefficient.
        warmup_epochs : int
            Number of linear warm-up epochs before cosine decay begins.
        name : str
            Run / checkpoint name prefix.
        """
        optimizer = self._build_optimizer(lr0, weight_decay)
        scheduler = self._build_scheduler(optimizer, epochs, warmup_epochs, lrf)

        best_val_loss = math.inf
        os.makedirs(f"runs/{name}", exist_ok=True)

        print(
            f"\n[Train] epochs={epochs}  lr0={lr0}  lrf={lrf}  "
            f"wd={weight_decay}  warmup={warmup_epochs}\n"
        )

        for epoch in range(1, epochs + 1):
            # ---- training pass ----------------------------------------
            self.model.train()
            train_loss = 0.0

            for batch_idx, (images, targets) in enumerate(train_loader):
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                optimizer.zero_grad(set_to_none=True)   # slightly faster than zero_grad()

                with autocast(enabled=self.device.type == "cuda"):
                    loss_dict = self.model(images, targets)
                    loss = sum(loss_dict.values())

                self.scaler.scale(loss).backward()

                # Gradient clipping (unscale first for correct norm)
                if self.grad_clip > 0:
                    self.scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )

                self.scaler.step(optimizer)
                self.scaler.update()
                self._update_ema()

                train_loss += loss.item()

            scheduler.step()

            avg_train_loss = train_loss / max(len(train_loader), 1)
            current_lr = optimizer.param_groups[0]["lr"]

            # ---- validation pass --------------------------------------
            val_loss = self._val_loss(val_loader)

            print(
                f"Epoch [{epoch:>3}/{epochs}]  "
                f"train_loss={avg_train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"lr={current_lr:.2e}"
            )

            # ---- checkpoint -------------------------------------------
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt = {
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "ema_state": self.ema_model.state_dict() if self.ema_model else None,
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "val_loss": best_val_loss,
                }
                torch.save(ckpt, f"runs/{name}/best.pt")
                print(f"  ✓ Saved best checkpoint  (val_loss={best_val_loss:.4f})")

        print(f"\n[Train] Done. Best val_loss={best_val_loss:.4f}")

    @torch.no_grad()
    def _val_loss(self, val_loader) -> float:
        """Quick validation-loss pass (model in eval mode, no grad)."""
        self.model.eval()
        total = 0.0
        for images, targets in val_loader:
            images  = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            with autocast(enabled=self.device.type == "cuda"):
                loss_dict = self.model(images, targets)
            total += sum(loss_dict.values()).item()
        return total / max(len(val_loader), 1)

    def validate(self, val_loader=None):
        """
        Full COCO-metric evaluation using the EMA model if available,
        otherwise the live model.  Mirrors YOLOTrainer.validate() in spirit.
        """
        eval_model = self.ema_model.to(self.device) if self.ema_model else self.model
        eval_model.eval()

        # RF-DETR exposes its own evaluator; delegate to it.
        metrics = eval_model.evaluate(val_loader)

        # Move EMA back to CPU after eval to keep VRAM free.
        if self.ema_model:
            self.ema_model.cpu()

        return metrics


if __name__ == "__main__":

    params = training_config

    # RF-DETR needs dataset-specific DataLoaders; build them from the YAML path
    # the same way the config supplies it for YOLO.

    train_ds = build_dataset(root=params["yaml"], split="train")
    val_ds   = build_dataset(root=params["yaml"], split="val")

    train_loader = build_dataloader(
        train_ds,
        batch_size=params["batch"],
        shuffle=True,
        num_workers=4,
    )
    val_loader = build_dataloader(
        val_ds,
        batch_size=params["batch"],
        shuffle=False,
        num_workers=4,
    )

    trainer = RFDTRTrainer(
        num_classes=params["num_classes"],
        device=params["device"],
    )

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=params["epochs"],
        lr0=params.get("lr0", 1e-4),
        lrf=params.get("lrf", 0.01),
        weight_decay=params.get("weight_decay", 1e-4),
        warmup_epochs=params.get("warmup_epochs", 3),
        name=params["name"],
    )

    metrics = trainer.validate(val_loader)
    print(metrics)
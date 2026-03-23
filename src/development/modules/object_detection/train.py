import os
import sys

sys.path.append("./")

from src.development.modules.object_detection.train_logging import save_run_config, plot_results
from src.development.modules.object_detection.config import project_config, training_config
from ultralytics import YOLO


class YoloTrainer:
    def __init__(self, path_dataset: str, path_yaml: str):
        # Check for last checkpoint
        self.last_ckpt = os.path.join(
            project_config["runs"],
            project_config["name"],
            "weights",
            "last.pt",
        )

        if os.path.exists(self.last_ckpt):
            print(f"🔁 Resuming training from {self.last_ckpt}")
            self.model = YOLO(self.last_ckpt)
            self.resume = True
        else:
            print("🆕 Starting new training run")
            self.model = YOLO(training_config["model"])
            self.resume = False

        self.yaml = os.path.join(path_dataset, path_yaml)

        # Placeholder run_dir; will be updated after YOLO starts training
        self.run_dir = None
        self.status_dir = None

        # Register callback
        self.model.add_callback("on_fit_epoch_end", self.on_epoch_end)

    def on_epoch_end(self, _):
        # Dynamically get the correct YOLO run folder
        if hasattr(self.model, "trainer") and hasattr(self.model.trainer, "run_dir"):
            self.run_dir = self.model.trainer.run_dir
            self.status_dir = os.path.join(self.run_dir, "status")
            os.makedirs(self.status_dir, exist_ok=True)

            # YOLO uses metrics.csv
            metrics_csv = os.path.join(self.run_dir, "metrics.csv")
            if os.path.exists(metrics_csv):
                plot_results(metrics_csv, self.status_dir)
                print(f"📊 Updated plots at {self.status_dir}/training_metrics.png")
            else:
                print(f"⚠️ metrics.csv not found yet at {self.run_dir}")

    def train(self):
        self.model.train(
            data=self.yaml,
            epochs=training_config["epochs"],
            batch=training_config["batch"],
            imgsz=training_config["size"],
            project=project_config["runs"],
            name=project_config["name"],
            exist_ok=True,
            device=project_config["device"],
            resume=self.resume,
        )


if __name__ == "__main__":
    trainer = YoloTrainer(
        path_dataset=project_config["path"],
        path_yaml="data.yaml",
    )
    trainer.train()

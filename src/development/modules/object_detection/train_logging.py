import os
import math
import pandas as pd
import matplotlib.pyplot as plt

from src.development.modules.object_detection.config import project_config, training_config


def save_run_config(status_dir):
    os.makedirs(status_dir, exist_ok=True)
    with open(os.path.join(status_dir, "training_config.txt"), "w") as f:
        f.write("PROJECT CONFIG\n")
        for k, v in project_config.items():
            f.write(f"{k}: {v}\n")
        f.write("\nTRAINING CONFIG\n")
        for k, v in training_config.items():
            f.write(f"{k}: {v}\n")


def plot_results(metrics_csv, status_dir):
    if not os.path.exists(metrics_csv):
        print(f"⚠️ Metrics CSV not found at {metrics_csv}")
        return

    df = pd.read_csv(metrics_csv)

    metrics = [c for c in df.columns if c != "epoch"]
    n = len(metrics)

    cols = 2
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(20, 6 * rows))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        axes[i].plot(df["epoch"], df[metric], marker='o')
        axes[i].set_title(metric)
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel(metric)
        axes[i].grid(True)

    # Remove extra axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(status_dir, "training_metrics.png"))
    plt.close()

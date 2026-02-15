#!/usr/bin/env python3
"""Centralized utilities for writing results."""

# --- Import ---------------------------------------------------------------
# Required imports for serialization, CSV, and context manager.

import argparse
import csv
import json
from contextlib import contextmanager


# --- Constants ------------------------------------------------------------
# Default training metrics used to complete payloads.

DEFAULT_TRAIN_METRICS = {
    "final_train_loss": 0.0,
    "final_val_loss": 0.0,
    "val_top1": 0.0,
    "val_top3": 0.0,
    "test_loss": 0.0,
    "test_top1": 0.0,
    "test_top3": 0.0,
}


# --- CLI arguments --------------------------------------------------------
# Argument parser definition (consistent with other modules).

def build_parser():
    parser = argparse.ArgumentParser(
        prog="results_writer",
        description="Utility for writing artifacts and reports.",
    )
    return parser


# --- Training artifact writing -------------------------------------------
# Utility to export training results and reports.

def write_training_artifacts(
    run_dir,
    run_id,
    run_params,
    labels,
    metrics,
    confusion_matrix,
):
    """Write labels, training metrics, and confusion matrix."""
    (run_dir / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")

    payload = {
        "run_id": run_id,
        "params": run_params,
        "metrics": {
            **DEFAULT_TRAIN_METRICS,
            **metrics,
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with (run_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", *labels])
        for lbl, row in zip(labels, confusion_matrix):
            writer.writerow([lbl, *row])


# --- Inference result writing --------------------------------------------
# Utility to write output and metrics from inference stage.

@contextmanager
def prediction_results_writer(out_csv):
    """Open labels.csv and provide a ready-to-use writer."""
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "pred_label", "pred_conf", "top3"])
        yield writer


def write_inference_metrics(model_path, run_metrics):
    """Write aggregated inference metrics."""
    (model_path / "inference_metrics.json").write_text(
        json.dumps(run_metrics, indent=2),
        encoding="utf-8",
    )


# --- Training plot --------------------------------------------------------
# Utility to generate the final run plot.

def write_training_plot(run_dir, epoch_metrics, run_params):
    """Generate loss/accuracy plot for the training run."""
    if not epoch_metrics:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs_idx = [int(m["epoch"]) for m in epoch_metrics]
        train_losses = [float(m["train_loss"]) for m in epoch_metrics]
        val_losses = [float(m["val_loss"]) for m in epoch_metrics]
        val_top1s = [float(m["val_top1"]) for m in epoch_metrics]

        fig, ax1 = plt.subplots(figsize=(9, 5.5))
        ax1.plot(epochs_idx, train_losses, label="train loss", marker="o")
        ax1.plot(epochs_idx, val_losses, label="val loss", marker="o")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(epochs_idx, val_top1s, label="val top1", marker="s", color="tab:green")
        ax2.set_ylabel("val top1")
        ax2.set_ylim(0.0, 1.0)

        lines, labels_ = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels_ + labels2, loc="best")

        params_txt = (
            f"epochs={run_params['epochs']} | "
            f"batch_size={run_params['batch_size']} | "
            f"img_size={run_params['img_size']} | "
            f"lr={run_params['lr']} | "
            f"seed={run_params['seed']} | "
            f"backend={run_params['backend']}"
        )
        fig.text(
            0.5,
            0.01,
            params_txt,
            ha="center",
            va="bottom",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f4f4f4", "alpha": 0.9},
        )

        plot_path = run_dir / "loss_plot.png"
        fig.tight_layout(rect=[0.0, 0.08, 1.0, 1.0])
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] Plot not generated: {exc}")


# Note: this module is intended as a support library, without a script entrypoint.

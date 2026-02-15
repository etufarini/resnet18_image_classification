#!/usr/bin/env python3
"""Simple batch inference script with ONNX Runtime."""

# --- Import ---------------------------------------------------------------
# Standard dependencies for CLI, file system, and timing.
import argparse
import os
import time
from pathlib import Path

# Scientific and ONNX inference dependencies.
import numpy as np
import onnxruntime as ort
from PIL import Image

# Local modules for backend selection and result writing.
from backend import BackendError, resolve_ort_providers
from results_writer import prediction_results_writer, write_inference_metrics

# --- Constants ------------------------------------------------------------
# Shared preprocessing and backend configuration.
BACKEND_CHOICES = ["auto", "cpu", "cuda", "rocm"]
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# --- CLI ------------------------------------------------------------------
# CLI argument definition.
def build_parser():
    parser = argparse.ArgumentParser(
        prog="inference",
        description="Batch inference on a folder (non-recursive).",
    )
    parser.add_argument("--model_dir", required=True, help="Run folder with model.onnx and labels.txt.")
    parser.add_argument("--input_dir", required=True, help="Image folder.")
    parser.add_argument("--threshold", type=float, default=0.50, help="Minimum confidence to accept prediction.")
    parser.add_argument(
        "--backend",
        choices=BACKEND_CHOICES,
        default="auto",
        help="Inference backend (auto|cpu|cuda|rocm).",
    )
    return parser


# --- Support: model -------------------------------------------------------
# Load labels, ONNX model, and ORT session.
def _load_model_and_session(model_path, backend):
    labels_file = model_path / "labels.txt"
    onnx_file = model_path / "model.onnx"

    if not labels_file.exists():
        raise FileNotFoundError(f"missing labels.txt in: {model_path}")
    if not onnx_file.exists():
        raise FileNotFoundError(f"missing model.onnx in: {model_path}")

    labels = [
        line.strip()
        for line in labels_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not labels:
        raise ValueError("empty labels.txt")

    available_providers = ort.get_available_providers()
    try:
        providers = resolve_ort_providers(backend, available_providers)
    except BackendError as exc:
        raise SystemExit(
            f"Invalid backend for inference: {exc}. "
            f"Available: {', '.join(available_providers)}"
        ) from exc

    try:
        session = ort.InferenceSession(str(onnx_file), providers=providers)
    except Exception as exc:
        # CPU fallback only for the known external-data model case.
        if "model_path must not be empty" not in str(exc):
            raise
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(
            str(onnx_file),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return labels, session, input_name, output_name


# --- Support: prediction --------------------------------------------------
# Preprocess + forward + postprocess for a single image.
def _predict_single(session, input_name, output_name, img_path, labels, threshold):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = ((arr - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)

    logits = session.run([output_name], {input_name: arr})[0].astype(np.float32)
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = (exp / np.sum(exp, axis=1, keepdims=True))[0]

    top_k = min(3, len(labels))
    top_idx = np.argsort(probs)[::-1][:top_k]
    pred_idx = int(top_idx[0])
    pred_conf = float(probs[pred_idx])
    pred_label = labels[pred_idx] if pred_conf >= threshold else "unknown"
    top3 = ";".join(f"{labels[i]}:{probs[i]:.4f}" for i in top_idx)
    return pred_label, pred_conf, top3


# --- Inference ------------------------------------------------------------
# Main flow: input validation, image loop, csv, and metrics.
def inference(model_dir, input_dir, threshold, backend):
    model_path = Path(model_dir)
    input_path = Path(input_dir)
    out_csv = model_path / "labels.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"model_dir not found: {model_path}")
    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"invalid input_dir: {input_path}")

    images = [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    labels, session, input_name, output_name = _load_model_and_session(model_path, backend)

    processed = 0
    failed = 0
    start = time.perf_counter()

    with prediction_results_writer(out_csv) as writer:
        for img_path in images:
            processed += 1
            rel_path = os.path.relpath(img_path, input_path)
            try:
                pred_label, pred_conf, top3 = _predict_single(
                    session=session,
                    input_name=input_name,
                    output_name=output_name,
                    img_path=img_path,
                    labels=labels,
                    threshold=threshold,
                )
            except Exception:
                failed += 1
                pred_label = "error"
                pred_conf = 0.0
                top3 = ""
            writer.writerow([rel_path, pred_label, f"{pred_conf:.2f}", top3])

    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / processed * 1000) if processed > 0 else 0.0

    run_metrics = {
        "images_total": len(images),
        "images_processed": processed,
        "images_failed": failed,
        "avg_ms_per_image": round(avg_ms, 3),
        "threshold": threshold,
    }
    write_inference_metrics(model_path, run_metrics)

    print(f"labels.csv created at: {out_csv}")
    print(f"Images: {processed}, avg_ms_per_image: {avg_ms:.2f}")


# --- Main -----------------------------------------------------------------
# CLI entrypoint.
def main():
    args = build_parser().parse_args()
    inference(
        model_dir=args.model_dir,
        input_dir=args.input_dir,
        threshold=args.threshold,
        backend=args.backend,
    )


# --- Script start ---------------------------------------------------------
# Direct start from shell.
if __name__ == "__main__":
    main()

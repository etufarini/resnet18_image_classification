# Resnet18 Image Classification
Two-stage pipeline: training (PyTorch -> ONNX) and batch inference on an image folder with metrics and run logs.

## Minimal Structure
- `src/train.py` -> training script and logic
- `src/inference.py` -> inference script and logic
- `src/backend.py` -> backend resolution (CPU/CUDA/ROCm) for train and inference
- `src/augment.py` -> offline data augmentation on local dataset
- `src/results_writer.py` -> utility for writing outputs/metrics in CSV/JSON format
- `artifacts/runs/<timestamp>/` -> generated artifacts
- `data/` -> dataset with `train/val/test`
- `artifacts/inputs/` -> input images for inference (non-recursive)
- `scripts/` -> quick scripts (`setup.sh`, `run_train.sh`, `run_inference.sh`, `run_augment.sh`)

## `src/` File Overview
- `src/augment.py`: generates augmented image copies inside `train/val/test`.
- `src/backend.py`: handles device/provider selection and backend checks (`auto|cpu|cuda|rocm`).
- `src/inference.py`: runs ONNX Runtime inference on an image folder and saves `labels.csv` + metrics.
- `src/train.py`: trains ResNet18, exports ONNX, and saves run artifacts (`metrics.json`, `labels.txt`, `confusion_matrix.csv`, `loss_plot.png`).
- `src/results_writer.py`: helper for writing structured inference/training outputs.

## Environment Setup (venv)
Use a virtual environment and install dependencies from repo root.

Quick setup (CPU/CUDA):

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

ROCm setup (AMD):

```bash
bash scripts/setup_rocm.sh
source .venv/bin/activate
```

## Dataset
`--data_dir` must point to a folder with this structure:

```text
data/
  train/
    class_a/
    class_b/
  val/
    class_a/
    class_b/
  test/
    class_a/
    class_b/
```

## Data Augmentation (Offline, Manual)
Training in `src/train.py` does **not** apply online augmentation.
It uses a deterministic pipeline (`Resize -> ToTensor -> Normalize`).

If you want augmentation, run it **before training** with `src/augment.py`:

```bash
python3 src/augment.py --root data --per-image 2 --seed 42
```

Equivalent quick script:

```bash
bash scripts/run_augment.sh -r data -p 2 -s 42
```

Script flags:
- `-r` dataset root (maps to `--root`)
- `-p` augmented copies per image (maps to `--per-image`)
- `-s` random seed (maps to `--seed`)

Main options:
- `--root`: dataset root (default `data`)
- `--per-image`: augmented copies per image (default `2`)
- `--seed`: random seed (default `42`)

Note: the script avoids re-augmenting files that already contain `_aug` in the name.

## Training
Example:

```bash
python3 src/train.py \
  --data_dir data \
  --epochs 5 \
  --batch_size 32 \
  --img_size 224 \
  --lr 1e-3 \
  --seed 42 \
  --backend auto
```

Parameters:
- `--data_dir` (required)
- `--epochs` (default `1`)
- `--batch_size` (default `8`)
- `--img_size` (default `224`)
- `--lr` (default `1e-3`)
- `--seed` (default `42`)
- `--backend` (default `auto`) values: `auto|cpu|cuda|rocm`

Note: the model uses a pretrained ResNet18 with early layers frozen, and trains `layer4` + `fc`.

## Inference
Example:

```bash
python3 src/inference.py \
  --model_dir artifacts/runs/<timestamp> \
  --input_dir artifacts/inputs \
  --threshold 0.50 \
  --backend auto
```

Parameters:
- `--model_dir` (required): run folder containing `model.onnx` and `labels.txt`
- `--input_dir` (required): image folder (non-recursive)
- `--threshold` (default `0.50`)
- `--backend` (default `auto`) values: `auto|cpu|cuda|rocm`

## Quick Scripts
- `bash scripts/run_augment.sh -r data -p 2 -s 42`
- `bash scripts/run_train.sh -e 5 -b 16 -i 224 -l 1e-3 -s 42 -k auto`
- `bash scripts/run_inference.sh -i artifacts/inputs -t 0.50 -k auto`

## Expected Outputs
Inside `artifacts/runs/<timestamp>/`:
- `model.onnx`
- `labels.txt`
- `metrics.json`
- `confusion_matrix.csv`
- `loss_plot.png`
- `labels.csv`
- `inference_metrics.json`

## Dataset License
For the Kaggle `pest-dataset` (declared as `CC0: Public Domain`), see:
- `DATA_LICENSE.md`

#!/usr/bin/env bash

# --- Shell options --------------------------------------------------------
# Safe shell options: fail fast and stop on undefined variables.

set -euo pipefail

# --- Default configuration ------------------------------------------------
# Default parameters used when not provided via CLI flags.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 1) Default parameters (can be overridden by flags)
DATA_DIR="$ROOT/data"
EPOCHS=5
BATCH_SIZE=16
IMG_SIZE=224
LR=1e-3
SEED=42
NUM_WORKERS=4
BACKEND="auto"

# --- Help -----------------------------------------------------------------
# Help message for script usage and available options.

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]
  -d DATA_DIR    Dataset path with train/val/test (default: $DATA_DIR)
  -e EPOCHS      Epochs (default: $EPOCHS)
  -b BATCH_SIZE  Batch size (default: $BATCH_SIZE)
  -i IMG_SIZE    Image size (default: $IMG_SIZE)
  -l LR          Learning rate (default: $LR)
  -s SEED        Seed (default: $SEED)
  -w WORKERS     DataLoader workers (default: $NUM_WORKERS)
  -k BACKEND     Backend (auto|cpu|cuda|rocm) (default: $BACKEND)
EOF
}

# --- CLI parsing ----------------------------------------------------------
# Parse command line options.

while getopts ":d:e:b:i:l:s:w:k:h" opt; do
  case "$opt" in
    d) DATA_DIR="$OPTARG" ;;
    e) EPOCHS="$OPTARG" ;;
    b) BATCH_SIZE="$OPTARG" ;;
    i) IMG_SIZE="$OPTARG" ;;
    l) LR="$OPTARG" ;;
    s) SEED="$OPTARG" ;;
    w) NUM_WORKERS="$OPTARG" ;;
    k) BACKEND="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 1 ;;
    :) echo "Option -$OPTARG requires a value" >&2; usage; exit 1 ;;
  esac
done

# --- Execution ------------------------------------------------------------
# Start Python training with resolved parameters.

python3 "$ROOT/src/train.py" \
  --data_dir "$DATA_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --img_size "$IMG_SIZE" \
  --lr "$LR" \
  --seed "$SEED" \
  --num_workers "$NUM_WORKERS" \
  --backend "$BACKEND"

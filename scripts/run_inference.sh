#!/usr/bin/env bash

# --- Shell options --------------------------------------------------------
# Safe shell options: fail fast and stop on undefined variables.

set -euo pipefail

# --- Default configuration ------------------------------------------------
# Default parameters used when not provided via CLI flags.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 1) Default parameters (empty MODEL_DIR = use latest run)
RUNS_DIR="$ROOT/artifacts/runs"
INPUT_DIR="$ROOT/artifacts/inputs"
MODEL_DIR=""
THRESHOLD="0.50"
BACKEND="auto"

# --- Help -----------------------------------------------------------------
# Help message for script usage and available options.

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]
  -m MODEL_DIR   Run path (default: latest run in $RUNS_DIR)
  -i INPUT_DIR   Image folder (default: $INPUT_DIR)
  -t THRESHOLD   Confidence threshold (default: $THRESHOLD)
  -k BACKEND     Backend (auto|cpu|cuda|rocm) (default: $BACKEND)
EOF
}

# --- CLI parsing ----------------------------------------------------------
# Parse command line options.

while getopts ":m:i:t:k:h" opt; do
  case "$opt" in
    m) MODEL_DIR="$OPTARG" ;;
    i) INPUT_DIR="$OPTARG" ;;
    t) THRESHOLD="$OPTARG" ;;
    k) BACKEND="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 1 ;;
    :) echo "Option -$OPTARG requires a value" >&2; usage; exit 1 ;;
  esac
done

# --- Run resolution -------------------------------------------------------
# Select run: use provided one or latest available.

if [ ! -d "$RUNS_DIR" ]; then
  echo "Runs directory not found: $RUNS_DIR" >&2
  exit 1
fi

if [ -z "$MODEL_DIR" ]; then
  LATEST_RUN=$(ls -1 "$RUNS_DIR" | sort | tail -n 1)
  if [ -z "$LATEST_RUN" ]; then
    echo "No runs found in: $RUNS_DIR" >&2
    exit 1
  fi
  MODEL_DIR="$RUNS_DIR/$LATEST_RUN"
fi

# --- Execution ------------------------------------------------------------
# Start Python inference with resolved parameters.

python3 "$ROOT/src/inference.py" \
  --model_dir "$MODEL_DIR" \
  --input_dir "$INPUT_DIR" \
  --threshold "$THRESHOLD" \
  --backend "$BACKEND"

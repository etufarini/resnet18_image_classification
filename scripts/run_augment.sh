#!/usr/bin/env bash

# --- Shell options --------------------------------------------------------
# Use safe shell options: fail fast and stop on undefined variables.

set -euo pipefail

# --- Default configuration ------------------------------------------------
# Default parameters that can be overridden via CLI flags.

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_root="$root_dir/data"
per_image=2
seed=42

# --- Help -----------------------------------------------------------------
# Show usage and available options.

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]
  -r DATA_ROOT   Dataset root with train/val/test (default: $dataset_root)
  -p PER_IMAGE   Number of augmented images per original (default: $per_image)
  -s SEED        Random seed (default: $seed)
EOF
}

# --- CLI parsing ----------------------------------------------------------
# Parse command line flags.

while getopts ":r:p:s:h" opt; do
  case "$opt" in
    r) dataset_root="$OPTARG" ;;
    p) per_image="$OPTARG" ;;
    s) seed="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 1 ;;
    :) echo "Option -$OPTARG requires a value" >&2; usage; exit 1 ;;
  esac
done

# --- Execution ------------------------------------------------------------
# Run the Python augmentation script with resolved parameters.

python3 "$root_dir/src/augment.py" \
  --root "$dataset_root" \
  --per-image "$per_image" \
  --seed "$seed"

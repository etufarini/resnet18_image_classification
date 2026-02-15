#!/usr/bin/env bash

# --- Shell options --------------------------------------------------------
# Safe shell options: fail fast and stop on undefined variables.

set -euo pipefail

# --- Notes ----------------------------------------------------------------
# Script context: install PyTorch ROCm wheels from AMD repository.

# Install PyTorch ROCm wheels for Ubuntu 24.04 + Python 3.12 (ROCm 7.2).
# Source: AMD ROCm documentation "Install PyTorch for ROCm".

# --- Configuration --------------------------------------------------------
# Resolve target python, pip command, and working directory.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" && -n "${VIRTUAL_ENV:-}" ]]; then
  PYTHON_BIN="$VIRTUAL_ENV/bin/python"
fi
if [[ -z "$PYTHON_BIN" && -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT/.venv/bin/python"
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "[rocm] creating venv in $ROOT/.venv"
  python3 -m venv "$ROOT/.venv"
  PYTHON_BIN="$ROOT/.venv/bin/python"
fi
PIP_CMD=("$PYTHON_BIN" -m pip)
WORKDIR="${WORKDIR:-/tmp/rocm_torch_wheels}"

# --- ROCm wheel packages --------------------------------------------------
# List of required ROCm wheel URLs.

WHEEL_URLS=(
  "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl"
  "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchvision-0.24.0%2Brocm7.2.0.gitb919bd0c-cp312-cp312-linux_x86_64.whl"
  "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/triton-3.5.1%2Brocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl"
)

# --- Validations ----------------------------------------------------------
# Check valid python/venv before installation.

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "[rocm] python not found. Provide PYTHON_BIN or activate a venv." >&2
  exit 1
fi

echo "[rocm] python: $("$PYTHON_BIN" -V)"
echo "[rocm] python bin: $PYTHON_BIN"

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1; then
import sys
raise SystemExit(0 if sys.prefix != sys.base_prefix else 1)
PY
  echo "[rocm] not a venv: $PYTHON_BIN" >&2
  exit 1
fi

# --- Pip configuration ----------------------------------------------------
# Update pip/wheel and remove previous torch installations.

"${PIP_CMD[@]}" install --upgrade pip wheel
"${PIP_CMD[@]}" uninstall -y torch torchvision || true

# --- Wheel download -------------------------------------------------------
# Download missing wheels into the working directory.

mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "[rocm] downloading wheels (skip if present)..."
for url in "${WHEEL_URLS[@]}"; do
  fname="$(basename "$url")"
  if [[ -f "$fname" ]]; then
    echo "[rocm] $fname already present, skipping download"
  else
    wget "$url"
  fi
done

# --- Installation ---------------------------------------------------------
# Install downloaded ROCm wheels locally.

echo "[rocm] installing wheels..."
"${PIP_CMD[@]}" install ./torch-*.whl ./torchvision-*.whl ./triton-*.whl

# --- Verification ---------------------------------------------------------
# Run a quick check on the installed torch build.

echo "[rocm] verification:"
"$PYTHON_BIN" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda is_available:", torch.cuda.is_available())
print("hip:", torch.version.hip)
print("cuda:", torch.version.cuda)
print("device_count:", torch.cuda.device_count())
if torch.cuda.device_count():
    print("device0:", torch.cuda.get_device_name(0))
PY

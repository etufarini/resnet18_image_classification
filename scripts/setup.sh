#!/usr/bin/env bash

# --- Shell options --------------------------------------------------------
# Safe shell options: fail fast and stop on undefined variables.

set -euo pipefail

# --- Configuration --------------------------------------------------------
# Base variables for repo path and virtual environment.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VENV_DIR=".venv"

# --- Venv creation --------------------------------------------------------
# Create the virtual environment if it does not exist.

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

# --- Dependency installation ----------------------------------------------
# Activate the venv and install project requirements.

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python3 -m pip install -U pip
python3 -m pip install -r "$ROOT_DIR/requirements.txt"

echo "Setup completato. Attiva il venv con: source $VENV_DIR/bin/activate"

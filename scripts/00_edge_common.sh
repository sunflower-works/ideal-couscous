#!/usr/bin/env bash
# 00_edge_common.sh
# Usage: bash 00_edge_common.sh ~/edge-yolo
# Creates a Python venv and installs core dependencies for YOLOv8 export + benchmarking.
set -euo pipefail
ROOT="${1:-$HOME/edge-yolo}"
PY="${PY:-python3}"
echo "[i] Root: $ROOT"
mkdir -p "$ROOT"
cd "$ROOT"
# System deps (best-effort). Skip if sudo unavailable.
if command -v apt-get >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1; then
    echo "[i] Installing system packages (python venv, build tools, cmake, ninja)"
    sudo apt-get update -y && \
    sudo apt-get install -y python3-venv python3-pip git pkg-config build-essential cmake ninja-build || \
      { echo "[e] apt-get failed"; exit 1; }
  else
    echo "[w] sudo not present; skipping apt-get install (ensure build-essential & cmake exist)"
  fi
fi
# Venv
$PY -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
# Install all required Python deps (fail fast if something missing)
pip install ultralytics onnx onnxsim numpy opencv-python tqdm rich || { echo "[e] Python dependency install failed"; exit 1; }

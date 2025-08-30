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
    sudo apt-get update -y && sudo apt-get install -y python3-venv python3-pip git pkg-config || echo "[w] apt-get partial failure (continuing)"
  else
    echo "[w] sudo not present; skipping apt-get install"
  fi
fi
# Venv
$PY -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install ultralytics onnx onnxsim numpy opencv-python tqdm rich
echo "[i] Common environment ready at $ROOT/.venv"


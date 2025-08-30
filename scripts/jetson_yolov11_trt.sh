#!/usr/bin/env bash
# Wrapper to run TensorRT export/build/benchmark for YOLOv11 on Jetson.
# Usage:
#   bash jetson_yolov11_trt.sh <root_dir> [imgsz]
# Env overrides: PRECISION, WARMUP, ITERS, WORKSPACE_MB, CALIB_DIR, CALIB_SAMPLES, CALIB_BATCH, LOG_DIR, MAP_EVAL, etc.
set -euo pipefail
ROOT="${1:-$HOME/edge-yolo}"
IMG_SIZE="${2:-${IMG_SIZE:-640}}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# Forward all env/args to the generic TRT script with MODEL pinned to yolov11n.pt
MODEL="yolov11n.pt" bash "$SCRIPT_DIR/10_jetson_yolov8_trt.sh" "$ROOT" "$IMG_SIZE"

#!/usr/bin/env bash
# 11_jetson_batch_precisions.sh
# Batch run YOLOv8 TensorRT builds & benchmarks for FP32, FP16, INT8 on Jetson.
# Uses existing 10_jetson_yolov8_trt.sh for each precision, aggregates CSV.
# Usage:
#   bash 11_jetson_batch_precisions.sh <root_dir> [imgsz]
# Env overrides:
#   IMG_SIZE (alias for positional) WARMUP ITERS WORKSPACE_MB LOG_DIR FORCE_REBUILD=1
#   CALIB_DIR=<folder with images> CALIB_SAMPLES=128 CALIB_BATCH=8 (required for INT8 calibrator)
#   SKIP_INT8=1 (skip INT8 phase)
#   EXTRA_PRECS="fp32 fp16 int8" (custom order/subset)
# Output:
#   detector_latency_jetson_<size>_<precision>.json per precision
#   latency_comparative.csv (aggregated) if all runs succeed
#   log files under logs/
set -euo pipefail
ROOT="${1:-$HOME/edge-yolo}"
IMG_SIZE="${2:-${IMG_SIZE:-640}}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
WARMUP="${WARMUP:-30}"
ITERS="${ITERS:-300}"
WORKSPACE_MB="${WORKSPACE_MB:-2048}"
CALIB_DIR="${CALIB_DIR:-}"
CALIB_SAMPLES="${CALIB_SAMPLES:-128}"
CALIB_BATCH="${CALIB_BATCH:-8}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
SKIP_INT8="${SKIP_INT8:-0}"
EXTRA_PRECS="${EXTRA_PRECS:-fp32 fp16 int8}"
MODEL="${MODEL:-yolov8n.pt}"
TS=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/batch_${IMG_SIZE}_${TS}.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "[i] Batch run start (imgsz=$IMG_SIZE warmup=$WARMUP iters=$ITERS)"

cd "$ROOT"
[ -f .venv/bin/activate ] && source .venv/bin/activate || true

RUN_PRECS=()
for p in $EXTRA_PRECS; do
  case $p in
    fp32|fp16) RUN_PRECS+=("$p") ;;
    int8) [ "$SKIP_INT8" = 1 ] && echo "[i] Skipping INT8 (SKIP_INT8=1)" || RUN_PRECS+=("int8") ;;
    *) echo "[w] Ignoring unknown precision token '$p'" ;;
  esac
done

if printf '%s\n' "${RUN_PRECS[@]}" | grep -qx int8; then
  if [ -z "$CALIB_DIR" ]; then
    echo "[w] INT8 requested but CALIB_DIR not set. Will build via trtexec without calibrator (less optimal, may warn)."
  elif [ ! -d "$CALIB_DIR" ]; then
    echo "[e] CALIB_DIR '$CALIB_DIR' not found" >&2; exit 2
  else
    echo "[i] INT8 calibration directory: $CALIB_DIR (samples=$CALIB_SAMPLES batch=$CALIB_BATCH)"
  fi
fi

PHASE_OK=1
for PREC in "${RUN_PRECS[@]}"; do
  echo "[i] === Phase: $PREC ==="
  STEM=$(basename "$MODEL"); STEM="${STEM%.*}"
  ENGINE="${STEM}_${IMG_SIZE}_${PREC}.engine"
  if [ -f "$ENGINE" ] && [ "$FORCE_REBUILD" != 1 ]; then
    echo "[i] Existing engine $ENGINE (reuse). Use FORCE_REBUILD=1 to rebuild."
  fi
  PRECISION=$PREC WARMUP=$WARMUP ITERS=$ITERS WORKSPACE_MB=$WORKSPACE_MB \
    CALIB_DIR="$CALIB_DIR" CALIB_SAMPLES=$CALIB_SAMPLES CALIB_BATCH=$CALIB_BATCH MODEL="$MODEL" \
    bash "$SCRIPT_DIR/10_jetson_yolov8_trt.sh" "$ROOT" "$IMG_SIZE" || { echo "[e] Phase $PREC failed"; PHASE_OK=0; break; }
  echo "[i] Phase $PREC done"
  echo
  sleep 2
 done

if [ $PHASE_OK -eq 1 ]; then
  echo "[i] All phases completed; aggregating"
  if [ -f aggregate_latency.py ]; then
    python aggregate_latency.py --glob "detector_latency_jetson_${IMG_SIZE}_*.json" --out "latency_comparative_${IMG_SIZE}.csv" || true
    if [ -f "latency_comparative_${IMG_SIZE}.csv" ]; then
      echo "[i] Wrote latency_comparative_${IMG_SIZE}.csv"
      echo "[i] Preview:"; head -n 5 "latency_comparative_${IMG_SIZE}.csv" || true
    fi
  else
    echo "[w] aggregate_latency.py missing; skipping aggregation"
  fi
else
  echo "[w] Skipping aggregation due to earlier failure"
fi

echo "[i] Batch run complete (log=$MASTER_LOG)"

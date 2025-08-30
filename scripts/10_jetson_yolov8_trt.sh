#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$HOME/edge-yolo}"
INPUT_SIZE="${2:-416}"
PRECISION="${PRECISION:-fp16}"
WARMUP="${WARMUP:-30}"
ITERS="${ITERS:-300}"
WORKSPACE_MB="${WORKSPACE_MB:-2048}"
CALIB_CACHE="${INT8_CALIB_CACHE:-}"
CALIB_DIR="${CALIB_DIR:-}"
CALIB_SAMPLES="${CALIB_SAMPLES:-128}"
CALIB_BATCH="${CALIB_BATCH:-8}"
LOG_DIR="${LOG_DIR:-logs}"
AGGREGATE="${AGGREGATE:-0}"
AUTO_CALIB_COCO_ROOT="${AUTO_CALIB_COCO_ROOT:-}"
AUTO_CALIB_SUBSETS="${AUTO_CALIB_SUBSETS:-val2017}"
AUTO_CALIB_DETERMINISTIC="${AUTO_CALIB_DETERMINISTIC:-1}"
MAP_EVAL="${MAP_EVAL:-0}"
MAP_DATA="${MAP_DATA:-coco128.yaml}"
MAP_BATCH="${MAP_BATCH:-16}"
INT8_LAYER_REPORT="${INT8_LAYER_REPORT:-0}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/run_${INPUT_SIZE}_${PRECISION}_${TS}.log"
BUILD_LOG="$LOG_DIR/build_${INPUT_SIZE}_${PRECISION}_${TS}.log"
# Log everything
exec > >(tee -a "$RUN_LOG") 2>&1

echo "[i] Starting build+benchmark (size=$INPUT_SIZE precision=$PRECISION)"

cd "$ROOT"
[ -f .venv/bin/activate ] && source .venv/bin/activate || true
PYTHON=${PYTHON:-python3}

case "$PRECISION" in
  fp32|fp16|int8) ;;
  fp8|fp4) echo "[e] Unsupported precision $PRECISION on this device." >&2; exit 2;;
  *) echo "[e] Unsupported PRECISION=$PRECISION (use fp32|fp16|int8)" >&2; exit 2;;
esac

if ! command -v trtexec >/dev/null 2>&1; then
  echo "[e] trtexec not in PATH. Install or add to PATH." >&2; exit 1
fi

TRT_HELP=$(trtexec --help 2>/dev/null || true)
if echo "$TRT_HELP" | grep -q -- '--memPoolSize'; then
  TRT_NEW_API=1
else
  TRT_NEW_API=0
fi

echo "[i] TensorRT API mode: $([ $TRT_NEW_API -eq 1 ] && echo 'new(>=10)' || echo 'legacy(<10)')"

# Ensure helper scripts present (force copy if missing)
HELPERS=(export_model.py bench_yolov8.py bench_trtexec.py build_int8_engine.py aggregate_latency.py make_calib_subset.py generate_latency_table.py evaluate_map.py list_int8_layers.py)
for f in "${HELPERS[@]}"; do
  if [ ! -f "$f" ]; then
    if [ -f "$SCRIPT_DIR/$f" ]; then
      cp "$SCRIPT_DIR/$f" .
      echo "[i] Copied $f from scripts dir"
    else
      echo "[w] Helper $f not found locally or in script dir"
    fi
  fi
done

# Auto-generate calibration subset if needed
if [ "$PRECISION" = int8 ] && [ -z "$CALIB_DIR" ] && [ -n "$AUTO_CALIB_COCO_ROOT" ]; then
  AUTO_CALIB_OUT="calib_auto_${INPUT_SIZE}"  # separate per size if different resolutions desired later
  if [ ! -d "$AUTO_CALIB_OUT" ]; then
    echo "[i] AUTO calib subset from $AUTO_CALIB_COCO_ROOT subsets=$AUTO_CALIB_SUBSETS count=$CALIB_SAMPLES -> $AUTO_CALIB_OUT"
    DET_FLAG=""
    [ "$AUTO_CALIB_DETERMINISTIC" = 1 ] && DET_FLAG="--deterministic"
    $PYTHON make_calib_subset.py --coco-root "$AUTO_CALIB_COCO_ROOT" --subsets $AUTO_CALIB_SUBSETS \
      --count "$CALIB_SAMPLES" --out-dir "$AUTO_CALIB_OUT" $DET_FLAG || { echo "[e] auto calib subset creation failed" >&2; exit 1; }
  else
    echo "[i] Reusing existing auto calib subset $AUTO_CALIB_OUT"
  fi
  CALIB_DIR="$AUTO_CALIB_OUT"
fi

$PYTHON export_model.py --model yolov8n.pt --imgsz "$INPUT_SIZE" --out "yolov8n_${INPUT_SIZE}.onnx" --dynamic false || { echo "[e] export failed" >&2; exit 1; }

ONNX_RAW="yolov8n_${INPUT_SIZE}.onnx"
ONNX_SIM="yolov8n_${INPUT_SIZE}_sim.onnx"
if $PYTHON -c 'import onnxsim' 2>/dev/null; then
  echo "[i] Simplifying ONNX"
  $PYTHON - <<PY "$ONNX_RAW" "$ONNX_SIM"
import sys, onnx
from onnxsim import simplify
src,dst=sys.argv[1:3]
sm,ok=simplify(onnx.load(src))
assert ok
onnx.save(sm,dst)
print('[i] Wrote', dst)
PY
else
  cp "$ONNX_RAW" "$ONNX_SIM"
fi
ONNX="$ONNX_SIM"
ENGINE="yolov8n_${INPUT_SIZE}_${PRECISION}.engine"

PY_TRT=0
if $PYTHON - <<'PY'
import sys
try: import tensorrt
except Exception: sys.exit(1)
else: sys.exit(0)
PY
then PY_TRT=1; fi

# Optional Python INT8 build (calibrator)
if [ "$PRECISION" = int8 ] && [ ! -f "$ENGINE" ] && [ -z "$CALIB_CACHE" ] && [ -n "$CALIB_DIR" ]; then
  [ -d "$CALIB_DIR" ] || { echo "[e] CALIB_DIR $CALIB_DIR not found" >&2; exit 1; }
  CALIB_CACHE="yolov8n_${INPUT_SIZE}_int8.calib"
  echo "[i] Python INT8 calibration (dir=$CALIB_DIR samples=$CALIB_SAMPLES batch=$CALIB_BATCH)"
  $PYTHON build_int8_engine.py --onnx "$ONNX" --engine "$ENGINE" --imgsz "$INPUT_SIZE" \
    --batch "$CALIB_BATCH" --calib_samples "$CALIB_SAMPLES" --calib_dir "$CALIB_DIR" --calib_cache "$CALIB_CACHE" || { echo "[e] Python INT8 build failed" >&2; exit 1; }
fi

if [ ! -f "$ENGINE" ]; then
  BUILD_ARGS=( --onnx="$ONNX" --saveEngine="$ENGINE" )
  case "$PRECISION" in
    fp32) : ;;
    fp16) BUILD_ARGS+=( --fp16 ) ;;
    int8)
      BUILD_ARGS+=( --int8 )
      [ -n "$CALIB_CACHE" ] && BUILD_ARGS+=( --calib="$CALIB_CACHE" )
      ;;
  esac
  if [ $TRT_NEW_API -eq 1 ]; then
    BUILD_ARGS+=( --memPoolSize=workspace:${WORKSPACE_MB}M --skipInference )
  else
    BUILD_ARGS+=( --workspace=$WORKSPACE_MB --buildOnly )
  fi
  echo "[i] trtexec ${BUILD_ARGS[*]}" | tee -a "$BUILD_LOG"
  if ! trtexec "${BUILD_ARGS[@]}" 2>&1 | tee -a "$BUILD_LOG"; then
    echo "[e] trtexec build failed" >&2; exit 1
  fi
else
  echo "[i] Reusing existing engine $ENGINE"
fi

# INT8 layer coverage report (after build)
if [ "$PRECISION" = int8 ] && [ "$INT8_LAYER_REPORT" = 1 ]; then
  if [ -f list_int8_layers.py ]; then
    echo "[i] Generating INT8 layer coverage report"
    $PYTHON list_int8_layers.py --engine "$ENGINE" --out "int8_layers_${INPUT_SIZE}.json" || echo "[w] INT8 layer report failed"
  else
    echo "[w] list_int8_layers.py missing; skip layer report"
  fi
fi

OUT_JSON="detector_latency_jetson_${INPUT_SIZE}_${PRECISION}.json"

USE_PYCUDA=0
if [ $PY_TRT -eq 1 ] && $PYTHON - <<'PY'
import sys
try: import pycuda.driver  # noqa
except Exception: sys.exit(1)
else: sys.exit(0)
PY
then USE_PYCUDA=1; fi

if [ $USE_PYCUDA -eq 1 ]; then
  echo "[i] Benchmark via pycuda"
  $PYTHON bench_yolov8.py --backend trt --engine "$ENGINE" --imgsz "$INPUT_SIZE" --warmup "$WARMUP" --iters "$ITERS" --out "$OUT_JSON"
else
  if [ -f bench_trtexec.py ]; then
    echo "[i] Benchmark via trtexec fallback"
    $PYTHON bench_trtexec.py --engine "$ENGINE" --imgsz "$INPUT_SIZE" --warmup "$WARMUP" --iters "$ITERS" --out "$OUT_JSON"
  else
    echo "[w] bench_trtexec.py missing; rough timing with trtexec built-in"
    trtexec --loadEngine="$ENGINE" --iterations="$ITERS" --warmUp="$((WARMUP*10))" || true
  fi
fi

echo "[i] Output JSON: $OUT_JSON"
if command -v jq >/dev/null 2>&1; then jq '.summary' "$OUT_JSON" 2>/dev/null || true; fi

# Optional mAP evaluation (ONNX)
if [ "$MAP_EVAL" = 1 ]; then
  if [ -f evaluate_map.py ]; then
    echo "[i] Evaluating mAP on $MAP_DATA"
    $PYTHON evaluate_map.py --onnx "$ONNX" --data "$MAP_DATA" --imgsz "$INPUT_SIZE" --batch "$MAP_BATCH" --out "map_metrics_${INPUT_SIZE}_${PRECISION}.json" || echo "[w] mAP eval failed"
  else
    echo "[w] evaluate_map.py missing; skip mAP"
  fi
fi

echo "[i] Logs stored: run=$RUN_LOG build=$BUILD_LOG"

# Optional aggregate CSV across runs
if [ "$AGGREGATE" = 1 ] && [ -f aggregate_latency.py ]; then
  echo "[i] Aggregating latency JSON files"
  $PYTHON aggregate_latency.py --glob 'detector_latency_jetson_*_*.json' --out latency_comparative.csv || true
  [ -f latency_comparative.csv ] && echo "[i] Wrote latency_comparative.csv"
fi

echo "[i] Done."

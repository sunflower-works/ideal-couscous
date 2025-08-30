#!/usr/bin/env bash
# 10_rpi_yolov8_ort.sh
# Usage: bash 10_rpi_yolov8_ort.sh <root_dir> [imgsz]
# Exports YOLOv8n to ONNX (static) and benchmarks with ONNX Runtime CPU.
set -euo pipefail
QUIET_MODE="${QUIET:-0}"
ROOT="${1:-$HOME/edge-yolo}"
INPUT_SIZE="${2:-320}"
OUT_SUFFIX="${OUT_SUFFIX:-}"
WARMUP="${WARMUP:-20}"
ITERS="${ITERS:-200}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
# Ensure exporter + bench utilities present
for f in export_model.py bench_yolov8.py; do
  cp -f "$SCRIPT_DIR/$f" . || { echo "[e] Unable to copy $f"; exit 1; }
done
# Dependencies (best-effort: onnxruntime may already be present due to ultralytics deps)
python - <<'PY'
import sys, subprocess
try:
  import onnxruntime  # type: ignore  # noqa: F401
  print('[i] onnxruntime present')
except Exception:
  print('[i] Installing missing: onnxruntime')
  subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'onnxruntime'])
PY
if [ "$QUIET_MODE" = "1" ]; then
  python export_model.py --model yolov8n.pt --imgsz "$INPUT_SIZE" --out "yolov8n_${INPUT_SIZE}.onnx" --dynamic false >/dev/null 2>&1 || { echo "[e] export failed"; exit 1; }
else
  python export_model.py --model yolov8n.pt --imgsz "$INPUT_SIZE" --out "yolov8n_${INPUT_SIZE}.onnx" --dynamic false || { echo "[e] export failed"; exit 1; }
fi
# Simplify
python - <<PY "yolov8n_${INPUT_SIZE}.onnx" "yolov8n_${INPUT_SIZE}_sim.onnx"
import sys, onnx
from onnxsim import simplify
src, dst = sys.argv[1], sys.argv[2]
print(f"[i] Simplifying {src}")
m = onnx.load(src)
sm, ok = simplify(m)
assert ok, 'simplify failed'
onnx.save(sm,dst)
print(f"[i] Wrote {dst}")
PY
if [ "$QUIET_MODE" = "1" ]; then
  python bench_yolov8.py \
    --backend ort \
    --onnx "yolov8n_${INPUT_SIZE}_sim.onnx" \
    --imgsz "$INPUT_SIZE" \
    --warmup "$WARMUP" --iters "$ITERS" \
    --source "$SCRIPT_DIR/../test_coco/val2017" \
    --out "detector_latency_rpi5_${INPUT_SIZE}_ort${OUT_SUFFIX}.json" >/dev/null 2>&1 || { echo "[e] bench failed"; exit 1; }
else
  python bench_yolov8.py \
  --backend ort \
  --onnx "yolov8n_${INPUT_SIZE}_sim.onnx" \
  --imgsz "$INPUT_SIZE" \
  --warmup "$WARMUP" --iters "$ITERS" \
  --source "$SCRIPT_DIR/../test_coco/val2017" \
  --out "detector_latency_rpi5_${INPUT_SIZE}_ort${OUT_SUFFIX}.json"
fi


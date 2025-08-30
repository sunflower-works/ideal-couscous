#!/usr/bin/env bash
# 10_rpi_yolov8_ov.sh
# Usage: bash 10_rpi_yolov8_ov.sh <root_dir> [imgsz]
# Exports YOLOv8n to ONNX (static) and benchmarks with OpenVINO CPU.
set -euo pipefail
ROOT="${1:-$HOME/edge-yolo}"
INPUT_SIZE="${2:-320}"
WARMUP="${WARMUP:-20}"
ITERS="${ITERS:-200}"
QUIET_MODE="${QUIET:-0}"
OUT_SUFFIX="${OUT_SUFFIX:-}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

# Ensure exporter + bench utilities present (refresh)
for f in export_model.py bench_yolov8.py; do
  cp -f "$SCRIPT_DIR/$f" . || { echo "[e] Unable to copy $f"; exit 1; }
done

# Dependencies (pin OpenVINO to a known-good version)
python - <<'PY'
import sys, subprocess

TARGETS = [
  ('openvino==2023.3.0', '2023.3'),
  ('openvino==2024.1.0', '2024.1'),
]

def have_version(prefix: str) -> bool:
  try:
    import importlib.metadata as md
    v = md.version('openvino')
    print(f"[i] openvino version detected: {v}")
    return v.startswith(prefix)
  except Exception:
    return False

if any(have_version(pfx) for _, pfx in TARGETS):
  print("[i] openvino version OK")
else:
  # Try first target, then fallback
  for spec, pfx in TARGETS:
    try:
      print(f"[i] Installing {spec}")
      subprocess.check_call([sys.executable, '-m', 'pip', 'install', spec])
      if have_version(pfx):
        break
    except Exception as e:
      print(f"[w] install failed for {spec}: {e}")
PY

# Export to ONNX
python export_model.py --model yolov8n.pt --imgsz "$INPUT_SIZE" --out "yolov8n_${INPUT_SIZE}.onnx" --dynamic false || { echo "[e] export failed"; exit 1; }

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

# Benchmark with OpenVINO
if [ "$QUIET_MODE" = "1" ]; then
  python bench_yolov8.py \
    --backend ov \
    --onnx "yolov8n_${INPUT_SIZE}_sim.onnx" \
    --imgsz "$INPUT_SIZE" \
    --warmup "$WARMUP" --iters "$ITERS" \
    --source "$SCRIPT_DIR/../test_coco/val2017" \
    --out "detector_latency_rpi5_${INPUT_SIZE}_ov${OUT_SUFFIX}.json" >/dev/null 2>&1 || { echo "[e] bench failed"; exit 1; }
else
  python bench_yolov8.py \
    --backend ov \
    --onnx "yolov8n_${INPUT_SIZE}_sim.onnx" \
    --imgsz "$INPUT_SIZE" \
    --warmup "$WARMUP" --iters "$ITERS" \
    --source "$SCRIPT_DIR/../test_coco/val2017" \
    --out "detector_latency_rpi5_${INPUT_SIZE}_ov${OUT_SUFFIX}.json"
fi

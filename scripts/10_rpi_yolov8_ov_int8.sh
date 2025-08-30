#!/usr/bin/env bash
# 10_rpi_yolov8_ov_int8.sh
# Usage: bash 10_rpi_yolov8_ov_int8.sh <root_dir> [imgsz]
# Quantizes YOLOv8n ONNX to INT8 (OpenVINO POT) and benchmarks with OpenVINO CPU.
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

# Ensure OpenVINO + POT
python - <<'PY'
import sys, subprocess
def ensure(pkg):
    try:
        __import__(pkg)
        print(f"[i] {pkg} present")
    except Exception:
        print(f"[i] Installing missing: {pkg}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])
ensure('openvino')
try:
    import openvino.tools.pot  # noqa: F401
    print('[i] openvino.tools.pot present')
except Exception:
    print('[i] Installing openvino-dev for POT')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'openvino-dev'])
PY

# Export to ONNX and simplify
python export_model.py --model yolov8n.pt --imgsz "$INPUT_SIZE" --out "yolov8n_${INPUT_SIZE}.onnx" --dynamic false ${QUIET_MODE:+ >/dev/null 2>&1}
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

# Quantize to INT8 with POT using calib_images if present
python - <<PY "yolov8n_${INPUT_SIZE}_sim.onnx" "yolov8n_${INPUT_SIZE}_int8.xml"
import sys, os, glob
import numpy as np
import cv2 as cv
import openvino as ov
from openvino.tools.pot import IEEngine, load_model, save_model, create_pipeline, compress_model_weights

onnx_path, out_xml = sys.argv[1], sys.argv[2]
core = ov.Core()
model = core.read_model(onnx_path)
# Serialize FP32 IR as POT expects IR
tmp_xml = f"{os.path.splitext(out_xml)[0]}_fp32.xml"
tmp_bin = f"{os.path.splitext(out_xml)[0]}_fp32.bin"
ov.serialize(model, tmp_xml, tmp_bin)

# Data loader over calib_images
calib_dir = os.path.join(os.getcwd(), 'calib_images')
paths = []
for ext in ('*.jpg','*.jpeg','*.png'):
    paths.extend(glob.glob(os.path.join(calib_dir, ext)))
paths = sorted(paths)[:512] if os.path.isdir(calib_dir) else []
if not paths:
    print('[w] No calib_images found; POT will quantize with minimal sampling')

inp_name = model.inputs[0].get_any_name()
_, c, h, w = (1,3,int(model.inputs[0].get_partial_shape()[2].get_length()), int(model.inputs[0].get_partial_shape()[3].get_length())) if len(model.inputs[0].get_partial_shape())==4 else (1,3,int(os.path.basename(onnx_path).split('_')[-2]), int(os.path.basename(onnx_path).split('_')[-2]))

def letterbox(im, new_size):
    ih, iw = im.shape[:2]
    r = min(new_size/ih, new_size/iw)
    nh, nw = int(round(ih*r)), int(round(iw*r))
    top = (new_size - nh)//2
    left = (new_size - nw)//2
    out = np.full((new_size,new_size,3), 114, dtype=np.uint8)
    resized = cv.resize(im,(nw,nh), interpolation=cv.INTER_LINEAR)
    out[top:top+nh, left:left+nw] = resized
    return out

class DL:
    def __init__(self, files):
        self.files = files
    def __len__(self):
        return len(self.files)
    def __iter__(self):
        for p in self.files:
            im = cv.imread(p)
            if im is None:
                continue
            x = letterbox(im, h).transpose(2,0,1)[None].astype(np.float32)/255.0
            yield {inp_name: x}

engine = IEEngine(config={'device': 'CPU'}, data_loader=DL(paths), metric=None)
algos = [{'name': 'DefaultQuantization', 'params': {'preset': 'performance', 'stat_subset_size': min(300, len(paths)) if paths else 1}}]
pipeline = create_pipeline(algos, engine)
model_pot = load_model(tmp_xml)
compress_model_weights(model_pot)
model_pot = pipeline.run(model_pot)
save_model(model_pot, save_path=os.path.dirname(out_xml), model_name=os.path.splitext(os.path.basename(out_xml))[0])
print(f"[i] Quantized model saved: {out_xml}")
PY

# Benchmark INT8 IR with OpenVINO
if [ "$QUIET_MODE" = "1" ]; then
  python bench_yolov8.py \
    --backend ov \
    --onnx "yolov8n_${INPUT_SIZE}_int8.xml" \
    --imgsz "$INPUT_SIZE" \
    --warmup "$WARMUP" --iters "$ITERS" \
    --source "$SCRIPT_DIR/../test_coco/val2017" \
    --out "detector_latency_rpi5_${INPUT_SIZE}_ov_int8${OUT_SUFFIX}.json" >/dev/null 2>&1 || { echo "[e] bench failed"; exit 1; }
else
  python bench_yolov8.py \
    --backend ov \
    --onnx "yolov8n_${INPUT_SIZE}_int8.xml" \
    --imgsz "$INPUT_SIZE" \
    --warmup "$WARMUP" --iters "$ITERS" \
    --source "$SCRIPT_DIR/../test_coco/val2017" \
    --out "detector_latency_rpi5_${INPUT_SIZE}_ov_int8${OUT_SUFFIX}.json"
fi

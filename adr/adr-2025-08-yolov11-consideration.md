# ADR: Consider adopting YOLOv11 for export and TRT builds (2025-08)

## Status
Proposed

## Context
- Current pipeline targets YOLOv8 (Ultralytics) with ONNX export and TensorRT.
- We encountered ecosystem churn: torchvision/SymPy interactions during embedded NMS, TRT10 API changes, and ABI pinning for NumPy.
- Ultralytics has continued to update model families (e.g., YOLOv11) with potential improvements in export stability and ops coverage.

## Problem
- Maintaining custom export flags (embedded NMS/TopK) across versions is brittle.
- TRT kernel/tactic coverage and plugins (e.g., EfficientNMS) evolve; newer model heads may export more cleanly.

## Proposal
- Evaluate YOLOv11n baseline:
  1) Swap model artifact to `yolov11n.pt` (or corresponding alias) in `scripts/export_model.py`.
  2) Export ONNX at target sizes (320/480) with `--dynamic false` and without embedded NMS initially.
  3) Build TRT engines using the existing `10_jetson_yolov8_trt.sh` script (no code changes required thanks to TRT10 path in `bench_yolov8.py`).
  4) Benchmark fp32/fp16/int8 and compare `latency_comparative_*.csv` to YOLOv8n.
- If export quality is good, optionally re-enable embedded NMS in ONNX if `torchvision.ops` is available, or rely on TRT EfficientNMS during engine build.

## Risks
- YOLOv11 export may pull in different op sets; ensure onnx/onnxruntime support.
- Accuracy/latency trade-offs differ; validate mAP on coco128 or a small subset.

## Acceptance criteria
- End-to-end batch succeeds at sizes 320 and 480 (fp32/fp16/int8) with TRT10 pycuda path.
- Aggregated CSV and report generation remain unchanged.
- Optional: parity or improvement on latency vs YOLOv8n at similar accuracy.

## Next steps
- Add a make variable `MODEL?=yolov8n.pt` in `Makefile.jetson` and plumb it into `export_model.py` (small patch) to toggle between v8/v11 easily.
- Prepare a small comparison section in the report to present v8 vs v11 latency.

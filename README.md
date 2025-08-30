# Edge YOLO Benchmarking on Jetson (TRT10 + PyCUDA)

This repo automates exporting YOLO models, building TensorRT engines, and benchmarking fp32/fp16/int8 on Jetson. It also generates a small report and comparison tables.

## Quick start (Jetson)

1) System TensorRT Python bindings (from NVIDIA repos):
- sudo apt-get install -y python3-libnvinfer python3-libnvinfer-dev

2) Python env with system packages and pinned deps:
- make -C scripts -f Makefile.jetson env
  - Installs ultralytics, onnx/onnxsim/onnxruntime, numpy<2, opencv-python, tqdm, rich, cuda-python, and tries pycuda.
  - If pycuda isn’t available via apt, install it into the venv: /home/<you>/edge-yolo/.venv/bin/pip install pycuda

3) Verify environment (no heavy imports):
- /home/<you>/edge-yolo/.venv/bin/python scripts/env_report.py
  - Look for: tensorrt_available True, pycuda_available True, numpy 1.x; onnx/onnxruntime present.

4) Dataset and calibration subset:
- make -C scripts -f Makefile.jetson coco-download
- make -C scripts -f Makefile.jetson calib-subset

5) Run batch and generate report:
- make -C scripts -f Makefile.jetson batch IMG_SIZE=480
- make -C scripts -f Makefile.jetson report IMG_SIZE=480

Artifacts are under ~/edge-yolo:
- Engines: yolov8n_<size>_<precision>.engine
- Latency: detector_latency_jetson_<size>_<prec>.json, latency_comparative_<size>.csv
- Report: results/figures/*.tex|json

## Why you might still see trtexec fallback
The benchmark uses PyCUDA if both TensorRT and PyCUDA are importable in the venv. If not, it falls back to running trtexec. Ensure:
- Venv created with --system-site-packages (env target already does this)
- python3-libnvinfer is installed (TensorRT Python)
- pycuda is installed in the venv (pip) if apt python3-pycuda isn’t available

## Important pins
- numpy<2 (to avoid cv2 and other binary wheels built for NumPy 1.x from crashing)
- ONNX/ONNX Runtime installed in the venv (export and baseline CPU)

## Troubleshooting quick reference
- Export crashes referencing torchvision/SymPy: embedded NMS is auto-disabled if torchvision.ops.nms is missing.
- TRT error: 'ICudaEngine' has no attribute get_binding_index: fixed by using TRT10 named tensor API in scripts/bench_yolov8.py.
- ValueError: ndarray is not contiguous: fixed by ensuring contiguous float32 input before CUDA memcpy in bench_yolov8.
- cv2 import errors / _ARRAY_API not found / multiarray failed to import: ensure numpy<2 and reinstall opencv-python.
- Still using trtexec fallback: install TensorRT Python (python3-libnvinfer) and PyCUDA into the venv, then re-run scripts/env_report.py.

## Docs
- See adr/adr-2025-08-jetson-trt10-pycuda-env.md for background, decisions, and step-by-step reproduction.

## License
Internal benchmarking scripts and docs.

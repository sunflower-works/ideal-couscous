# ADR: Jetson TRT10 + PyCUDA benchmarking, environment hardening, and export fixes (2025-08)

## Status
Accepted

## Context
We benchmark YOLOv8 on a Jetson (Orin Nano) across fp32/fp16/int8 with TensorRT. The pipeline originally:
- Fell back to `trtexec` because Python TensorRT and PyCUDA were not importable in the venv.
- Broke exports when embedding NMS (torchvision/SymPy import path issues).
- Hit TensorRT 10 API differences (legacy bindings vs new named tensor API).
- Suffered NumPy 2.x ABI incompatibilities with cv2 and other compiled deps.
- Encountered non-contiguous NumPy arrays causing CUDA memcpy failures.

## Decision
- Use system TensorRT Python bindings on Jetson with a venv created via `--system-site-packages`.
- Install PyCUDA via pip (aarch64 wheel/build) when `python3-pycuda` apt is unavailable.
- Pin `numpy<2` in the venv to avoid cv2 ABI crashes.
- Export ONNX without embedded NMS when `torchvision.ops` is not importable; rely on TRT-side NMS.
- Update TRT runner to support TensorRT 10’s execute_async_v3 and named tensor API while keeping legacy path.
- Ensure input tensors are contiguous float32 before CUDA memcpy to avoid `ValueError: ndarray is not contiguous`.

## Changes
- `scripts/Makefile.jetson`: env target installs onnxruntime and pins `numpy<2`; retains `--system-site-packages`.
- `scripts/export_model.py`: auto-disables NMS embedding if `torchvision.ops.nms` is missing.
- `scripts/bench_yolov8.py`: TRT10 support via `set_input_shape`, `set_tensor_address`, `execute_async_v3`; legacy path unchanged. Added contiguous-buffer guard.
- `scripts/env_report.py`: uses importlib.metadata to read versions without importing heavy modules; writes `results/figures/env_report.txt`.

## Alternatives considered
- Conda environments on Jetson: rejected. TRT Python bindings for aarch64 are provided via apt; conda mixes poorly with NVIDIA system CUDA/TRT.
- Forcing torchvision install to keep NMS in ONNX: deferred; not required for benchmarking as TRT provides EfficientNMS.

## Reproducible setup (Jetson)
1) System packages (TensorRT Python):
   - `sudo apt-get install -y python3-libnvinfer python3-libnvinfer-dev`
   - If `python3-pycuda` is not available in your repo, install PyCUDA in the venv via pip (next step).
2) Create venv with system packages and Python deps:
   - `make -C scripts -f Makefile.jetson env`
   - This installs: ultralytics, onnx, onnxsim, onnxruntime, numpy<2, opencv-python, tqdm, rich, and attempts cuda-python/pycuda.
   - If PyCUDA is still missing: `/home/<you>/edge-yolo/.venv/bin/pip install pycuda` (builds wheel on aarch64).
3) Verify environment (no heavy imports):
   - `/home/<you>/edge-yolo/.venv/bin/python scripts/env_report.py`
   - Confirm: `tensorrt_available True`, `pycuda_available True`, numpy 1.x, onnx/onnxruntime present.
4) Dataset + calibration subset:
   - `make -C scripts -f Makefile.jetson coco-download`
   - `make -C scripts -f Makefile.jetson calib-subset`
5) Batch run and report:
   - `make -C scripts -f Makefile.jetson batch IMG_SIZE=480`
   - `make -C scripts -f Makefile.jetson report IMG_SIZE=480`

## Verification
- `bench_yolov8.py` prints “Benchmark via pycuda” and completes without API/contiguity errors.
- `latency_comparative_480.csv` contains fp32/fp16/int8 rows; optional `ovcpu` row if OpenVINO path used.
- `results/figures/env_report.txt` records versions and flags.

## Rollback plan
- If TRT10 API path fails on specific engines, fallback to `bench_trtexec.py` remains available.
- If cv2 import errors reappear, re-pin `numpy<2` in the venv and reinstall `opencv-python`.
- If NMS embedding is desired later, install a torchvision build compatible with the current torch and SymPy.

## Lessons learned
- On Jetson, prefer venv with `--system-site-packages` to access NVIDIA’s TRT Python bindings.
- Pin `numpy<2` unless all binary wheels are NumPy 2-ready.
- Support both TRT APIs in code; don’t assume legacy bindings on TRT10+.
- Guard against non-contiguous NumPy arrays before CUDA transfers.

#!/usr/bin/env python3
"""Build an INT8 TensorRT engine with simple entropy calibrator.
Requires: TensorRT Python bindings, cuda-python (preferred) or pycuda.
If calibration cache already exists, it will be reused to avoid recalibration.

Example:
  python build_int8_engine.py \
    --onnx yolov8n_640_sim.onnx --engine yolov8n_640_int8.engine \
    --calib_dir calib_images --imgsz 640 --batch 8 --calib_samples 128 \
    --calib_cache yolov8n_640_int8.calib
"""
import argparse
import glob
import os
import sys

import cv2 as cv
import numpy as np

try:
    import tensorrt as trt
except Exception as e:  # pragma: no cover
    print(f"[e] Cannot import tensorrt: {e}", file=sys.stderr)
    sys.exit(1)

# Prefer cuda-python for allocations; fallback to pycuda
try:
    from cuda import cudart  # type: ignore

    USE_CUDART = True
except Exception:  # pragma: no cover
    USE_CUDART = False
    try:
        import pycuda.driver as cuda  # type: ignore
        import pycuda.autoinit  # type: ignore
    except Exception as e:  # pragma: no cover
        print("[e] Neither cuda-python nor pycuda available.", file=sys.stderr)
        sys.exit(1)


def letterbox(img, size, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    top = (size - nh) // 2
    left = (size - nw) // 2
    out = np.full((size, size, 3), color, dtype=np.uint8)
    resized = cv.resize(img, (nw, nh), interpolation=cv.INTER_LINEAR)
    out[top : top + nh, left : left + nw] = resized
    return out


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):  # pragma: no cover (runtime)
    def __init__(self, images, batch_size, input_shape, cache_file):
        super().__init__()
        self.images = images
        self.batch_size = batch_size
        self.input_shape = input_shape  # (C,H,W)
        self.cache_file = cache_file
        self.index = 0
        nbytes = np.prod((batch_size,) + input_shape) * 4
        if USE_CUDART:
            err, self.d_ptr = cudart.cudaMalloc(int(nbytes))  # type: ignore
            if err != 0:
                raise RuntimeError("cudaMalloc failed")
        else:
            self.d_mem = cuda.mem_alloc(int(nbytes))  # type: ignore

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):  # names: list[str]
        if self.index >= len(self.images):
            return None
        batch = []
        for _ in range(self.batch_size):
            if self.index >= len(self.images):
                break
            path = self.images[self.index]
            im = cv.imread(path)
            if im is None:
                self.index += 1
                continue
            im = letterbox(im, self.input_shape[1])
            x = im.transpose(2, 0, 1).astype(np.float32) / 255.0
            batch.append(x)
            self.index += 1
        if not batch:
            return None
        arr = np.stack(batch, axis=0)
        # Pad if last batch smaller
        if arr.shape[0] < self.batch_size:
            pad = np.zeros(
                (self.batch_size - arr.shape[0],) + self.input_shape, dtype=np.float32
            )
            arr = np.concatenate([arr, pad], axis=0)
        if USE_CUDART:
            err = cudart.cudaMemcpy(self.d_ptr, arr.ctypes.data, arr.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)  # type: ignore
            if err != 0:
                raise RuntimeError("cudaMemcpy H2D failed")
            return [int(self.d_ptr)]
        else:
            cuda.memcpy_htod(self.d_mem, arr)  # type: ignore
            return [int(self.d_mem)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"[i] Using existing calibration cache {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"[i] Wrote calibration cache {self.cache_file}")


def build_int8(
    onnx_path, engine_path, imgsz, batch, calib_samples, calib_dir, cache_file
):
    logger = trt.Logger(trt.Logger.WARNING)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    builder = trt.Builder(logger)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i), file=sys.stderr)
            raise SystemExit("[e] Failed to parse ONNX")
    input_tensor = network.get_input(0)
    c, h, w = input_tensor.shape[1:]
    assert h == imgsz and w == imgsz, "Input size mismatch"
    # Collect calibration images
    exts = ("*.jpg", "*.jpeg", "*.png")
    imgs = []
    for e in exts:
        imgs.extend(glob.glob(os.path.join(calib_dir, e)))
    imgs = sorted(imgs)[:calib_samples]
    if not imgs:
        raise SystemExit(f"[e] No calibration images found in {calib_dir}")
    print(f"[i] Calibration images: {len(imgs)} (batch={batch})")
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.max_workspace_size = 4 << 30  # 4GB cap
    calibrator = EntropyCalibrator(imgs, batch, (c, h, w), cache_file)
    config.int8_calibrator = calibrator
    print("[i] Building INT8 engine...")
    engine = builder.build_engine(network, config)
    if engine is None:
        raise SystemExit("[e] Engine build failed")
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())
    print(f"[i] Wrote INT8 engine {engine_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--calib_samples", type=int, default=128)
    ap.add_argument("--calib_dir", required=True)
    ap.add_argument("--calib_cache", required=True)
    args = ap.parse_args()
    if os.path.exists(args.engine):
        print(f"[i] Engine already exists: {args.engine}")
        return
    build_int8(
        args.onnx,
        args.engine,
        args.imgsz,
        args.batch,
        args.calib_samples,
        args.calib_dir,
        args.calib_cache,
    )


if __name__ == "__main__":  # pragma: no cover
    main()

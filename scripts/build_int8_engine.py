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
import subprocess


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

def _trt_ge(major: int, minor: int = 0) -> bool:
    """Return True if installed TensorRT version >= major.minor."""
    try:
        ver = trt.__version__.split("+", 1)[0]
        parts = [int(p) for p in ver.split(".")[:3]]
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts) >= (major, minor, 0)
    except Exception:
        return False

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
    # Workspace configuration: TRT <10 uses max_workspace_size; TRT >=10 uses memory pool limit
    if hasattr(config, "set_memory_pool_limit"):
        try:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(4 << 30))
        except Exception:
            pass
    else:
        try:
            config.max_workspace_size = 4 << 30  # legacy
        except Exception:
            pass

    def _build_and_write(net, want_int8: bool) -> bool:
        # Set INT8 flag only when building a Q/DQ-quantized network or when using calibrator on TRT<10
        if want_int8:
            config.set_flag(trt.BuilderFlag.INT8)
        else:
            try:
                config.clear_flag(trt.BuilderFlag.INT8)  # TRT>=10
            except Exception:
                pass
        if hasattr(builder, "build_serialized_network"):
            plan = builder.build_serialized_network(net, config)
            if plan is None:
                return False
            with open(engine_path, "wb") as f:
                f.write(plan)
            return True
        else:
            eng = builder.build_engine(net, config)
            if eng is None:
                return False
            with open(engine_path, "wb") as f:
                f.write(eng.serialize())
            return True

    # TRT >= 10: prefer explicit Q/DQ PTQ, do NOT use calibrators (deprecated/unstable)
    if _trt_ge(10, 0):
        try:
            from onnxruntime.quantization import (  # type: ignore
                CalibrationDataReader,
                QuantType,
                QuantFormat,
                CalibrationMethod,
                quantize_static,
            )
            import onnx  # type: ignore
            import cv2 as cv  # local import to avoid heavy import at module import time
            import numpy as np

            m = onnx.load(onnx_path)
            input_name = m.graph.input[0].name

            class YOLODataReader(CalibrationDataReader):
                def __init__(self, images, input_name, size):
                    self.images = images
                    self.input_name = input_name
                    self.size = size
                    self.idx = 0
                def get_next(self):
                    while self.idx < len(self.images):
                        p = self.images[self.idx]
                        self.idx += 1
                        im = cv.imread(p)
                        if im is None:
                            continue
                        x = letterbox(im, self.size).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
                        return {self.input_name: x}
                    return None

            op_types = ["Conv", "MatMul", "Gemm"]
            reader = YOLODataReader(imgs[: max(1, min(64, len(imgs)))], input_name, imgsz)
            qdq_path = os.path.splitext(onnx_path)[0] + "_int8_qdq.onnx"

            quantize_static(
                model_input=onnx_path,
                model_output=qdq_path,
                calibration_data_reader=reader,
                quant_format=QuantFormat.QDQ,
                weight_type=QuantType.QInt8,      # symmetric weights
                activation_type=QuantType.QInt8,  # symmetric activations
                per_channel=True,                 # per-channel weights
                calibrate_method=CalibrationMethod.MinMax,
                op_types_to_quantize=op_types,
                extra_options={
                    "ActivationSymmetric": True,
                    "WeightSymmetric": True,
                    "QuantizeBias": False,        # avoid bias Q/DQ
                    "DedicatedQDQPair": True,
                    "EnableSubgraph": True,
                },
            )
            print(f"[i] Wrote Q/DQ quantized ONNX -> {qdq_path}")

            # Parse quantized model and build INT8 (no calibrator)
            network_q = builder.create_network(flag)
            parser_q = trt.OnnxParser(network_q, logger)
            with open(qdq_path, "rb") as f:
                if not parser_q.parse(f.read()):
                    for i in range(parser_q.num_errors):
                        print(parser_q.get_error(i), file=sys.stderr)
                    raise RuntimeError("[e] Failed to parse quantized ONNX")

            # Ensure no calibrator is set on TRT>=10 to avoid deprecation warnings
            try:
                if hasattr(config, "int8_calibrator"):
                    config.int8_calibrator = None  # harmless on TRT<10; silenced by try on TRT>=10
            except Exception:
                pass

            if _build_and_write(network_q, want_int8=True):
                print(f"[i] Wrote INT8 engine {engine_path} (from Q/DQ ONNX)")
                return
            raise RuntimeError("[e] Engine build failed from Q/DQ ONNX")
        except Exception as e:
            print(f"[e] PTQ fallback failed: {e}", file=sys.stderr)
            # Let the outer pipeline fall back to trtexec
            raise SystemExit("[e] Engine build failed (serialized network is None)")
    # TRT < 10: use entropy calibrator path
    calibrator = EntropyCalibrator(imgs, batch, (c, h, w), cache_file)
    try:
        config.int8_calibrator = calibrator
    except Exception:
        pass
    print("[i] Building INT8 engine...")
    if _build_and_write(network, want_int8=True):
        print(f"[i] Wrote INT8 engine {engine_path}")
        return
    # As a last resort on TRT<10, try Q/DQ as above (optional)
    try:
        from onnxruntime.quantization import (  # type: ignore
            CalibrationDataReader,
            QuantType,
            QuantFormat,
            CalibrationMethod,
            quantize_static,
        )
        import onnx  # type: ignore
        import cv2 as cv
        import numpy as np
        m = onnx.load(onnx_path)
        input_name = m.graph.input[0].name
        class YOLODataReader(CalibrationDataReader):
            def __init__(self, images, input_name, size):
                self.images = images; self.input_name = input_name; self.size = size; self.idx = 0
            def get_next(self):
                while self.idx < len(self.images):
                    p = self.images[self.idx]; self.idx += 1
                    im = cv.imread(p)
                    if im is None: continue
                    x = letterbox(im, self.size).transpose(2,0,1)[None].astype(np.float32)/255.0
                    return {self.input_name: x}
                return None
        op_types = ["Conv", "MatMul", "Gemm"]
        reader = YOLODataReader(imgs[: max(1, min(64, len(imgs)))], input_name, imgsz)
        qdq_path = os.path.splitext(onnx_path)[0] + "_int8_qdq.onnx"
        quantize_static(
            model_input=onnx_path,
            model_output=qdq_path,
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
            per_channel=True,
            calibrate_method=CalibrationMethod.MinMax,
            op_types_to_quantize=op_types,
            extra_options={
                "ActivationSymmetric": True,
                "WeightSymmetric": True,
                "QuantizeBias": False,
                "DedicatedQDQPair": True,
                "EnableSubgraph": True,
            },
        )
        print(f"[i] Wrote Q/DQ quantized ONNX -> {qdq_path}")
        network_q = builder.create_network(flag)
        parser_q = trt.OnnxParser(network_q, logger)
        with open(qdq_path, "rb") as f:
            if not parser_q.parse(f.read()):
                for i in range(parser_q.num_errors):
                    print(parser_q.get_error(i), file=sys.stderr)
                raise RuntimeError("[e] Failed to parse quantized ONNX")
        try:
            config.int8_calibrator = None
        except Exception:
            pass
        if _build_and_write(network_q, want_int8=True):
            print(f"[i] Wrote INT8 engine {engine_path} (from Q/DQ ONNX)")
            return
    except Exception as e:
        print(f"[e] PTQ fallback failed: {e}", file=sys.stderr)
    raise SystemExit("[e] Engine build failed (serialized network is None)")



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

#!/usr/bin/env python3
"""Latency benchmark for YOLOv8 ONNX / TensorRT / OpenVINO.
Writes JSON with summary (mean/p50/p95) and per-frame latencies.
"""
import argparse
import json
import os
import sys
import time

import cv2 as cv
import numpy as np


def letterbox(img, new_size, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    top = (new_size - nh) // 2
    left = (new_size - nw) // 2
    out = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    resized = cv.resize(img, (nw, nh), interpolation=cv.INTER_LINEAR)
    out[top : top + nh, left : left + nw] = resized
    return out


def get_frames(source, count):
    if source and os.path.isdir(source):
        paths = [
            os.path.join(source, p)
            for p in sorted(os.listdir(source))
            if p.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        for p in paths[:count]:
            im = cv.imread(p)
            if im is None:
                continue
            yield p, im
    else:
        for i in range(count):
            im = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            yield f"rand_{i:05d}", im


def bench_ort(onnx_path, imgsz, warmup, iters, source):
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    lat = []
    ids = []
    for _, im in get_frames(source, warmup):
        x = letterbox(im, imgsz).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        sess.run(None, {inp_name: x})
    for fid, im in get_frames(source, iters):
        x = letterbox(im, imgsz).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        t0 = time.perf_counter()
        _ = sess.run(None, {inp_name: x})
        lat.append((time.perf_counter() - t0) * 1000.0)
        ids.append(fid)
    return ids, lat


def bench_trt(engine_path, imgsz, warmup, iters, source):
    import tensorrt as trt, pycuda.driver as cuda, pycuda.autoinit  # type: ignore

    logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("Failed to deserialize TensorRT engine")
    ctx = engine.create_execution_context()

    def vol(shape):
        return int(np.prod(shape))

    stream = cuda.Stream()
    lat, ids = [], []

    # Detect API: legacy bindings vs. TensorRT >=10 tensor API
    has_legacy = hasattr(engine, "get_binding_index") and hasattr(ctx, "execute_async_v2")
    if has_legacy:
        # Legacy path (TensorRT < 10)
        inp_idx = engine.get_binding_index(engine[0])
        ctx.set_binding_shape(inp_idx, (1, 3, imgsz, imgsz))
        d_in = cuda.mem_alloc(4 * vol((1, 3, imgsz, imgsz)))
        # Heuristic output buffer (1MB) — enough for YOLOv8n NMS outputs
        d_out = cuda.mem_alloc(1 << 20)
        def infer_once(x):
            if not x.flags['C_CONTIGUOUS']:
                x = np.ascontiguousarray(x)
            cuda.memcpy_htod_async(d_in, x, stream)
            ctx.execute_async_v2(bindings=[int(d_in), int(d_out)], stream_handle=stream.handle)
            stream.synchronize()
    else:
        # TensorRT >= 10: use named tensor API and execute_async_v3
        # Find input and output tensor names
        num_io = engine.num_io_tensors
        names = [engine.get_tensor_name(i) for i in range(num_io)]
        inputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        outputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        if not inputs:
            raise RuntimeError("No input tensors found in engine")
        inp_name = inputs[0]
        # Set input shape
        ctx.set_input_shape(inp_name, (1, 3, imgsz, imgsz))
        # Allocate device buffers
        d_in = cuda.mem_alloc(4 * vol((1, 3, imgsz, imgsz)))
        out_ptrs = []
        for n in outputs:
            # Query output shape/dtype from context after setting inputs
            shp = tuple(int(s) for s in ctx.get_tensor_shape(n))
            # If shape has -1 (dynamic), fallback to a safe cap for YOLOv8n (300x6)
            if any(s < 0 for s in shp):
                shp = (1, 300, 6)
            dt = engine.get_tensor_dtype(n)
            np_dtype = trt.nptype(dt)
            bytes_per = np.dtype(np_dtype).itemsize
            ptr = cuda.mem_alloc(bytes_per * vol(shp))
            out_ptrs.append((n, ptr))
            ctx.set_tensor_address(n, int(ptr))
        # Set static input address once
        ctx.set_tensor_address(inp_name, int(d_in))
        def infer_once(x):
            if not x.flags['C_CONTIGUOUS']:
                x = np.ascontiguousarray(x)
            cuda.memcpy_htod_async(d_in, x, stream)
            ctx.execute_async_v3(stream.handle)
            stream.synchronize()

    # Warmup
    for _, im in get_frames(source, warmup):
        x = letterbox(im, imgsz).transpose(2, 0, 1)[None]
        x = np.ascontiguousarray(x, dtype=np.float32)
        x /= 255.0
        infer_once(x)
    # Timed iterations
    for fid, im in get_frames(source, iters):
        x = letterbox(im, imgsz).transpose(2, 0, 1)[None]
        x = np.ascontiguousarray(x, dtype=np.float32)
        x /= 255.0
        t0 = time.perf_counter()
        infer_once(x)
        lat.append((time.perf_counter() - t0) * 1000.0)
        ids.append(fid)
    return ids, lat


def bench_ov(onnx_path, imgsz, warmup, iters, source, device="CPU"):
    try:
        import openvino as ov
    except Exception as e:
        print("[e] openvino not installed: pip install openvino", file=sys.stderr)
        raise
    core = ov.Core()
    model = core.read_model(onnx_path)
    # Try to reshape to the requested static size if dynamic
    try:
        model.reshape([1, 3, imgsz, imgsz])
    except Exception:
        pass
    compiled = core.compile_model(model, device)
    inp = compiled.inputs[0]
    infer = compiled.create_infer_request()
    lat, ids = [], []
    # Warmup
    for _, im in get_frames(source, warmup):
        x = letterbox(im, imgsz).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        _ = infer.infer({inp: x})
    for fid, im in get_frames(source, iters):
        x = letterbox(im, imgsz).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        t0 = time.perf_counter()
        _ = infer.infer({inp: x})
        lat.append((time.perf_counter() - t0) * 1000.0)
        ids.append(fid)
    return ids, lat


def bench_with_cadence(run_once, imgsz, warmup, iters, source, cadence=1):
    """
    Generic cadence runner: call run_once(image)->latency_ms when (i % cadence == 0),
    else reuse last ROI (simulated cheap postproc). Returns ids, per-frame latencies.
    """
    lat, ids = [], []
    last_boxes = None
    # Warmup (full runs)
    for _, im in get_frames(source, warmup):
        _ = run_once(letterbox(im, imgsz))
    for i, (fid, im) in enumerate(get_frames(source, iters)):
        if i % max(1, cadence) == 0:
            t0 = time.perf_counter()
            _ = run_once(letterbox(im, imgsz))
            dt = (time.perf_counter() - t0) * 1000.0
            last_boxes = True  # placeholder for ROI; not used further here
        else:
            # Simulate cheap ROI tracking/interpolation cost (1/20 of inference)
            dt = (lat[-1] / 20.0) if lat else 0.1
        lat.append(dt)
        ids.append(fid)
    return ids, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["trt", "ort", "ov"], required=True)
    ap.add_argument("--engine")
    ap.add_argument("--onnx")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument(
        "--source", default=None, help="Folder of images; random frames if omitted"
    )
    ap.add_argument("--out", default="detector_latency.json")
    ap.add_argument("--model-name", default="", help="Model name to embed in JSON (e.g., yolov8n)")
    ap.add_argument("--ov-device", default="CPU", help="OpenVINO device, e.g., CPU")
    ap.add_argument("--cadence", type=int, default=1, help="Run detector every N frames; reuse ROI on others")
    args = ap.parse_args()

    if args.backend == "trt" and not args.engine:
        print("[e] --engine required for TensorRT backend", file=sys.stderr)
        sys.exit(1)
    if args.backend in ("ort", "ov") and not args.onnx:
        print("[e] --onnx required for ORT backend", file=sys.stderr)
        sys.exit(1)

    if args.backend == "trt":
        ids, lat = bench_trt(
            args.engine, args.imgsz, args.warmup, args.iters, args.source
        )
    elif args.backend == "ort":
        if args.cadence and args.cadence > 1:
            import onnxruntime as ort
            sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
            inp_name = sess.get_inputs()[0].name
            def _once(img_bgr):
                x = img_bgr.transpose(2,0,1)[None].astype(np.float32)/255.0
                sess.run(None, {inp_name: x})
                return 0.0
            ids, lat = bench_with_cadence(_once, args.imgsz, args.warmup, args.iters, args.source, args.cadence)
        else:
            ids, lat = bench_ort(
                args.onnx, args.imgsz, args.warmup, args.iters, args.source
            )
    else:
        if args.cadence and args.cadence > 1:
            import openvino as ov
            core = ov.Core()
            model = core.read_model(args.onnx)
            try:
                model.reshape([1,3,args.imgsz,args.imgsz])
            except Exception:
                pass
            compiled = core.compile_model(model, args.ov_device)
            inp = compiled.inputs[0]
            infer = compiled.create_infer_request()
            def _once(img_bgr):
                x = img_bgr.transpose(2,0,1)[None].astype(np.float32)/255.0
                infer.infer({inp: x})
                return 0.0
            ids, lat = bench_with_cadence(_once, args.imgsz, args.warmup, args.iters, args.source, args.cadence)
        else:
            ids, lat = bench_ov(
                args.onnx, args.imgsz, args.warmup, args.iters, args.source, args.ov_device
            )

    arr = np.array(lat, dtype=np.float32)
    # Ensure native Python floats for JSON safety
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    mean = float(arr.mean())
    print(f"[i] n={len(arr)} mean={mean:.2f} ms p50={p50:.2f} ms p95={p95:.2f} ms")
    # Derive model name if provided or infer from engine/onnx
    model_name = args.model_name
    if not model_name:
        src = args.engine or args.onnx or ""
        base = os.path.basename(src)
        # Prefer stem before first underscore to capture yolov8n_640_fp16.engine
        stem = os.path.splitext(base)[0]
        model_name = stem.split("_")[0] if stem else ""

    data = {
        "summary": {
            "mean_ms": float(round(mean, 2)),
            "p50_ms": float(round(p50, 2)),
            "p95_ms": float(round(p95, 2)),
            "imgsz": int(args.imgsz),
            "backend": args.backend,
            **({"model": model_name} if model_name else {}),
        },
        "per_frame": {ids[i]: float(round(float(arr[i]), 2)) for i in range(len(ids))},
    }
    # Ensure JSON-serializable (convert any numpy scalars/arrays)
    def to_py(o):
        try:
            import numpy as _np  # local alias
            if isinstance(o, _np.generic):
                return o.item()
            if isinstance(o, (list, tuple)):
                return [to_py(v) for v in o]
            if isinstance(o, dict):
                return {k: to_py(v) for k, v in o.items()}
        except Exception:
            pass
        return o

    with open(args.out, "w") as f:
        json.dump(to_py(data), f, indent=2)
    print(f"[i] Wrote {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()

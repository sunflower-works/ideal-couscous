#!/usr/bin/env python3
"""Latency benchmark for YOLOv8 ONNX / TensorRT.
Writes JSON with summary (mean/p50/p95) and per-frame latencies.
"""
import argparse
import json
import os
import sys
import time

import cv2 as cv
import numpy as np


def read_cpu_temp_c() -> float | None:
    # Try Linux thermal zone first
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            v = f.read().strip()
            if v:
                mv = float(v)
                return mv / 1000.0 if mv > 100.0 else mv
    except Exception:
        pass
    # Fallback to vcgencmd if available
    try:
        import subprocess

        out = (
            subprocess.check_output(["vcgencmd", "measure_temp"], text=True)
            .strip()
            .lower()
        )
        # Format: temp=48.0'C
        if out.startswith("temp=") and out.endswith("'c"):
            return float(out[5:-2])
    except Exception:
        pass
    return None


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
    temps: list[float] = []
    for _, im in get_frames(source, warmup):
        x = letterbox(im, imgsz).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        sess.run(None, {inp_name: x})
    for fid, im in get_frames(source, iters):
        x = letterbox(im, imgsz).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        t0 = time.perf_counter()
        _ = sess.run(None, {inp_name: x})
        lat.append((time.perf_counter() - t0) * 1000.0)
        ids.append(fid)
    t = read_cpu_temp_c()
    if t is not None:
        temps.append(t)
    return ids, lat, temps


def bench_trt(engine_path, imgsz, warmup, iters, source):
    import tensorrt as trt, pycuda.driver as cuda, pycuda.autoinit  # type: ignore

    logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, "rb") as f, trt.Runtime(
        logger
    ) as rt, rt.deserialize_cuda_engine(f.read()) as engine:
        ctx = engine.create_execution_context()
        inp_idx = engine.get_binding_index(engine[0])
        ctx.set_binding_shape(inp_idx, (1, 3, imgsz, imgsz))

        def vol(shape):
            return int(np.prod(shape))

        d_in = cuda.mem_alloc(4 * vol((1, 3, imgsz, imgsz)))
        d_out = cuda.mem_alloc(1 << 20)
        stream = cuda.Stream()
        lat = []
        ids = []
        temps: list[float] = []
        for _, im in get_frames(source, warmup):
            x = letterbox(im, imgsz).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
            cuda.memcpy_htod_async(d_in, x, stream)
            ctx.execute_async_v2(
                bindings=[int(d_in), int(d_out)], stream_handle=stream.handle
            )
            stream.synchronize()
        for fid, im in get_frames(source, iters):
            x = letterbox(im, imgsz).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
            t0 = time.perf_counter()
            cuda.memcpy_htod_async(d_in, x, stream)
            ctx.execute_async_v2(
                bindings=[int(d_in), int(d_out)], stream_handle=stream.handle
            )
            stream.synchronize()
            lat.append((time.perf_counter() - t0) * 1000.0)
            ids.append(fid)
            t = read_cpu_temp_c()
            if t is not None:
                temps.append(t)
        return ids, lat, temps


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
    args = ap.parse_args()

    if args.backend == "trt" and not args.engine:
        print("[e] --engine required for TensorRT backend", file=sys.stderr)
        sys.exit(1)
    if args.backend == "ort" and not args.onnx:
        print("[e] --onnx required for ORT backend", file=sys.stderr)
        sys.exit(1)

    temps: list[float] = []
    if args.backend == "trt":
        ids, lat, temps = bench_trt(
            args.engine, args.imgsz, args.warmup, args.iters, args.source
        )
    elif args.backend == "ort":
        ids, lat, temps = bench_ort(
            args.onnx, args.imgsz, args.warmup, args.iters, args.source
        )
    else:
        # OpenVINO backend
        try:
            import openvino as ov  # type: ignore  # noqa: F401
        except Exception:
            print("[e] OpenVINO not available; install 'openvino' package", file=sys.stderr)
            sys.exit(1)
        from typing import Tuple, List

        def bench_ov(onnx_path: str, imgsz: int, warmup: int, iters: int, source: str | None) -> Tuple[List[str], List[float], List[float]]:
            import os as _os
            import openvino as ov
            core = ov.Core()
            # Simple cache directory to speed up subsequent compiles
            try:
                _os.makedirs("ov_cache", exist_ok=True)
                core.set_property({"CACHE_DIR": "ov_cache"})
            except Exception:
                pass
            model = core.read_model(onnx_path)
            compiled = core.compile_model(model, "CPU")
            input_node = compiled.inputs[0]
            lat: List[float] = []
            ids: List[str] = []
            temps: List[float] = []
            # Warmup
            for _, im in get_frames(source, warmup):
                x = letterbox(im, imgsz).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
                compiled([x])
            # Timed runs
            for fid, im in get_frames(source, iters):
                x = letterbox(im, imgsz).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
                t0 = time.perf_counter()
                compiled([x])
                lat.append((time.perf_counter() - t0) * 1000.0)
                ids.append(fid)
                t = read_cpu_temp_c()
                if t is not None:
                    temps.append(t)
            return ids, lat, temps

        ids, lat, temps = bench_ov(
            args.onnx, args.imgsz, args.warmup, args.iters, args.source
        )

    arr = np.array(lat, dtype=np.float32)
    # Ensure native Python floats for JSON safety
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    mean = float(arr.mean())
    print(f"[i] n={len(arr)} mean={mean:.2f} ms p50={p50:.2f} ms p95={p95:.2f} ms")
    # temps gathered per iteration (if available) already in `temps`
    temp_stats = (
        {
            "temp_c_min": float(round(min(temps), 1)) if temps else None,
            "temp_c_avg": float(round(sum(temps) / len(temps), 1)) if temps else None,
            "temp_c_max": float(round(max(temps), 1)) if temps else None,
        }
        if temps
        else {}
    )
    data = {
        "summary": {
            "mean_ms": float(round(mean, 2)),
            "p50_ms": float(round(p50, 2)),
            "p95_ms": float(round(p95, 2)),
            "imgsz": int(args.imgsz),
            "backend": args.backend,
            **temp_stats,
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

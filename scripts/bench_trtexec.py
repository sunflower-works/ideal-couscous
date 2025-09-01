#!/usr/bin/env python3
"""Fallback benchmark using trtexec when Python TensorRT+pycuda path unavailable.
Generates JSON similar to bench_yolov8.py (summary only; per_frame approximated).
"""
import argparse
import json
import os
import re
import subprocess
import sys
from statistics import mean
import shutil


def find_trtexec():
    # Prefer explicit override
    env = os.environ
    t = env.get("TRTEXEC")
    if t and (os.path.isfile(t) and os.access(t, os.X_OK) or shutil.which(t)):
        return t
    # PATH
    w = shutil.which("trtexec")
    if w:
        return w
    # Common Jetson locations
    candidates = [
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/src/tensorrt/samples/trtexec",
        "/opt/nvidia/tensorrt/bin/trtexec",
        "/usr/local/tensorrt/bin/trtexec",
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def parse_metrics(text: str):
    # Try TRT10+ format: "Inference time: min=..., mean=..., median=..., percentile (95%)=..."
    m = re.search(r"Inference time:.*?mean=([0-9.]+).*?median=([0-9.]+).*?percentile\s*\(95%\)\s*=([0-9.]+)", text)
    if m:
        return {
            "mean_ms": float(m.group(1)),
            "p50_ms": float(m.group(2)),
            "p95_ms": float(m.group(3)),
        }
    # TRT8/9 format: "Average on N runs - GPU latency: min = ..., mean = ..., median = ..., max = ..."
    m = re.search(r"Average on .*? runs\s*-\s*GPU latency:.*?mean\s*=\s*([0-9.]+).*?median\s*=\s*([0-9.]+)", text)
    if m:
        mean_v = float(m.group(1))
        p50_v = float(m.group(2))
        # Try to extract 95th percentile if present elsewhere
        m95 = re.search(r"95%\s*=\s*([0-9.]+)\s*ms", text)
        return {
            "mean_ms": mean_v,
            "p50_ms": p50_v,
            "p95_ms": float(m95.group(1)) if m95 else mean_v,
        }
    # Generic fallback: first "mean = X ms" and optional "median = Y ms"
    m = re.search(r"mean\s*=\s*([0-9.]+)\s*ms", text)
    if m:
        mean_v = float(m.group(1))
        m50 = re.search(r"median\s*=\s*([0-9.]+)\s*ms", text) or re.search(r"p50\s*=\s*([0-9.]+)\s*ms", text)
        m95 = re.search(r"95%\s*=\s*([0-9.]+)\s*ms", text) or re.search(r"percentile\s*\(95%\)\s*=\s*([0-9.]+)", text)
        return {
            "mean_ms": mean_v,
            "p50_ms": float(m50.group(1)) if m50 else mean_v,
            "p95_ms": float(m95.group(1)) if m95 else mean_v,
        }
    return None


def run_trtexec(engine, warmup, iters):
    trtexec = find_trtexec()
    if not trtexec:
        raise SystemExit("trtexec not found; set TRTEXEC env or install TensorRT")
    # Use --iterations for fixed iteration count; --noDataTransfers keeps IO realistic? we keep default.
    cmd = [
        trtexec,
        f"--loadEngine={engine}",
        f"--warmUp={warmup}",  # warmUp in milliseconds, approximate
        f"--iterations={iters}",
        "--avgRuns=1",  # we want raw iteration list
    ]
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
        raise SystemExit("trtexec failed")
    lines = proc.stdout.splitlines()
    # Collect per iteration lines: pattern like 'Iteration: <n> Time: <ms>' may differ across versions; fallback to latency table.
    times = []
    iter_re = re.compile(r"\b([0-9]+)\.\s+ms:\s*([0-9.]+)")  # placeholder (unlikely)
    for ln in lines:
        # Newer trtexec prints 'Inference time: min=..., mean=..., median=..., max=..., percentile...' once.
        if "Inference time:" in ln:
            summary = parse_metrics(ln)
            if summary:
                return summary, []

    # If no summary line, try parse any collected times
    if times:
        avg = mean(times)
        return {"mean_ms": avg, "p50_ms": avg, "p95_ms": avg}, times
    # Last try: parse across full output
    summary = parse_metrics(proc.stdout)
    if summary:
        return summary, []

    raise SystemExit("Could not parse trtexec output for latency metrics")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--out", default="detector_latency_trtexec.json")
    ap.add_argument("--model-name", default="", help="Model name to embed in JSON (e.g., yolov8n)")

    args = ap.parse_args()
    summary, per_iter = run_trtexec(args.engine, args.warmup, args.iters)
    # Fill missing with mean if needed
    for k in ("mean_ms", "p50_ms", "p95_ms"):
        if summary[k] is None:
            summary[k] = round(summary["mean_ms"], 2)
    data = {
        "summary": {
            **{k: round(float(v), 2) for k, v in summary.items()},
            "imgsz": args.imgsz,
            "backend": "trtexec",
            **({"model": args.model_name} if args.model_name else {}),

        },
        "per_frame": {
            f"iter_{i:04d}": round(float(t), 2) for i, t in enumerate(per_iter)
        },
    }
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[i] Wrote {args.out}")


if __name__ == "__main__":
    main()

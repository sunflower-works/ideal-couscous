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


def run_trtexec(engine, warmup, iters):
    # Use --iterations for fixed iteration count; --noDataTransfers keeps IO realistic? we keep default.
    cmd = [
        "trtexec",
        f"--loadEngine={engine}",
        f"--warmUp={warmup}",  # warmUp in milliseconds, approximate
        f"--iterations={iters}",
        "--avgRuns=1",  # we want raw iteration list
    ]
    env = os.environ.copy()
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
            # parse mean/median (p50)
            m = re.search(r"mean=([0-9.]+)", ln)
            p50 = re.search(r"median=([0-9.]+)", ln)
            p95 = re.search(r"percentile\s*\(95%\)=([0-9.]+)", ln)
            summary = {
                "mean_ms": float(m.group(1)) if m else None,
                "p50_ms": float(p50.group(1)) if p50 else None,
                "p95_ms": float(p95.group(1)) if p95 else None,
            }
            return summary, []
    # If no summary line, try parse any collected times
    if times:
        avg = mean(times)
        return {"mean_ms": avg, "p50_ms": avg, "p95_ms": avg}, times
    raise SystemExit("Could not parse trtexec output for latency metrics")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--out", default="detector_latency_trtexec.json")
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

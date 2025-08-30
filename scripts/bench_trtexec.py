#!/usr/bin/env python3
"""Fallback TensorRT engine benchmark using trtexec CLI.
Produces a JSON summary compatible with bench_yolov8.py output subset.
This does NOT provide per-frame latencies (trtexec doesn't expose them directly).

Usage:
  python bench_trtexec.py --engine yolov8n_416_fp16.engine --imgsz 416 --warmup 30 --iters 300 --out detector_latency_jetson_416.json

It runs trtexec once with requested iteration counts and parses the latency line.
"""
import argparse, json, os, re, shutil, subprocess, sys
from datetime import datetime

def parse_latency(trtexec_stdout: str):
    # Example line:
    # Latency: min = 1.234 ms, max = 2.345 ms, mean = 1.567 ms, median = 1.500 ms, percentile(99%) = 2.300 ms
    lat_re = re.compile(r"Latency: min = ([0-9.]+) ms, max = ([0-9.]+) ms, mean = ([0-9.]+) ms, median = ([0-9.]+) ms, percentile\(99%\) = ([0-9.]+) ms")
    for line in trtexec_stdout.splitlines()[::-1]:  # search from bottom
        m = lat_re.search(line)
        if m:
            return {
                "min_ms": float(m.group(1)),
                "max_ms": float(m.group(2)),
                "mean_ms": float(m.group(3)),
                "p50_ms": float(m.group(4)),
                "p99_ms": float(m.group(5)),
            }
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=300, help="Number of timed iterations (trtexec --iterations)")
    ap.add_argument("--out", default="detector_latency_trtexec.json")
    args = ap.parse_args()

    if not shutil.which("trtexec"):
        print("[e] trtexec not found in PATH", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.engine):
        print(f"[e] Engine file not found: {args.engine}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        "trtexec",
        f"--loadEngine={args.engine}",
        "--useSpinWait",  # lower jitter
        f"--warmUp={args.warmup}",
        f"--iterations={args.iters}",
        "--streams=1",
        "--noDataTransfers",  # pure compute; remove if you want H2D/D2H included
        "--avgRuns=1",
    ]
    print("[i] Running:", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = proc.stdout
    with open(args.out + ".raw.txt", "w") as f:
        f.write(out)
    if proc.returncode != 0:
        print("[e] trtexec failed; raw log saved", file=sys.stderr)
        sys.exit(proc.returncode)

    lat = parse_latency(out)
    if not lat:
        print("[e] Could not parse latency line; see raw log", file=sys.stderr)
        sys.exit(2)
    summary = {
        "backend": "trtexec",
        "imgsz": args.imgsz,
        **{k: round(v, 2) for k, v in lat.items()},
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    # Align keys with bench_yolov8 primary subset (mean_ms, p50_ms)
    result = {"summary": summary, "note": "Per-frame latencies unavailable via trtexec fallback"}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("[i] Summary:", summary)
    print(f"[i] Wrote {args.out} and raw log {args.out}.raw.txt")

if __name__ == "__main__":
    main()


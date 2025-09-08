#!/usr/bin/env python3
"""
soak_runner.py

Run a mixed workload soak test: periodically triggers perception benchmarks, CPU/GPU stress, and I/O
while logging health via tegrastats_logger.py (if available).

Usage:
  python soak_runner.py --duration-sec 3600 --out-dir logs/soak --imgsz 640 --trt-root $HOME/edge-yolo \
      [--stress --cpu 2 --io 1]

This script is conservative: if trtexec paths or edge-yolo engines do not exist, it will skip TRT phases.
Outputs:
  <out-dir>/events.log, optional sub-logs created by invoked helpers.
"""
import argparse
import os
import shutil
import subprocess as sp
import sys
import time
from datetime import datetime
from pathlib import Path


def stamp() -> str:
    return datetime.utcnow().isoformat() + "Z"


def maybe(
    cmd: list[str] | str, cwd: str | None = None, env: dict | None = None, logf=None
):
    try:
        p = sp.Popen(
            cmd,
            cwd=cwd,
            env=env,
            shell=isinstance(cmd, str),
            stdout=sp.PIPE,
            stderr=sp.STDOUT,
            text=True,
        )
        for line in p.stdout:  # type: ignore
            if logf:
                logf.write(line)
        p.wait()
    except Exception as e:
        if logf:
            logf.write(f"[w] command failed: {cmd} err={e}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration-sec", type=int, default=1800)
    ap.add_argument("--out-dir", default="logs/soak")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--trt-root", default=str(Path.home() / "edge-yolo"))
    ap.add_argument("--stress", action="store_true")
    ap.add_argument("--cpu", type=int, default=1)
    ap.add_argument("--io", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events = (out_dir / "events.log").open("w")
    events.write(
        f"[{stamp()}] soak_start duration={args.duration_sec}s imgsz={args.imgsz} trt_root={args.trt_root}\n"
    )
    events.flush()

    # Start tegrastats logger if present
    tlog = sp.Popen(
        [
            sys.executable,
            str(Path(__file__).with_name("tegrastats_logger.py")),
            "--out",
            str(out_dir / "tegrastats.csv"),
            "--interval-ms",
            "1000",
            "--duration-sec",
            str(args.duration_sec),
        ]
    )

    t_end = time.time() + max(60, args.duration_sec)
    phase = 0
    try:
        while time.time() < t_end:
            # Phase A: optional TRT inference warm bursts
            if Path(args.trt_root).exists():
                events.write(f"[{stamp()}] phase{phase}: trt_burst\n")
                events.flush()
                maybe(
                    f"python3 bench_trtexec.py --engine yolov8n_{args.imgsz}_fp16.engine --imgsz {args.imgsz} --iters 100 --warmup 10",
                    cwd=args.trt_root,
                    logf=events,
                )
            else:
                events.write(f"[{stamp()}] phase{phase}: skip trt (root not found)\n")
                events.flush()
            # Phase B: optional stress-ng
            if args.stress and shutil.which("stress-ng"):
                events.write(
                    f"[{stamp()}] phase{phase}: stress-ng cpu={args.cpu} io={args.io}\n"
                )
                events.flush()
                maybe(
                    f"stress-ng --cpu {args.cpu} --io {args.io} --timeout 60s --metrics-brief",
                    logf=events,
                )
            else:
                events.write(f"[{stamp()}] phase{phase}: idle (no stress)\n")
                events.flush()
            # Short sleep between phases
            time.sleep(5)
            phase += 1
    except KeyboardInterrupt:
        pass
    finally:
        events.write(f"[{stamp()}] soak_end\n")
        events.flush()
        events.close()
        try:
            tlog.terminate()
        except Exception:
            pass

    print("[i] Soak logs at", out_dir)


if __name__ == "__main__":
    main()

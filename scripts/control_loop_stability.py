#!/usr/bin/env python3
"""
control_loop_stability.py

Measure control-loop timing stability at a target frequency.
Logs per-iteration timestamps and jitter; writes a CSV and prints summary.

Usage:
  python control_loop_stability.py --hz 50 --duration-sec 120 --out logs/ctrl_loop_50hz.csv \
      [--busywork 0.1] [--workload "bash -lc 'stress-ng --cpu 2'" ]

Columns: ts_iso, idx, dt_target_ms, dt_actual_ms, jitter_ms
"""
import argparse
import os
import subprocess as sp
import sys
import time
from datetime import datetime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--duration-sec", type=int, default=60)
    ap.add_argument("--out", default="logs/ctrl_loop.csv")
    ap.add_argument(
        "--busywork",
        type=float,
        default=0.0,
        help="seconds of dummy work per cycle (0=none)",
    )
    ap.add_argument(
        "--workload", default="", help="optional background workload command"
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Optional background workload
    bg = None
    if args.workload:
        print("[i] Starting workload:", args.workload)
        bg = sp.Popen(args.workload, shell=True)

    period = 1.0 / max(1e-6, args.hz)
    t_end = time.perf_counter() + max(1, args.duration_sec)
    t_next = time.perf_counter()
    idx = 0
    rows = []

    try:
        with open(args.out, "w") as f:
            f.write("ts_iso,idx,dt_target_ms,dt_actual_ms,jitter_ms\n")
            while time.perf_counter() < t_end:
                t0 = time.perf_counter()
                # Busywork simulation
                if args.busywork > 0:
                    t_busy_end = t0 + args.busywork
                    x = 0.0
                    while time.perf_counter() < t_busy_end:
                        x += 1.0  # dummy FP ops
                # Sleep until next tick
                t_next += period
                dt_sleep = max(0.0, t_next - time.perf_counter())
                if dt_sleep > 0:
                    time.sleep(dt_sleep)
                t1 = time.perf_counter()
                dt_actual = t1 - t0
                jitter = dt_actual - period
                ts = datetime.utcnow().isoformat() + "Z"
                f.write(
                    f"{ts},{idx},{period*1000.0:.3f},{dt_actual*1000.0:.3f},{jitter*1000.0:.3f}\n"
                )
                idx += 1
    except KeyboardInterrupt:
        pass
    finally:
        if bg is not None:
            try:
                bg.terminate()
            except Exception:
                pass

    print("[i] Wrote:", args.out)


if __name__ == "__main__":
    main()

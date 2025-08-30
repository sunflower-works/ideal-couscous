#!/usr/bin/env python3
"""Aggregate detector latency JSON files into comparative CSV.
Looks for files matching detector_latency_*_<precision>.json
Columns: device,imgsz,precision,mean_ms,p50_ms,p95_ms,fps_mean,speedup_vs_fp32
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict


def parse_file(path):
    with open(path) as f:
        data = json.load(f)
    summ = data.get("summary", {})
    fname = os.path.basename(path)
    # Extract device and precision heuristically
    m = re.match(
        r"detector_latency_(?P<device>[^_]+)_(?P<imgsz>\d+)_(?P<precision>[^.]+)\.json",
        fname,
    )
    if not m:
        return None
    d = m.groupdict()
    return {
        "device": d["device"],
        "imgsz": int(d["imgsz"]),
        "precision": d["precision"],
        "mean_ms": float(summ.get("mean_ms", 0)),
        "p50_ms": float(summ.get("p50_ms", 0)),
        "p95_ms": float(summ.get("p95_ms", 0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="detector_latency_*.json")
    ap.add_argument("--out", default="latency_comparative.csv")
    args = ap.parse_args()
    rows = []
    for path in glob.glob(args.glob):
        r = parse_file(path)
        if r:
            rows.append(r)
    if not rows:
        print("[e] No latency JSON files found")
        return
    # Group by (device,imgsz)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["device"], r["imgsz"])].append(r)
    # Compute speedup vs fp32
    out_rows = []
    for (device, imgsz), lst in sorted(groups.items()):
        base_fp32 = next((x for x in lst if x["precision"] == "fp32"), None)
        base_mean = base_fp32["mean_ms"] if base_fp32 else None
        for r in sorted(lst, key=lambda x: x["precision"]):
            speedup = base_mean / r["mean_ms"] if base_mean else ""
            out_rows.append(
                {
                    "device": device,
                    "imgsz": imgsz,
                    "precision": r["precision"],
                    "mean_ms": f"{r['mean_ms']:.2f}",
                    "p50_ms": f"{r['p50_ms']:.2f}",
                    "p95_ms": f"{r['p95_ms']:.2f}",
                    "fps_mean": f"{1000.0/r['mean_ms']:.2f}",
                    "speedup_vs_fp32": f"{speedup:.2f}" if speedup else "",
                }
            )
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "device",
                "imgsz",
                "precision",
                "mean_ms",
                "p50_ms",
                "p95_ms",
                "fps_mean",
                "speedup_vs_fp32",
            ],
        )
        w.writeheader()
        w.writerows(out_rows)
    print(f"[i] Wrote {args.out} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()

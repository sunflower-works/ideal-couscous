#!/usr/bin/env python3
"""Aggregate detector latency JSON files into comparative CSV.
Looks for files matching detector_latency_<device>_<imgsz>_<precision>[ _suffix].json
Aggregates multiple trials per (device,imgsz,precision) by averaging numeric fields.
Columns: device,imgsz,precision,mean_ms,p50_ms,p95_ms,fps_mean,speedup,
         temp_c_min,temp_c_avg,temp_c_max
Baseline for speedup: prefer fp32; otherwise ORT if present; else 1.00.
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
    # Extract device, imgsz, precision (ignore optional suffix like _t1)
    m = re.match(
        r"detector_latency_(?P<device>[^_]+)_(?P<imgsz>\d+)_(?P<precision>[a-z0-9_]+)(?:_.*)?\.json",
        fname,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    d = m.groupdict()
    # Normalise precision: strip only trailing trial suffixes like _t1 while
    # preserving legitimate underscore tokens such as ov_int8.
    raw_prec = d["precision"]
    norm_prec = re.sub(r"_t\d+$", "", raw_prec, flags=re.IGNORECASE)
    out = {
        "device": d["device"],
        "imgsz": int(d["imgsz"]),
        "precision": norm_prec,
        "mean_ms": float(summ.get("mean_ms", 0)),
        "p50_ms": float(summ.get("p50_ms", 0)),
        "p95_ms": float(summ.get("p95_ms", 0)),
    }
    # Optional temperature fields
    for k in ("temp_c_min", "temp_c_avg", "temp_c_max"):
        v = summ.get(k)
        if v is not None:
            try:
                out[k] = float(v)
            except Exception:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="detector_latency_*.json")
    ap.add_argument("--out", default="latency_comparative.csv")
    args = ap.parse_args()
    files = []
    for path in glob.glob(args.glob):
        r = parse_file(path)
        if r:
            files.append(r)
    if not files:
        print("[e] No latency JSON files found")
        return
    # Group by (device,imgsz,precision) to average trials
    trials = defaultdict(list)
    for r in files:
        trials[(r["device"], r["imgsz"], r["precision"])].append(r)

    # Aggregate fields per (device,imgsz,precision)
    def avg_field(lst, key):
        vals = [x[key] for x in lst if key in x]
        return sum(vals) / len(vals) if vals else None
    def min_field(lst, key):
        vals = [x[key] for x in lst if key in x]
        return min(vals) if vals else None
    def max_field(lst, key):
        vals = [x[key] for x in lst if key in x]
        return max(vals) if vals else None

    merged = defaultdict(list)  # (device,imgsz) -> list of averaged rows per precision
    for (device, imgsz, prec), lst in trials.items():
        row = {
            "device": device,
            "imgsz": imgsz,
            "precision": prec,
        }
        # Latency stats averaged across trials
        for k in ("mean_ms", "p50_ms", "p95_ms"):
            v = avg_field(lst, k)
            if v is not None:
                row[k] = v
        # Temps: min(min), avg(avg), max(max) across trials
        vmin = min_field(lst, "temp_c_min")
        if vmin is not None:
            row["temp_c_min"] = vmin
        vavg = avg_field(lst, "temp_c_avg")
        if vavg is not None:
            row["temp_c_avg"] = vavg
        vmax = max_field(lst, "temp_c_max")
        if vmax is not None:
            row["temp_c_max"] = vmax
        merged[(device, imgsz)].append(row)

    # Compute speedup vs baseline (fp32 preferred, else ort)
    out_rows = []
    for (device, imgsz), lst in sorted(merged.items()):
        base = next((x for x in lst if x["precision"].lower() == "fp32"), None)
        if base is None:
            base = next((x for x in lst if x["precision"].lower() == "ort"), None)
        base_mean = base.get("mean_ms") if base else None
        for r in sorted(lst, key=lambda x: x["precision"]):
            mean_ms = r.get("mean_ms")
            fps = (1000.0 / mean_ms) if mean_ms else None
            if base_mean and mean_ms:
                speedup = base_mean / mean_ms
            else:
                speedup = 1.0 if base is r else None
            out = {
                "device": device,
                "imgsz": imgsz,
                "precision": r["precision"],
                "mean_ms": f"{mean_ms:.2f}" if mean_ms else "",
                "p50_ms": f"{r.get('p50_ms', 0):.2f}" if r.get("p50_ms") is not None else "",
                "p95_ms": f"{r.get('p95_ms', 0):.2f}" if r.get("p95_ms") is not None else "",
                "fps_mean": f"{fps:.2f}" if fps else "",
                "speedup": f"{speedup:.2f}" if speedup is not None else "",
                "temp_c_min": f"{r.get('temp_c_min', 0):.1f}" if r.get("temp_c_min") is not None else "",
                "temp_c_avg": f"{r.get('temp_c_avg', 0):.1f}" if r.get("temp_c_avg") is not None else "",
                "temp_c_max": f"{r.get('temp_c_max', 0):.1f}" if r.get("temp_c_max") is not None else "",
            }
            out_rows.append(out)
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
                "speedup",
                "temp_c_min",
                "temp_c_avg",
                "temp_c_max",
            ],
        )
        w.writeheader()
        w.writerows(out_rows)
    print(f"[i] Wrote {args.out} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()

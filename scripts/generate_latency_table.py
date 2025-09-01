#!/usr/bin/env python3
"""Generate LaTeX tables from latency CSV (latency_comparative*.csv).

Reads one or more CSV files (pattern) containing columns:
 device,imgsz,precision,mean_ms,p50_ms,p95_ms,fps_mean,speedup_vs_fp32
Outputs a LaTeX file with one table per (device,imgsz) group.

Usage:
  python generate_latency_table.py \
      --csv latency_comparative_640.csv \
      --out results/figures/latency_table_generated.tex \
      --caption "YOLOv8n latency on Jetson Orin Nano" --label tab:lat_jetson

Multiple CSV inputs are concatenated before grouping.
"""
import argparse
import csv
from pathlib import Path
from collections import defaultdict

HEADER_ORDER = ["precision", "mean_ms", "p50_ms", "p95_ms", "fps_mean", "speedup_vs_fp32"]
HEADER_LABELS = {
    "precision": "Precision",
    "mean_ms": "Mean (ms)",
    "p50_ms": "p50 (ms)",
    "p95_ms": "p95 (ms)",
    "fps_mean": "FPS",
    "speedup_vs_fp32": "Speedup vs FP32",
}

def load_rows(paths):
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(row)
    return rows

def fmt(v):
    if v is None or v == "":
        return "--"
    # Keep numeric formatting short
    try:
        f = float(v)
        if f.is_integer():
            return str(int(f))
        return f"{f:.2f}"
    except ValueError:
        return v

def build_table(device, size, rows, caption_base, label_base):
    # Sort by typical order fp32, fp16, int8, others alphabetically
    order_key = {"fp32":0, "fp16":1, "int8":2}
    rows_sorted = sorted(rows, key=lambda r: order_key.get(r['precision'], 99))
    header_line = " ".join([HEADER_LABELS[h] for h in HEADER_ORDER])
    lines = []
    lines.append("% Device: {}  Size: {}".format(device, size))
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\small")
    col_spec = "l" + "r" * (len(HEADER_ORDER) - 1)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    header = " ".join(HEADER_LABELS[h] for h in HEADER_ORDER).replace(" ", " & ") + " \\\\"
    lines.append(header)
    lines.append("\\midrule")
    for r in rows_sorted:
        row_line = " & ".join(fmt(r.get(h)) for h in HEADER_ORDER) + " \\\\" 
        lines.append(row_line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    cap = f"{caption_base} (device={device}, {size}px)"
    label = f"{label_base}_{device}_{size}".replace(" ", "_")
    lines.append(f"\\caption{{{cap}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    lines.append("")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', nargs='+', required=True)
    ap.add_argument('--out', default='results/figures/latency_table_generated.tex')
    ap.add_argument('--caption', default='YOLOv8n latency results')
    ap.add_argument('--label', default='tab:latency')
    ap.add_argument('--append', action='store_true', help='Append instead of overwrite')
    args = ap.parse_args()
    rows = load_rows(args.csv)
    if not rows:
        raise SystemExit('[e] No rows loaded from CSV inputs')
    groups = defaultdict(list)
    for r in rows:
        groups[(r['device'], r['imgsz'])].append(r)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = 'a' if args.append else 'w'
    with open(out_path, mode) as f:
        if not args.append:
            f.write('% Auto-generated latency tables. Do not edit manually.\n')
            f.write('% Source CSV files: ' + ', '.join(args.csv) + '\n\n')
        for (device, size), lst in sorted(groups.items()):
            f.write(build_table(device, size, lst, args.caption, args.label))
    print(f"[i] Wrote LaTeX tables -> {out_path}")

if __name__ == '__main__':
    main()


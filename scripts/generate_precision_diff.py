#!/usr/bin/env python3
"""Generate accuracy and latency deltas between precisions.

Inputs (defaults assume prior batch run at imgsz):
  --latency-csv latency_comparative_<imgsz>.csv
  --map-glob map_metrics_<imgsz>_*.json
  --imgsz 640
  --out-tex results/figures/precision_diff_summary.tex
  --out-json results/figures/precision_diff_summary.json

Computes:
  * Δ mean latency vs FP32 (ms, %)
  * Speedup (already in CSV, revalidated)
  * Δ mAP50 / mAP50-95 vs FP32
Outputs LaTeX snippet with bullet list + small table.
Gracefully degrades if some precisions missing.
"""
import argparse
import csv
import glob
import json
from pathlib import Path

PREC_ORDER = ["fp32", "fp16", "int8"]


def load_latency(csv_path):
    data = {}
    if not Path(csv_path).exists():
        return data
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            data[row["precision"].lower()] = row
    return data


def load_map(pattern, imgsz):
    out = {}
    for p in glob.glob(pattern):
        try:
            with open(p) as f:
                js = json.load(f)
            if js.get("imgsz") != imgsz:
                continue
            parts = Path(p).stem.split("_")
            prec = parts[-1].lower()
            out[prec] = {
                "map50": float(js.get("map50", js.get("metrics/mAP50(B)", 0.0)) or 0.0),
                "map50_95": float(
                    js.get("map50_95", js.get("metrics/mAP50-95(B)", 0.0)) or 0.0
                ),
            }
        except Exception:
            continue
    return out


def fmt(v, digits=2):
    return f"{v:.{digits}f}" if isinstance(v, (int, float)) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency-csv", required=True)
    ap.add_argument("--map-glob", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out-tex", default="results/figures/precision_diff_summary.tex")
    ap.add_argument("--out-json", default="results/figures/precision_diff_summary.json")
    args = ap.parse_args()

    lat = load_latency(args.latency_csv)
    maps = load_map(args.map_glob, args.imgsz)

    base_lat = (
        float(lat.get("fp32", {}).get("mean_ms", "nan")) if "fp32" in lat else None
    )
    base_m50 = maps.get("fp32", {}).get("map50") if "fp32" in maps else None
    base_m5095 = maps.get("fp32", {}).get("map50_95") if "fp32" in maps else None

    rows = []
    for prec in PREC_ORDER:
        if prec not in lat and prec not in maps:
            continue
        lrow = lat.get(prec, {})
        mrow = maps.get(prec, {})
        mean_ms = float(lrow.get("mean_ms", "nan")) if "mean_ms" in lrow else None
        speed = lrow.get("speedup_vs_fp32")
        d_ms = None
        d_pct = None
        if base_lat and mean_ms and prec != "fp32":
            d_ms = mean_ms - base_lat
            d_pct = (d_ms / base_lat) * 100.0
        dm50 = dm5095 = None
        if base_m50 is not None and "map50" in mrow and prec != "fp32":
            dm50 = mrow["map50"] - base_m50
        if base_m5095 is not None and "map50_95" in mrow and prec != "fp32":
            dm5095 = mrow["map50_95"] - base_m5095
        rows.append(
            {
                "precision": prec,
                "mean_ms": mean_ms,
                "speedup": float(speed) if speed else None,
                "delta_ms_vs_fp32": d_ms,
                "delta_pct_vs_fp32": d_pct,
                "map50": mrow.get("map50"),
                "delta_map50": dm50,
                "map50_95": mrow.get("map50_95"),
                "delta_map50_95": dm5095,
            }
        )

    summary = {"imgsz": args.imgsz, "rows": rows}
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))

    # Build LaTeX
    lines = [
        "% Auto-generated precision delta summary",
        "\\begin{table}[h]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "Prec & Mean(ms) & Speedup & dMs & d\% & mAP50 & d mAP50 & mAP50-95 & d mAP50-95 \\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            " {p} & {mm} & {sp} & {dms} & {dp} & {m50} & {dm50} & {m95} & {dm95} \\".format(
                p=r["precision"],
                mm=("--" if r["mean_ms"] is None else fmt(r["mean_ms"])),
                sp=("--" if r["speedup"] is None else fmt(r["speedup"])),
                dms=(
                    "--"
                    if r["delta_ms_vs_fp32"] is None
                    else fmt(r["delta_ms_vs_fp32"])
                ),
                dp=(
                    "--"
                    if r["delta_pct_vs_fp32"] is None
                    else fmt(r["delta_pct_vs_fp32"])
                ),
                m50=("--" if r["map50"] is None else fmt(r["map50"], 4)),
                dm50=("--" if r["delta_map50"] is None else fmt(r["delta_map50"], 4)),
                m95=("--" if r["map50_95"] is None else fmt(r["map50_95"], 4)),
                dm95=(
                    "--" if r["delta_map50_95"] is None else fmt(r["delta_map50_95"], 4)
                ),
            )
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{Precision latency & accuracy deltas (imgsz={args.imgsz}).}}",
        "\\label{tab:precision_deltas}",
        "\\end{table}",
        "",
    ]
    out_tex = Path(args.out_tex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines))
    print(f"[i] Wrote {out_json} and {out_tex}")


if __name__ == "__main__":
    main()

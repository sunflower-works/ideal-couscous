#!/usr/bin/env python3
"""Consolidate edge inference artifacts (latency, mAP, INT8 coverage) into LaTeX + JSON.

Inputs (flexible):
  --latency-csv latency_comparative_640.csv (or multiple via --latency-csv ...)
  --latency-json-glob 'detector_latency_jetson_640_*.json' (used if no CSV provided to build ad‑hoc CSV in memory)
  --map-glob 'map_metrics_640_*.json'
  --int8-layers int8_layers_640.json (optional)
  --device Jetson Orin Nano (label)
  --imgsz 640
  --out-tex results/figures/edge_inference_report.tex
  --out-json results/figures/edge_inference_report.json
  --caption "YOLOv8n Edge Inference Summary" (LaTeX caption context)

Behavior:
  * Loads latency metrics (mean_ms, fps_mean, speedup_vs_fp32) per precision.
  * Loads mAP metrics (map50, map50_95) from map_metrics_* json files.
  * Loads INT8 layer coverage json (ratio) if provided.
  * Produces consolidated JSON + a LaTeX snippet with:
      - A table merging latency + accuracy (per precision row)
      - An optional note about INT8 layer coverage.
  * Does NOT regenerate latency CSV (depends on upstream scripts) unless only JSONs present.

Example:
  python consolidate_edge_report.py \
      --latency-csv latency_comparative_640.csv \
      --map-glob 'map_metrics_640_*.json' \
      --int8-layers int8_layers_640.json \
      --device 'Jetson Orin Nano' --imgsz 640
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Dict, List

PRECISION_ORDER = {"fp32":0, "fp16":1, "int8":2}

LAT_HEADERS = ["precision","mean_ms","p50_ms","p95_ms","fps_mean","speedup_vs_fp32"]
ACC_HEADERS = ["map50","map50_95"]


def load_latency_csv(paths: List[Path], imgsz: int) -> Dict[str, Dict[str,str]]:
    data: Dict[str,Dict[str,str]] = {}
    for p in paths:
        if not p.exists():
            continue
        with p.open() as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    if int(row.get("imgsz", -1)) != imgsz:
                        continue
                except Exception:
                    continue
                prec = row["precision"].lower()
                data[prec] = row
    return data


def load_latency_from_json(pattern: str, imgsz: int) -> Dict[str, Dict[str,str]]:
    out: Dict[str,Dict[str,str]] = {}
    for path in glob.glob(pattern):
        try:
            with open(path) as f:
                js = json.load(f)
            summ = js.get("summary", {})
            if summ.get("imgsz") != imgsz:
                continue
            prec = Path(path).stem.split('_')[-1].lower()
            out[prec] = {
                'precision': prec,
                'mean_ms': f"{summ.get('mean_ms','')}",
                'p50_ms': f"{summ.get('p50_ms','')}",
                'p95_ms': f"{summ.get('p95_ms','')}",
                'fps_mean': f"{(1000.0/float(summ['mean_ms'])):.2f}" if 'mean_ms' in summ else '',
                'speedup_vs_fp32': ''  # fill later
            }
        except Exception:
            continue
    # Fill speedups if fp32 available
    if 'fp32' in out:
        base = float(out['fp32']['mean_ms'])
        for prec,row in out.items():
            if prec == 'fp32':
                row['speedup_vs_fp32'] = '1.00'
            else:
                try:
                    row['speedup_vs_fp32'] = f"{base/float(row['mean_ms']):.2f}"
                except Exception:
                    row['speedup_vs_fp32'] = ''
    return out


def load_map(pattern: str, imgsz: int) -> Dict[str, Dict[str,float]]:
    res: Dict[str, Dict[str,float]] = {}
    for path in glob.glob(pattern):
        try:
            with open(path) as f:
                js = json.load(f)
            if js.get('imgsz') != imgsz:
                continue
            # Expect filename map_metrics_<imgsz>_<precision>.json
            parts = Path(path).stem.split('_')
            # find last token that's a precision candidate
            prec = parts[-1].lower()
            res[prec] = {
                'map50': float(js.get('map50', js.get('metrics/mAP50(B)', 0.0)) or 0.0),
                'map50_95': float(js.get('map50_95', js.get('metrics/mAP50-95(B)', 0.0)) or 0.0)
            }
        except Exception:
            continue
    return res


def load_int8_layers(path: str | None) -> Dict[str, float]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with p.open() as f:
            js = json.load(f)
        return {
            'int8_ratio': float(js.get('int8_ratio', 0.0)),
            'int8_layers': int(js.get('int8_layers', 0)),
            'layers_total': int(js.get('layers_total', 0))
        }
    except Exception:
        return {}


def build_rows(lat: Dict[str,Dict[str,str]], maps: Dict[str,Dict[str,float]]) -> List[Dict[str,str]]:
    all_precs = sorted({*lat.keys(), *maps.keys()}, key=lambda p: PRECISION_ORDER.get(p, 99))
    rows: List[Dict[str,str]] = []
    for prec in all_precs:
        lrow = lat.get(prec, {'precision': prec})
        mrow = maps.get(prec, {})
        merged = {**lrow}
        for k in ACC_HEADERS:
            if k in mrow:
                merged[k] = f"{mrow[k]:.4f}" if mrow[k] else ''
            else:
                merged[k] = ''
        rows.append(merged)
    return rows


def latex_table(device: str, imgsz: int, rows: List[Dict[str,str]], caption: str, label: str, int8_cov: Dict[str,float]) -> str:
    # Determine columns: latency + accuracy
    cols = LAT_HEADERS + ACC_HEADERS
    labels = {
        'precision':'Precision','mean_ms':'Mean (ms)','p50_ms':'p50','p95_ms':'p95','fps_mean':'FPS','speedup_vs_fp32':'Speedup','map50':'mAP@0.5','map50_95':'mAP@0.5:0.95'
    }
    colspec = 'l' + 'r'*(len(cols)-1)
    lines = []
    lines.append('% Consolidated edge inference table (auto-generated)')
    lines.append('\\begin{table}[h]')
    lines.append('\\centering')
    lines.append('\\small')
    lines.append(f'\\begin{{tabular}}{{{colspec}}}')
    lines.append('\\toprule')
    lines.append(' & '.join(labels[c] for c in cols) + ' \\')
    lines.append('\\midrule')
    for r in rows:
        lines.append(' & '.join(r.get(c,'--') or '--' for c in cols) + ' \\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    cap_extra = ''
    if int8_cov:
        cap_extra = f" INT8 layer coverage: {int8_cov.get('int8_layers',0)}/{int8_cov.get('layers_total',0)} ({int8_cov.get('int8_ratio',0)*100:.1f}\\%)."
    lines.append(f"\\caption{{{caption} (Device: {device}, {imgsz}px).{cap_extra}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append('\\end{table}')
    lines.append('')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--latency-csv', nargs='*', default=[])
    ap.add_argument('--latency-json-glob', default='')
    ap.add_argument('--map-glob', default='')
    ap.add_argument('--int8-layers', default='')
    ap.add_argument('--device', default='Jetson Orin Nano')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--caption', default='YOLOv8n Edge Inference Summary')
    ap.add_argument('--label', default='tab:edge_infer')
    ap.add_argument('--out-tex', default='results/figures/edge_inference_report.tex')
    ap.add_argument('--out-json', default='results/figures/edge_inference_report.json')
    args = ap.parse_args()

    # Load latency
    latency = {}
    if args.latency_csv:
        latency = load_latency_csv([Path(p) for p in args.latency_csv], args.imgsz)
    elif args.latency_json_glob:
        latency = load_latency_from_json(args.latency_json_glob, args.imgsz)
    else:
        raise SystemExit('[e] Provide --latency-csv or --latency-json-glob')

    maps = load_map(args.map_glob, args.imgsz) if args.map_glob else {}
    int8_cov = load_int8_layers(args.int8_layers) if args.int8_layers else {}

    rows = build_rows(latency, maps)

    # JSON summary
    summary = {
        'device': args.device,
        'imgsz': args.imgsz,
        'rows': rows,
        'int8_coverage': int8_cov,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))

    # LaTeX
    tex_str = latex_table(args.device, args.imgsz, rows, args.caption, args.label, int8_cov)
    out_tex = Path(args.out_tex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(tex_str)

    print(f"[i] Wrote {out_json} and {out_tex}")

if __name__ == '__main__':  # pragma: no cover
    main()


#!/usr/bin/env python3
"""
Generate LaTeX macros and an optional small table for detector latency from
latency_comparative_<imgsz>.csv.

- Updates toolset/metrics_macros.tex with the appropriate macro based on --device:
  * jetson* -> \detInferMeanMsJetson
  * rpi*|rasp* -> \detInferMeanMsPi
- Writes an optional detector latency table (can be redirected per device).
"""
import argparse
import csv
from pathlib import Path


def load_rows(csv_path, imgsz, device_key):
    rows = []
    with csv_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                if int(row.get("imgsz", -1)) != imgsz:
                    continue
            except Exception:
                continue
            if row.get("device", "").lower() != str(device_key).lower():
                continue
            rows.append(row)
    return rows


def choose_row(rows):
    if not rows:
        return None
    for prec in ("fp16", "fp32", "int8", "ort", "ov", "ov_int8"):
        for r in rows:
            if r.get("precision", "").lower() == prec:
                return r

    def to_f(v):
        try:
            return float(v)
        except Exception:
            return 1e9

    return min(rows, key=lambda r: to_f(r.get("mean_ms", "1e9")))


def fmt_one_decimal(val):
    try:
        if val is None:
            return "--"
        return "{:.1f}".format(float(val))
    except Exception:
        return "--"


def macro_name_for_device(device_str):
    d = (device_str or "").lower()
    if d.startswith("jetson"):
        return "detInferMeanMsJetson"
    if d.startswith("rpi") or d.startswith("rasp"):
        return "detInferMeanMsPi"
    return "detInferMeanMsJetson"


def write_macros(macros_path, macro_name, value):
    macros_path.parent.mkdir(parents=True, exist_ok=True)
    content = []
    if macros_path.exists():
        try:
            content = macros_path.read_text().splitlines()
        except Exception:
            content = []
    needle = "\\renewcommand{\\%s}" % macro_name
    content = [ln for ln in content if not ln.strip().startswith(needle)]
    content.append("\\renewcommand{\\%s}{%s}" % (macro_name, value))
    macros_path.write_text("\n".join(content) + "\n")


def write_table(tex_path, rows, imgsz, macro_name, macro_value):
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "% Auto-generated detector latency (%s)\n" % macro_name
        + "% Consumed by validation.tex (macro updated below)\n"
        + "\\renewcommand{\\%s}{%s}\n" % (macro_name, macro_value)
        + "\\begin{table}[h]\n"
        "\\centering\n\\small\n"
        "\\begin{tabular}{lrrrr}\n"
        "\\toprule\n"
        "Precision & Mean (ms) & p50 (ms) & p95 (ms) & FPS \\\\ \n"
        "\\midrule\n"
    )
    lines = [header]
    for r in rows:
        prec = r.get("precision", "--")
        mean_ms = r.get("mean_ms", "") or "--"
        p50_ms = r.get("p50_ms", "") or "--"
        p95_ms = r.get("p95_ms", "") or "--"
        fps = r.get("fps_mean", "") or "--"
        lines.append(
            "%s & %s & %s & %s & %s \\\\ \n" % (prec, mean_ms, p50_ms, p95_ms, fps)
        )
    footer = (
        "\\bottomrule\n\\end{tabular}\n"
        "\\caption{Detector latency (input %dpx).}\n" % int(imgsz)
        + "\\label{tab:detector_latency_%s}\n" % macro_name.lower()
        + "\\end{table}\n"
    )
    lines.append(footer)
    tex_path.write_text("".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument(
        "--device", default="jetson", help="device key in CSV (e.g., jetson, rpi5)"
    )
    ap.add_argument(
        "--out-tex", default="toolset/figures/detector_latency_generated.tex"
    )
    ap.add_argument("--out-macros", default="toolset/metrics_macros.tex")
    ap.add_argument(
        "--no-table", action="store_true", help="only update macro; don't write a table"
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit("[e] CSV not found: %s" % csv_path)
    rows = load_rows(csv_path, args.imgsz, args.device)
    if not rows:
        raise SystemExit("[e] No rows for device/imgsz; run batch first")
    choice = choose_row(rows)
    try:
        p50 = float(choice.get("p50_ms", "") or "nan") if choice else None
    except Exception:
        p50 = None
    macro_value = fmt_one_decimal(p50)
    mname = macro_name_for_device(args.device)
    write_macros(Path(args.out_macros), mname, macro_value)
    if not args.no_table:
        write_table(Path(args.out_tex), rows, args.imgsz, mname, macro_value)
    print(
        "[i] Updated macro %s=%s; wrote table=%s"
        % (mname, macro_value, "no" if args.no_table else "yes")
    )


if __name__ == "__main__":
    main()

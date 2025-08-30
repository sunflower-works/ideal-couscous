"""Unified analysis script combining table build, run summaries, regression deltas,
optionally exploratory plots (baseline/quality/robustness/capacity) and JSON export.

Usage:
  python unified_analysis.py                 # full pipeline with plots (if plotly installed)
  python unified_analysis.py --no-plots      # skip plot generation
  python unified_analysis.py --just-table    # only build unified table

Environment overrides (optional):
  RESULTS_BASE_URL              Remote base URL to lazily fetch missing CSVs
  REGRESS_EMBED_MS_PCT          Embed latency regression threshold (pct increase)
  REGRESS_EXTRACT_MS_PCT        Extract latency regression threshold (pct increase)
  REGRESS_BER_ABS               BER absolute pct point increase threshold
  REGRESS_SUCCESS_PCT           Success rate negative pct point threshold

The script supports two directory layouts:
  (A) toolset/pi_suite/<group>/<group>.csv
  (B) toolset/pi_suite/<run_id>/<group>/<group>.csv (multiple chronological runs)

Outputs:
  toolset/unified/unified_table.(csv|parquet)
  toolset/parquet_all/<run>_<group>.parquet (if run layout)
  toolset/summary_latest.json (run history + latest deltas)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Additional optional group patterns beyond required_groups
EXTRA_GROUP_PATTERNS: List[Tuple[str, str]] = [
    ("robustness", "jpeg_sweep.csv"),
    ("robustness", "grid.csv"),
    ("robustness", "robustness.csv"),
    ("robustness", "attacks.csv"),  # new consolidated attacks file
    ("capacity", "capacity.csv"),
]

# Optional heavy plot libs
try:
    import plotly.express as px  # type: ignore
    import plotly.graph_objects as go  # type: ignore
except Exception:  # pragma: no cover
    px = go = None  # type: ignore

try:  # pretty display fallback
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover

    def display(obj):  # type: ignore
        if hasattr(obj, "to_string"):
            print(obj.to_string())
        else:
            print(obj)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    results_root: Path = Path("toolset")
    suite_name: str = "pi_suite"
    required_groups: Tuple[str, ...] = ("baseline", "quality", "runtime")
    remote_base_url: Optional[str] = os.getenv("RESULTS_BASE_URL")
    download_timeout: int = int(os.getenv("DOWNLOAD_TIMEOUT_SEC", "20"))
    regress_embed_ms_pct: float = float(os.getenv("REGRESS_EMBED_MS_PCT", "5"))
    regress_extract_ms_pct: float = float(os.getenv("REGRESS_EXTRACT_MS_PCT", "5"))
    regress_ber_abs: float = float(os.getenv("REGRESS_BER_ABS", "0.25"))
    regress_success_pct: float = float(os.getenv("REGRESS_SUCCESS_PCT", "2"))
    placeholder_sentinels: Tuple[float, ...] = (-1.0, 9999.0)
    parquet_dir: Path = Path("toolset") / "parquet_all"
    unified_out_dir: Path = Path("toolset") / "unified"
    unified_filename: str = "unified_table"
    summary_json: Path = Path("toolset") / "summary_latest.json"
    show_plots: bool = True
    host_platform: str = "unknown"


CFG = Config()
CFG.parquet_dir.mkdir(parents=True, exist_ok=True)
CFG.unified_out_dir.mkdir(parents=True, exist_ok=True)


def detect_host_platform() -> str:
    """Best-effort host platform tag for metrics: 'jetson', 'raspberrypi', or generic arch.
    Detection order:
      1. Jetson: presence of /etc/nv_tegra_release or 'NVIDIA Jetson' in /proc/device-tree/model
      2. Raspberry Pi: /proc/device-tree/model contains 'Raspberry Pi'
      3. Fallback: platform.machine() or uname summary.
    """
    try:
        dt_model = Path("/proc/device-tree/model")
        if dt_model.exists():
            model_txt = dt_model.read_text(errors="ignore").lower()
            if "jetson" in model_txt or Path("/etc/nv_tegra_release").exists():
                return "jetson"
            if "raspberry pi" in model_txt:
                return "raspberrypi"
        if Path("/etc/nv_tegra_release").exists():
            return "jetson"
    except Exception:
        pass
    # Fallback: arch + distro hint
    return platform.machine().lower()


CFG.host_platform = detect_host_platform()
print(f"[unified_analysis] Detected host platform: {CFG.host_platform}")

# ---------------------------------------------------------------------------
# CSV loading helpers (two-line schema format)
# ---------------------------------------------------------------------------
NON_NUMERIC_COLS = {
    "algorithm",
    "status",
    "fail_reason",
    "attack_chain",
    "attack_params_json",
    "tags",
    "git_commit",
    "image_id",
    "image_category",
    "attacked_status",
    "attacked_fail_reason",
    "crc_status",
    "blur_bin",
    "source",
    "__source_file",
    "run",
    "group",
    "host_platform",
    # new non-numeric columns
    "algo_variant",
    "blockchain",
    "gpu_backend",
    "gpu_name",
}
CSV_NAME_PATTERN = re.compile(r"^[a-z0-9_]+\.csv$", re.I)


def _http_fetch(url: str, timeout: int) -> Optional[bytes]:
    try:  # pragma: no cover (network optional)
        import urllib.request

        with urllib.request.urlopen(url, timeout=timeout) as r:  # type: ignore
            if getattr(r, "status", 200) == 200:
                return r.read()
    except Exception:
        return None
    return None


def ensure_file(local_path: Path, suite_rel: str):
    if local_path.exists() or not CFG.remote_base_url:
        return
    url = CFG.remote_base_url.rstrip("/") + "/" + suite_rel.lstrip("/")
    data = _http_fetch(url, CFG.download_timeout)
    if data:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)


def load_schema_csv(path: Path) -> pd.DataFrame:
    with path.open() as f:
        _schema = f.readline()
        header = f.readline().strip().split(",")
    df = pd.read_csv(path, comment="#", names=header, skiprows=2)
    return df


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c in NON_NUMERIC_COLS:
            continue
        if df[c].dtype == object:
            try:
                df[c] = pd.to_numeric(df[c])
            except Exception:
                pass
        if np.issubdtype(df[c].dtype, np.number):
            df[c] = df[c].replace(list(CFG.placeholder_sentinels), np.nan)
    return df


# ---------------------------------------------------------------------------
# Run discovery (supports flat or multi-run layout)
# ---------------------------------------------------------------------------


def discover_layout() -> str:
    root = CFG.results_root / CFG.suite_name
    if not root.exists():
        return "none"
    # Multi-run if subdirectories containing one of group csvs
    run_dirs = [
        p
        for p in root.iterdir()
        if p.is_dir()
        and any((p / g / f"{g}.csv").exists() for g in CFG.required_groups)
    ]
    if run_dirs:
        return "multi"
    # Flat if root/<group>/<group>.csv
    flat_ok = any((root / g / f"{g}.csv").exists() for g in CFG.required_groups)
    if flat_ok:
        return "flat"
    return "none"


def load_groups_for_run(run_path: Path, run_name: str) -> Dict[str, pd.DataFrame]:
    groups: Dict[str, pd.DataFrame] = {}
    # Required groups first
    for g in CFG.required_groups:
        csv_path = run_path / g / f"{g}.csv"
        ensure_file(csv_path, f"{CFG.suite_name}/{run_name}/{g}/{g}.csv")
        if csv_path.exists():
            df = coerce_numeric(load_schema_csv(csv_path))
        else:
            df = pd.DataFrame()
        groups[g] = df
    # Extra known patterns
    for sub, fname in EXTRA_GROUP_PATTERNS:
        csv_path = run_path / sub / fname
        ensure_file(csv_path, f"{CFG.suite_name}/{run_name}/{sub}/{fname}")
        key = sub  # treat subdir as group label
        if csv_path.exists():
            try:
                df = coerce_numeric(load_schema_csv(csv_path))
                # If group already exists (e.g., robustness multi files), concatenate
                if key in groups and not groups[key].empty:
                    groups[key] = pd.concat(
                        [groups[key], df], ignore_index=True, sort=False
                    )
                else:
                    groups[key] = df
            except Exception:
                pass
    # Any other two-level CSV not already captured
    for csv in run_path.rglob("*.csv"):
        # depth filter: skip deeply nested beyond 3 components
        rel = csv.relative_to(run_path)
        parts = rel.parts
        if len(parts) > 3:
            continue
        parent = parts[0]
        if parent in groups:  # already handled
            continue
        if csv.is_file():
            try:
                df = coerce_numeric(load_schema_csv(csv))
                groups[parent] = df
            except Exception:
                continue
    return groups


def load_all_runs() -> Dict[str, Dict[str, pd.DataFrame]]:
    layout = discover_layout()
    runs: Dict[str, Dict[str, pd.DataFrame]] = {}
    root = CFG.results_root / CFG.suite_name
    if layout == "multi":
        candidates = sorted(
            [p for p in root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime
        )
        for run_dir in candidates:
            groups = load_groups_for_run(run_dir, run_dir.name)
            if any(not df.empty for df in groups.values()):
                runs[run_dir.name] = groups
    elif layout == "flat":
        groups = load_groups_for_run(root, "__flat__")
        runs["__flat__"] = groups
    return runs


# ---------------------------------------------------------------------------
# Unified table (merge all CSV sources, derive fields)
# ---------------------------------------------------------------------------
ORDERED_COLUMNS: List[str] = [
    "run_uuid",
    "timestamp",
    "git_commit",
    "host_platform",
    "algorithm",
    "round_idx",
    "status",
    "fail_reason",
    "payload_bits_requested",
    "payload_bits_embedded",
    "payload_bits_recovered",
    "ber_percent",
    "ber_bits_wrong",
    "bits_total",
    "success_flag",
    "psnr_db",
    "ssim",
    "embed_ms",
    "extract_ms",
    "total_ms",
    "cpu_pct_mean",
    "cpu_pct_max",
    "rss_max_mib",
    "size_cover_bytes",
    "size_stego_bytes",
    "size_delta_bytes",
    "size_delta_pct",
    "attack_chain",
    "attack_params_json",
    "tags",
    "core_count",
    "openblas_threads",
    "temp_c_max",
    "freq_min_khz",
    "capacity_bits_est",
    "watermark_strength",
    "robustness_score",
    "imperceptibility_norm",
    "imperceptibility_combined",
    "capacity_clamped_row",
    "silent_trunc_flag",
    "crc_status",
    "attacked_status",
    "attacked_fail_reason",
    "attacked_payload_bits_recovered",
    "attacked_ber_percent",
    "attacked_ber_bits_wrong",
    "attacked_bits_total",
    "attacked_success_flag",
    "psnr_attack_db",
    "ssim_attack",
    "attack_delta_psnr",
    "attack_delta_ssim",
    "image_id",
    "image_category",
    "image_blur_metric",
    "image_face_detected",
    # anchoring / gpu base
    "anchoring_fee_usd",
    "digest_bytes",
    "tx_confirm_s",
    "gpu_flag",
    "algo_variant",
    "blockchain",
    # extended gpu / detector
    "gpu_backend",
    "gpu_device_count",
    "gpu_name",
    "detector_score",
    "detector_flag",
]

BOOLEAN_DEFAULTS = {
    "silent_trunc_flag",
    "capacity_clamped_row",
    "success_flag",
    "attacked_success_flag",
}


def derive_fields(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.copy()
    if {"embed_ms", "extract_ms"}.issubset(wide.columns):
        wide["total_ms"] = wide[["embed_ms", "extract_ms"]].sum(axis=1, min_count=1)
    else:
        wide["total_ms"] = np.nan
    if "capacity_bits_est" not in wide or wide["capacity_bits_est"].isna().all():
        if "payload_bits_embedded" in wide:
            wide["capacity_bits_est"] = wide.get("payload_bits_embedded")
    if "robustness_score" not in wide or wide["robustness_score"].isna().all():
        if "ber_percent" in wide:
            wide["robustness_score"] = 1.0 - (wide["ber_percent"] / 100.0)
    if (
        "imperceptibility_norm" not in wide
        or wide["imperceptibility_norm"].isna().all()
    ):
        if "psnr_db" in wide:
            wide["imperceptibility_norm"] = (wide["psnr_db"] / 50.0).clip(upper=1.0)
    if (
        "imperceptibility_combined" not in wide
        or wide["imperceptibility_combined"].isna().all()
    ):
        if {"imperceptibility_norm", "ssim"}.issubset(wide.columns):
            wide["imperceptibility_combined"] = 0.5 * (
                wide["imperceptibility_norm"] + wide["ssim"].fillna(0)
            )
    if {"payload_bits_embedded", "payload_bits_recovered"}.issubset(wide.columns):
        wide["silent_trunc_flag"] = (
            wide["payload_bits_recovered"].notna()
            & wide["payload_bits_embedded"].notna()
            & (wide["payload_bits_recovered"] < wide["payload_bits_embedded"])
        )
    if "capacity_clamped_row" not in wide:
        wide["capacity_clamped_row"] = False
    if {"psnr_db", "psnr_attack_db"}.issubset(wide.columns):
        wide["attack_delta_psnr"] = wide["psnr_attack_db"] - wide["psnr_db"]
    if {"ssim", "ssim_attack"}.issubset(wide.columns):
        wide["attack_delta_ssim"] = wide["ssim_attack"] - wide["ssim"]
    return wide


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ORDERED_COLUMNS:
        if col not in df.columns:
            if col in BOOLEAN_DEFAULTS:
                df[col] = False
            else:
                df[col] = np.nan
    return df[ORDERED_COLUMNS]


# ---------------------------------------------------------------------------
# Run aggregation & regressions
# ---------------------------------------------------------------------------
AGG_COLUMNS = dict(
    rounds=("round_idx", "count"),
    success_rate=("success_flag", "mean"),
    embed_ms_mean=("embed_ms", "mean"),
    extract_ms_mean=("extract_ms", "mean"),
    embed_ms_p95=("embed_ms", lambda s: s.quantile(0.95)),
    extract_ms_p95=("extract_ms", lambda s: s.quantile(0.95)),
    ber_mean=("ber_percent", "mean"),
    robustness_mean=("robustness_score", "mean"),
    strength_mean=("watermark_strength", "mean"),
    cpu_mean=("cpu_pct_mean", "mean"),
    rss_max=("rss_max_mib", "max"),
    anchoring_fee_mean=("anchoring_fee_usd", "mean"),
    tx_confirm_s_mean=("tx_confirm_s", "mean"),
    gpu_usage_rate=("gpu_flag", "mean"),
    detection_rate=("detector_flag", "mean"),
)


def summarize_run(group_frames: Dict[str, pd.DataFrame], run_name: str) -> pd.DataFrame:
    rows = []
    for group, df in group_frames.items():
        if df.empty or "algorithm" not in df:
            continue
        agg = df.groupby("algorithm").agg(**AGG_COLUMNS).reset_index()
        agg.insert(0, "group", group)
        agg.insert(0, "run", run_name)
        rows.append(agg)
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()


def compute_deltas(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history
    history = history.copy()
    # run ordering by first appearance
    history["run_order"] = pd.factorize(history["run"])[0]
    out = []
    for alg, sub in history.groupby("algorithm"):
        sub = sub.sort_values("run_order")
        prev = None
        for _, row in sub.iterrows():
            r = row.to_dict()
            if prev is not None:
                for metric in [
                    "embed_ms_mean",
                    "extract_ms_mean",
                    "ber_mean",
                    "success_rate",
                    "robustness_mean",
                    "cpu_mean",
                    "rss_max",
                ]:
                    if (
                        pd.notna(row.get(metric))
                        and pd.notna(prev.get(metric))
                        and prev.get(metric) != 0
                    ):
                        r[f"{metric}_delta_pct"] = (
                            100.0 * (row[metric] - prev[metric]) / prev[metric]
                        )
            out.append(r)
            prev = row
    return pd.DataFrame(out)


def flag_regressions(deltas: pd.DataFrame) -> pd.DataFrame:
    if deltas.empty:
        return deltas

    def reg(row):
        flags = []
        if (
            pd.notna(row.get("embed_ms_mean_delta_pct"))
            and row["embed_ms_mean_delta_pct"] > CFG.regress_embed_ms_pct
        ):
            flags.append("embed_ms↑")
        if (
            pd.notna(row.get("extract_ms_mean_delta_pct"))
            and row["extract_ms_mean_delta_pct"] > CFG.regress_extract_ms_pct
        ):
            flags.append("extract_ms↑")
        if (
            pd.notna(row.get("ber_mean_delta_pct"))
            and row["ber_mean_delta_pct"] > CFG.regress_ber_abs
        ):
            flags.append("ber↑")
        if (
            pd.notna(row.get("success_rate_delta_pct"))
            and row["success_rate_delta_pct"] < -CFG.regress_success_pct
        ):
            flags.append("success↓")
        return ",".join(flags)

    deltas = deltas.copy()
    deltas["regress_flags"] = deltas.apply(reg, axis=1)
    return deltas


# ---------------------------------------------------------------------------
# Plotting (optional)  -- each section guards missing columns
# ---------------------------------------------------------------------------


def generate_plots(latest_groups: Dict[str, pd.DataFrame]):  # side effects only
    if not px:
        print("Plotly not available; skipping plots.")
        return
    baseline_df = latest_groups.get("baseline", pd.DataFrame())
    quality_df = latest_groups.get("quality", pd.DataFrame())
    runtime_df = latest_groups.get("runtime", pd.DataFrame())
    attacks_df = pd.DataFrame()
    # Combine any robustness entries that look like attacks (attack_chain column present)
    robustness_df = latest_groups.get("robustness", pd.DataFrame())
    if not robustness_df.empty and "attack_chain" in robustness_df.columns:
        # Deduplicate combined robustness sources (jpeg_sweep + attacks) if both present
        attacks_df = robustness_df.drop_duplicates(
            subset=["run_uuid", "algorithm", "attack_chain"]
        )
    # ----------------- Optional CPU/RSS backfill -----------------
    if not baseline_df.empty and {"cpu_pct_mean", "rss_max_mib"}.issubset(
        baseline_df.columns
    ):
        if (
            baseline_df["cpu_pct_mean"].isna().all()
            or baseline_df["cpu_pct_mean"].eq("").all()
        ):
            try:
                import psutil  # type: ignore

                p = psutil.Process()
                cpu = p.cpu_percent(interval=0.1)  # short sample
                rss = p.memory_info().rss / (1024 * 1024)
                baseline_df["cpu_pct_mean"] = cpu
                baseline_df["rss_max_mib"] = rss
            except Exception:
                pass
    # ----------------- Existing baseline plots -----------------
    # --- Baseline performance distributions ---
    if not baseline_df.empty:
        for metric in ["embed_ms", "extract_ms"]:
            if metric in baseline_df.columns:
                fig = px.violin(
                    baseline_df,
                    x="algorithm",
                    y=metric,
                    box=True,
                    points="suspectedoutliers",
                    title=f"{metric} distribution",
                )
                fig.show()
    # Throughput
    if not baseline_df.empty and {"payload_bits_embedded", "embed_ms"}.issubset(
        baseline_df.columns
    ):
        tmp = baseline_df.copy()
        tmp["embed_bps"] = (
            tmp["payload_bits_embedded"] / (tmp["embed_ms"] / 1000.0)
        ).replace([np.inf, -np.inf], np.nan)
        fig = px.box(
            tmp,
            x="algorithm",
            y="embed_bps",
            points="outliers",
            title="Embedding Throughput (bits/s)",
        )
        fig.show()
    # BER distribution
    if not baseline_df.empty and "ber_percent" in baseline_df.columns:
        fig = px.histogram(
            baseline_df,
            x="ber_percent",
            color="algorithm",
            nbins=50,
            title="BER Percent Distribution",
        )
        fig.show()
        if "robustness_score" in baseline_df.columns:
            fig2 = px.histogram(
                baseline_df,
                x="robustness_score",
                color="algorithm",
                nbins=50,
                title="Robustness Score Distribution",
            )
            fig2.show()
    # Strength vs embed time
    if not baseline_df.empty and {"watermark_strength", "embed_ms"}.issubset(
        baseline_df.columns
    ):
        fig = px.scatter(
            baseline_df,
            x="watermark_strength",
            y="embed_ms",
            color="algorithm",
            opacity=0.5,
            trendline="lowess",
            title="Strength vs Embed Time",
        )
        fig.show()
    # Blur impact
    if not baseline_df.empty and {"image_blur_metric", "embed_ms"}.issubset(
        baseline_df.columns
    ):
        fig = px.scatter(
            baseline_df,
            x="image_blur_metric",
            y="embed_ms",
            color="algorithm",
            opacity=0.4,
            trendline="lowess",
            title="Blur vs Embed Time",
        )
        fig.show()
        if "robustness_score" in baseline_df.columns:
            fig2 = px.scatter(
                baseline_df,
                x="image_blur_metric",
                y="robustness_score",
                color="algorithm",
                opacity=0.4,
                trendline="lowess",
                title="Blur vs Robustness",
            )
            fig2.show()
    # CDF embed/extract baseline vs runtime
    if not baseline_df.empty or not runtime_df.empty:
        cdf_source = pd.concat(
            [
                (
                    baseline_df.assign(source="baseline")
                    if not baseline_df.empty
                    else pd.DataFrame()
                ),
                (
                    runtime_df.assign(source="runtime")
                    if not runtime_df.empty
                    else pd.DataFrame()
                ),
            ],
            ignore_index=True,
        )
        for metric in ["embed_ms", "extract_ms"]:
            if metric in cdf_source.columns and not cdf_source.empty:
                tmp = (
                    cdf_source[["algorithm", "source", metric]]
                    .dropna()
                    .sort_values(metric)
                )
                if tmp.empty:
                    continue
                tmp["cdf"] = tmp.groupby(["algorithm", "source"]).cumcount() / (
                    tmp.groupby(["algorithm", "source"])[metric].transform("count") - 1
                )
                fig = px.line(
                    tmp,
                    x=metric,
                    y="cdf",
                    color="algorithm",
                    line_dash="source",
                    title=f"CDF {metric}",
                )
                fig.update_layout(yaxis_title="CDF", xaxis_title=f"{metric} (ms)")
                fig.show()
    # Tail latency table quick print
    if not runtime_df.empty and "embed_ms" in runtime_df.columns:
        tail = (
            runtime_df.groupby("algorithm")["embed_ms"]
            .quantile([0.5, 0.9, 0.95, 0.99])
            .unstack()
            .rename(columns={0.5: "p50", 0.9: "p90", 0.95: "p95", 0.99: "p99"})
        )
        print("Tail latency (embed ms):")
        display(tail)
    # Radar composite (baseline)
    if not baseline_df.empty and "algorithm" in baseline_df:
        dims = {
            "speed": lambda d: (
                1.0 / max(d["embed_ms"].mean(), 1e-6) if "embed_ms" in d else np.nan
            ),
            "robustness": lambda d: (
                d["robustness_score"].mean() if "robustness_score" in d else np.nan
            ),
            "success": lambda d: (
                d["success_flag"].mean() if "success_flag" in d else np.nan
            ),
            "strength": lambda d: (
                d["watermark_strength"].mean() if "watermark_strength" in d else np.nan
            ),
        }
        rows = []
        for alg, sub in baseline_df.groupby("algorithm"):
            row = {"algorithm": alg}
            for k, fn in dims.items():
                try:
                    row[k] = float(fn(sub))
                except Exception:
                    row[k] = np.nan
            rows.append(row)
        radar_df = pd.DataFrame(rows)
        if not radar_df.empty:
            norm_cols = [c for c in radar_df.columns if c != "algorithm"]
            norm = radar_df.copy()
            for c in norm_cols:
                vals = norm[c].astype(float)
                mn, mx = vals.min(), vals.max()
                if np.isfinite(vals).sum() == 0:
                    norm[c] = 0.0
                elif math.isclose(mn, mx):
                    norm[c] = 1.0
                else:
                    norm[c] = (vals - mn) / (mx - mn)
            categories = norm_cols + [norm_cols[0]] if norm_cols else []
            fig = go.Figure()
            algs_sorted = list(norm["algorithm"])
            nalg = len(algs_sorted)
            for idx, alg in enumerate(algs_sorted):
                vals = norm[norm.algorithm == alg][norm_cols].values.flatten().tolist()
                if not vals:
                    continue
                vals.append(vals[0])
                opacity = 0.3 + (0.6 * idx / max(1, nalg - 1))  # gradient opacity
                fig.add_trace(
                    go.Scatterpolar(
                        r=vals,
                        theta=categories,
                        fill="toself",
                        name=alg,
                        opacity=opacity,
                    )
                )
            fig.update_layout(
                title="Normalized Composite Radar",
                polar=dict(radialaxis=dict(range=[0, 1])),
            )
            fig.show()
    # ----------------- Robustness attack breakdown plots -----------------
    if not attacks_df.empty:
        # Parse attack type and param
        atk = attacks_df.copy()
        atk["attack_type"] = "other"
        atk["attack_param"] = np.nan
        # JPEG
        m = atk["attack_chain"].str.extract(r"jpeg:quality=([0-9]+)")
        atk.loc[m[0].notna(), "attack_type"] = "jpeg"
        atk.loc[m[0].notna(), "attack_param"] = m[0].astype(float)
        # Noise
        m = atk["attack_chain"].str.extract(r"noise:gauss=([0-9]+)")
        atk.loc[m[0].notna(), "attack_type"] = "noise"
        atk.loc[m[0].notna(), "attack_param"] = m[0].astype(float)
        # Scale
        m = atk["attack_chain"].str.extract(r"geom:scale=([0-9.]+)")
        atk.loc[m[0].notna(), "attack_type"] = "scale"
        atk.loc[m[0].notna(), "attack_param"] = m[0].astype(float)
        # Rotate
        m = atk["attack_chain"].str.extract(r"geom:rotate=([-0-9]+)")
        atk.loc[m[0].notna(), "attack_type"] = "rotate"
        atk.loc[m[0].notna(), "attack_param"] = m[0].astype(float)
        # Choose BER column
        ber_col = "ber_percent"
        # Aggregate mean BER
        agg = (
            atk.dropna(subset=["attack_param"])
            .groupby(["algorithm", "attack_type", "attack_param"])[ber_col]
            .mean()
            .reset_index()
        )
        # Plot per attack type
        plotted_types = []
        for atype in ["jpeg", "noise", "scale", "rotate"]:
            sub = agg[agg.attack_type == atype]
            if sub.empty:
                continue
            plotted_types.append(atype)
            fig = px.line(
                sub.sort_values("attack_param"),
                x="attack_param",
                y=ber_col,
                color="algorithm",
                markers=True,
                title=f"Mean BER vs {atype} parameter",
            )
            fig.update_layout(xaxis_title=f"{atype} param", yaxis_title="Mean BER (%)")
            fig.show()
        # Combined facet plot (if multiple attack types present)
        if len(plotted_types) > 1:
            facet_agg = agg[agg.attack_type.isin(plotted_types)].copy()
            facet_fig = px.line(
                facet_agg.sort_values(["attack_type", "attack_param"]),
                x="attack_param",
                y=ber_col,
                color="algorithm",
                markers=True,
                facet_col="attack_type",
                facet_col_wrap=2,
                title="Mean BER vs Attack Parameter (Faceted by Attack Type)",
            )
            facet_fig.update_yaxes(matches=None)
            facet_fig.update_xaxes(title="parameter")
            facet_fig.update_layout(yaxis_title="Mean BER (%)")
            facet_fig.show()


# ---------------------------------------------------------------------------
# CLI argument parsing (re-added after edits)
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Unified watermark analysis pipeline")
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation even if plotly installed",
    )
    p.add_argument(
        "--just-table",
        action="store_true",
        help="Only build unified table; skip history/regressions/plots",
    )
    p.add_argument(
        "--purge-dummy",
        action="store_true",
        help="Delete dummy smoke-test CSVs (only if detected as safe).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pipeline entry
# ---------------------------------------------------------------------------


def build_unified_table(all_runs: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for run_name, groups in all_runs.items():
        for group, df in groups.items():
            if df.empty:
                continue
            df2 = df.copy()
            df2["run"] = run_name
            df2["group"] = group
            frames.append(df2)
    if not frames:
        return pd.DataFrame()
    base = pd.concat(frames, ignore_index=True, sort=False)
    # Attach host_platform (override only if missing)
    if "host_platform" not in base.columns:
        base["host_platform"] = CFG.host_platform
    else:
        base["host_platform"] = base["host_platform"].fillna(CFG.host_platform)
    base = derive_fields(base)
    base = ensure_columns(base)
    # Fill boolean defaults
    for c in BOOLEAN_DEFAULTS:
        if c in base:
            base[c] = base[c].fillna(False)
    # Persist
    out_csv = CFG.unified_out_dir / f"{CFG.unified_filename}.csv"
    out_parquet = CFG.unified_out_dir / f"{CFG.unified_filename}.parquet"
    base.to_csv(out_csv, index=False)
    try:
        base.to_parquet(out_parquet, index=False)
    except Exception as e:  # parquet optional
        print(f"[unified_analysis] Parquet write skipped: {e}")
    print(f"Unified table rows: {len(base)} -> {out_csv.name}")
    return base


def summarize_history(
    all_runs: Dict[str, Dict[str, pd.DataFrame]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: List[pd.DataFrame] = []
    for run_name, groups in all_runs.items():
        run_summary = summarize_run(groups, run_name)
        if not run_summary.empty:
            summaries.append(run_summary)
        # Persist per-run per-group parquet
        for g, df in groups.items():
            if df.empty:
                continue
            (CFG.parquet_dir / f"{run_name}_{g}.parquet").parent.mkdir(
                parents=True, exist_ok=True
            )
            try:
                df.to_parquet(CFG.parquet_dir / f"{run_name}_{g}.parquet", index=False)
            except Exception as e:
                print(f"[unified_analysis] Skip per-run parquet {run_name}_{g}: {e}")
    if not summaries:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    history = pd.concat(summaries, ignore_index=True)
    deltas = flag_regressions(compute_deltas(history))
    latest_run = sorted(all_runs.keys())[-1]
    latest_groups = all_runs[latest_run]
    return history, deltas, pd.DataFrame({"latest_run": [latest_run]})


def write_summary_json(history: pd.DataFrame, deltas: pd.DataFrame, latest_run: str):
    payload = {
        "config": asdict(CFG),
        "latest_run": latest_run,
        "generated_at": time.time(),
        "history_rows": history.to_dict(orient="records"),
        "latest_deltas": deltas[deltas.run == latest_run].to_dict(orient="records"),
    }
    CFG.summary_json.parent.mkdir(parents=True, exist_ok=True)
    CFG.summary_json.write_text(json.dumps(payload, indent=2))
    print("Wrote summary JSON ->", CFG.summary_json)


def _fmt_num(val: float, decimals: int = 0) -> str:
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return "NA"
    fmt = f"{{:.{decimals}f}}"
    s = fmt.format(val)
    # Strip trailing .0 for integers when decimals=0
    if decimals == 0:
        s = s.rstrip("0").rstrip(".")
    return s


JPEG_QUALITIES_STANDARD = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def write_metrics_macros(latest_groups: Dict[str, pd.DataFrame]):
    out_path = Path("toolset/metrics_macros.tex")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = ["% Auto-generated metrics macros (do not edit manually)"]
    baseline_df = latest_groups.get("baseline", pd.DataFrame())
    quality_df = latest_groups.get("quality", pd.DataFrame())
    robustness_df = latest_groups.get("robustness", pd.DataFrame())
    # Latency metrics (baseline)
    if (
        not baseline_df.empty
        and "embed_ms" in baseline_df
        and "extract_ms" in baseline_df
    ):
        embed_mean = baseline_df["embed_ms"].mean()
        embed_sd = baseline_df["embed_ms"].std()
        embed_median = baseline_df["embed_ms"].median()
        embed_p95 = baseline_df["embed_ms"].quantile(0.95)
        extract_mean = baseline_df["extract_ms"].mean()
        extract_sd = baseline_df["extract_ms"].std()
        extract_median = baseline_df["extract_ms"].median()
        extract_p95 = baseline_df["extract_ms"].quantile(0.95)
        lines += [
            f"\\renewcommand{{\\embedMeanMs}}{{{_fmt_num(embed_mean)}}}",
            f"\\renewcommand{{\\embedSdMs}}{{{_fmt_num(embed_sd)}}}",
            f"\\renewcommand{{\\embedMedianMs}}{{{_fmt_num(embed_median)}}}",
            f"\\renewcommand{{\\embedPninetyFiveMs}}{{{_fmt_num(embed_p95)}}}",
            f"\\renewcommand{{\\extractMeanMs}}{{{_fmt_num(extract_mean)}}}",
            f"\\renewcommand{{\\extractSdMs}}{{{_fmt_num(extract_sd)}}}",
            f"\\renewcommand{{\\extractMedianMs}}{{{_fmt_num(extract_median)}}}",
            f"\\renewcommand{{\\extractPninetyFiveMs}}{{{_fmt_num(extract_p95)}}}",
        ]
    # Detector latency macros (if detector_infer_ms present)
    if not baseline_df.empty and "detector_infer_ms" in baseline_df.columns:
        jet = baseline_df[
            (baseline_df.get("host_platform") == "jetson")
            & baseline_df["detector_infer_ms"].notna()
        ]["detector_infer_ms"]
        rpi = baseline_df[
            (baseline_df.get("host_platform") == "raspberrypi")
            & baseline_df["detector_infer_ms"].notna()
        ]["detector_infer_ms"]
        det_mean_jetson = (
            jet.mean() if not jet.empty else baseline_df["detector_infer_ms"].mean()
        )
        det_mean_pi = (
            rpi.mean() if not rpi.empty else baseline_df["detector_infer_ms"].mean()
        )
        lines += [
            f"\\renewcommand{{\\detInferMeanMsJetson}}{{{_fmt_num(det_mean_jetson)}}}",
            f"\\renewcommand{{\\detInferMeanMsPi}}{{{_fmt_num(det_mean_pi)}}}",
        ]
    # Imperceptibility metrics (quality)
    if not quality_df.empty:
        if "psnr_db" in quality_df:
            lines.append(
                f"\\renewcommand{{\\meanPSNR}}{{{_fmt_num(quality_df['psnr_db'].mean(),1)}}}"
            )
        if "ssim" in quality_df:
            lines.append(
                f"\\renewcommand{{\\meanSSIM}}{{{_fmt_num(quality_df['ssim'].mean(),3)}}}"
            )
    # Robustness JPEG accuracy
    if not robustness_df.empty:
        df = robustness_df.copy()
        if (
            "attacked_ber_percent" in df.columns
            and not df["attacked_ber_percent"].isna().all()
        ):
            ber_col = "attacked_ber_percent"
        else:
            ber_col = "ber_percent" if "ber_percent" in df.columns else None
        if ber_col:
            if "attack_chain" in df.columns:
                df["jpeg_quality"] = (
                    df["attack_chain"]
                    .str.extract(r"jpeg:quality=([0-9]+)")
                    .astype(float)
                )
            if "jpeg_quality" in df.columns:
                acc = 100.0 - df.groupby("jpeg_quality")[ber_col].mean()
                for q in JPEG_QUALITIES_STANDARD:
                    val = _fmt_num(acc.loc[q], 0) if q in acc.index else "NA"
                    macro_name = {
                        10: "accJPEGten",
                        20: "accJPEGtwenty",
                        30: "accJPEGthirty",
                        40: "accJPEGforty",
                        50: "accJPEGfifty",
                        60: "accJPEGsixty",
                        70: "accJPEGseventy",
                        80: "accJPEGeighty",
                        90: "accJPEGninety",
                        100: "accJPEGhundred",
                    }[q]
                    lines.append(f"\\renewcommand{{\\{macro_name}}}{{{val}}}")
                if 80 in acc.index:  # back-compat alias
                    lines.append(
                        f"\\renewcommand{{\\accJEPEighty}}{{{_fmt_num(acc.loc[80],0)}}}"
                    )
    # Anchoring sweet spot (batch size vs fee)
    for gname, gdf in latest_groups.items():
        if {"batch_size", "anchoring_fee_usd"}.issubset(gdf.columns) and not gdf.empty:
            fee_agg = gdf.groupby("batch_size")["anchoring_fee_usd"].mean()
            if not fee_agg.empty:
                sweet_batch = int(fee_agg.idxmin())
                lines.append(f"\\renewcommand{{\\batchsweetspot}}{{{sweet_batch}}}")
            break
    lines.append(f"\\renewcommand{{\\hostplatform}}{{{CFG.host_platform}}}")
    out_path.write_text("\n".join(lines) + "\n")
    print("Wrote metrics macros ->", out_path)


def generate_static_tikz(latest_groups: Dict[str, pd.DataFrame]):
    fig_dir = Path("toolset/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    baseline_df = latest_groups.get("baseline", pd.DataFrame())
    if not baseline_df.empty and {"embed_ms", "extract_ms"}.issubset(
        baseline_df.columns
    ):
        lat_path = fig_dir / "latency_table_generated.tex"
        emb_mean = baseline_df["embed_ms"].mean()
        emb_sd = baseline_df["embed_ms"].std()
        ext_mean = baseline_df["extract_ms"].mean()
        ext_sd = baseline_df["extract_ms"].std()
        lat_table = rf"""% Auto-generated latency table
\begin{{table}}[ht]
\centering
\caption{{Latency (mean $\pm$ SD).}}
\label{{tab:latency}}
\begin{{tabular}}{{|l|c|c|}}
\hline
\textbf{{Stage}} & \textbf{{Measured (ms)}} & \textbf{{Target (ms)}} \\ \hline
Embed (Edge) & {emb_mean:.0f}({emb_sd:.0f}) & 250 \\ \hline
Extract (Edge) & {ext_mean:.0f}({ext_sd:.0f}) & 300 \\ \hline
\end{{tabular}}
\end{{table}}
"""
        lat_path.write_text(lat_table)
        print("Wrote generated latency table ->", lat_path)
    # Detector latency table
    if not baseline_df.empty and "detector_infer_ms" in baseline_df.columns:
        det_path = fig_dir / "detector_latency_generated.tex"

        def _plat_stats(tag: str) -> Tuple[float, float]:
            sub = baseline_df[
                (baseline_df.get("host_platform") == tag)
                & baseline_df["detector_infer_ms"].notna()
            ]["detector_infer_ms"]
            if not sub.empty:
                return float(sub.mean()), float(sub.std())
            return float(baseline_df["detector_infer_ms"].mean()), float(
                baseline_df["detector_infer_ms"].std()
            )

        mean_jet, sd_jet = _plat_stats("jetson")
        mean_pi, sd_pi = _plat_stats("raspberrypi")
        det_table = rf"""% Auto-generated detector latency table
\begin{{table}}[ht]
\centering
\caption{{Detector latency (mean $\pm$ SD).}}
\label{{tab:detector-latency}}
\begin{{tabular}}{{|l|c|}}
\hline
\textbf{{Platform}} & \textbf{{YOLOv8n (ms)}} \\ \hline
Jetson Orin Nano & {mean_jet:.0f}({sd_jet:.0f}) \\ \hline
Raspberry Pi 5   & {mean_pi:.0f}({sd_pi:.0f}) \\ \hline
\end{{tabular}}
\end{{table}}
"""
        det_path.write_text(det_table)
        print("Wrote generated detector latency table ->", det_path)
    # Existing robustness and anchoring plots
    robustness_df = latest_groups.get("robustness", pd.DataFrame())
    if not robustness_df.empty:
        df = robustness_df.copy()
        if "attack_chain" in df.columns:
            df["jpeg_quality"] = (
                df["attack_chain"].str.extract(r"jpeg:quality=([0-9]+)").astype(float)
            )
        if "jpeg_quality" in df.columns:
            if (
                "attacked_ber_percent" in df.columns
                and not df["attacked_ber_percent"].isna().all()
            ):
                ber_col = "attacked_ber_percent"
            else:
                ber_col = "ber_percent" if "ber_percent" in df.columns else None
            if ber_col:
                acc = 100.0 - df.groupby("jpeg_quality")[ber_col].mean().sort_index()
                coords = " ".join(
                    f"({int(q)},{acc.loc[q]:.0f})"
                    for q in acc.index
                    if not math.isnan(acc.loc[q])
                )
                tikz_path = fig_dir / "accuracy_jpeg_generated.tikz"
                tikz_content = rf"""% Auto-generated accuracy vs JPEG plot
\begin{{tikzpicture}}
  \begin{{axis}}[
    width=\linewidth,height=6cm,
    xlabel={{JPEG quality (\%)}},ylabel={{Accuracy (\%)}},
    ymin=80,ymax=100,xmin=10,xmax=100,
    ymajorgrids,xmajorgrids,grid style={{dashed,gray!30}},
  ]
  \addplot+[mark=o,thick] coordinates {{ {coords} }};
  \end{{axis}}
\end{{tikzpicture}}
"""
                tikz_path.write_text(tikz_content)
                print("Wrote generated JPEG accuracy TikZ ->", tikz_path)
    anchoring_cols = ["batch_size", "anchoring_fee_usd"]
    for gname, df in latest_groups.items():
        if set(anchoring_cols).issubset(df.columns):
            agg = df.groupby("batch_size")["anchoring_fee_usd"].mean().sort_index()
            coords = " ".join(f"({int(b)},{v:.6f})" for b, v in agg.items())
            tikz_path = fig_dir / "anchoring_cost_generated.tikz"
            tikz_content = rf"""% Auto-generated anchoring cost vs batch size
\begin{{tikzpicture}}
  \begin{{axis}}[
    width=9cm,height=5.2cm,xlabel={{Batch size}},ylabel={{Fee (USD)}},
    ymin=0,xmin={min(agg.index)},xmax={max(agg.index)},grid=both,
    title={{Anchoring Fee vs Batch Size}},
  ]
  \addplot+[mark=*] coordinates {{ {coords} }};
  \end{{axis}}
\end{{tikzpicture}}
"""
            tikz_path.write_text(tikz_content)
            print("Wrote generated anchoring cost TikZ ->", tikz_path)
            break  # only once


# ---------------------------------------------------------------------------
# Unified payload LaTeX table generation
# ---------------------------------------------------------------------------


def write_unified_payload_table(unified: pd.DataFrame):
    if unified.empty:
        return
    # Aggregate by algorithm (latest run only if run column exists)
    if "run" in unified.columns:
        latest_run = sorted(unified["run"].unique())[-1]
        latest = unified[unified.run == latest_run]
    else:
        latest = unified
    grp = latest.groupby("algorithm")
    agg = grp.agg(
        payload_req_mean=("payload_bits_requested", "mean"),
        payload_emb_mean=("payload_bits_embedded", "mean"),
        payload_rec_mean=("payload_bits_recovered", "mean"),
        ber_mean=("ber_percent", "mean"),
        success_rate=("success_flag", "mean"),
        psnr_mean=("psnr_db", "mean"),
        ssim_mean=("ssim", "mean"),
        embed_ms_mean=("embed_ms", "mean"),
        extract_ms_mean=("extract_ms", "mean"),
    ).reset_index()
    if agg.empty:
        return
    # Derived throughput (bits/s)
    agg["embed_bps_mean"] = np.where(
        agg["embed_ms_mean"] > 0,
        agg["payload_emb_mean"] / (agg["embed_ms_mean"] / 1000.0),
        np.nan,
    )
    agg["extract_bps_mean"] = np.where(
        agg["extract_ms_mean"] > 0,
        agg["payload_rec_mean"] / (agg["extract_ms_mean"] / 1000.0),
        np.nan,
    )

    def fmt(v, d=0):
        if pd.isna(v):
            return "NA"
        return f"{v:.{d}f}" if d else f"{v:.0f}"

    lines = [
        "% Auto-generated unified payload metrics table (do not edit manually)",
        "\\begin{table}[ht]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Unified embedding / recovery metrics (latest run; means per algorithm).}",
        "\\label{tab:unified-payload}",
        "\\begin{tabularx}{\\linewidth}{l r r r r r r r r r r}",
        "\\toprule",
        "Alg & Req & Emb & Rec & BER (\\%) & Succ (\\%) & Emb B/s & Ext B/s & PSNR & SSIM & Emb/Ext (ms) \\\\",
        "\\midrule",
    ]

    # Simple LaTeX esc for algorithms
    def lex(s: str) -> str:
        return (
            s.replace("\\", "\\textbackslash{}")
            .replace("&", "\\&")
            .replace("%", "\\%")
            .replace("_", "\\_")
            .replace("#", "\\#")
            .replace("{", "\\{")
            .replace("}", "\\}")
        )

    for _, row in agg.sort_values("algorithm").iterrows():
        alg = lex(str(row["algorithm"]))
        req = fmt(row["payload_req_mean"])
        emb = fmt(row["payload_emb_mean"])
        rec = fmt(row["payload_rec_mean"])
        ber = fmt(row["ber_mean"], 1)
        succ = (
            fmt(row["success_rate"] * 100, 1)
            if not pd.isna(row["success_rate"])
            else "NA"
        )
        ebps = fmt(row["embed_bps_mean"])
        xbps = fmt(row["extract_bps_mean"])
        psnr = fmt(row["psnr_mean"], 1)
        ssim = fmt(row["ssim_mean"], 3)
        ems = fmt(row["embed_ms_mean"])
        xms = fmt(row["extract_ms_mean"])
        lines.append(
            f"{alg} & {req} & {emb} & {rec} & {ber} & {succ} & {ebps} & {xbps} & {psnr} & {ssim} & {ems}/{xms} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabularx}", "\\end{table}", ""]
    out_path = CFG.unified_out_dir / "unified_payload_table_generated.tex"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote unified payload table ->", out_path)


def _build_unified_table(all_runs: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    unified = build_unified_table(all_runs)
    if unified.empty:
        print("Unified table empty; aborting further analysis.")
        return unified  # return empty DataFrame instead of None
    write_unified_payload_table(unified)
    return unified


def main():  # pragma: no cover (script entry)
    args = parse_args()
    if args.no_plots:
        CFG.show_plots = False
    all_runs = load_all_runs()
    if not all_runs:
        print("No result runs found under", CFG.results_root / CFG.suite_name)
        return
    if args.purge_dummy:
        maybe_purge_dummy()
        return
    # Build unified table (all rows)
    unified = _build_unified_table(all_runs)
    if unified.empty:
        return
    if args.just_table:
        return
    # Summaries & deltas
    history, deltas = pd.DataFrame(), pd.DataFrame()
    history, deltas, latest_meta = summarize_history(all_runs)
    if history.empty:
        print("No history produced.")
        return
    latest_run = latest_meta["latest_run"].iloc[0]
    print("=== Aggregated Metric History ===")
    display(history)
    print("=== Latest Run Deltas (vs previous) ===")
    latest_view = deltas[deltas.run == latest_run]
    display(latest_view)
    write_summary_json(history, deltas, latest_run)
    # Generate LaTeX macro file + static TikZ assets
    write_metrics_macros(all_runs[latest_run])
    generate_static_tikz(all_runs[latest_run])
    # Plots on latest run groups
    if CFG.show_plots:
        generate_plots(all_runs[latest_run])


if __name__ == "__main__":
    main()

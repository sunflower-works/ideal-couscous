#!/usr/bin/env python3
"""
tegrastats_logger.py

Collects Jetson tegrastats output and parses key metrics to CSV. Optionally runs a workload command
in parallel while logging.

Usage:
  python tegrastats_logger.py --out logs/tegrastats.csv --interval-ms 1000 --duration-sec 600 \
      --run "bash -lc 'nvpmodel -q'"

Notes:
- Requires 'tegrastats' in PATH. If missing, exits with code 2.
- CSV headers: ts_iso,cpu,gpu,ram,swap,emc,temps,power
- Raw fields are lightly normalized; exact columns may vary by JetPack version so we keep consolidated fields.
"""
import argparse
import os
import shlex
import signal
import subprocess as sp
import sys
import time
from datetime import datetime


def has_tegrastats() -> bool:
    from shutil import which

    return which("tegrastats") is not None


def parse_line(line: str) -> dict:
    # Example fragments vary; we preserve some consolidated tokens
    out = {}
    s = line.strip()
    out["raw"] = s
    # split by spaces and commas
    # cpu:..., GR3D_FREQ..., RAM ... SWAP ... EMC_FREQ ... temperature... power
    try:
        # cpu
        cpu_idx = s.find("CPU ")
        if cpu_idx >= 0:
            seg = s[cpu_idx:].split(" ")
            out["cpu"] = seg[1] if len(seg) > 1 else ""
        # gpu
        if "GR3D_FREQ" in s:
            try:
                out["gpu"] = s.split("GR3D_FREQ")[1].split()[0].strip("=,")
            except Exception:
                pass
        # ram/swap
        if "RAM" in s:
            try:
                ram_seg = s.split("RAM")[1].split()[0]
                out["ram"] = ram_seg.strip(",")
            except Exception:
                pass
        if "SWAP" in s:
            try:
                swap_seg = s.split("SWAP")[1].split()[0]
                out["swap"] = swap_seg.strip(",")
            except Exception:
                pass
        # emc
        if "EMC_FREQ" in s:
            try:
                out["emc"] = s.split("EMC_FREQ")[1].split()[0].strip("=,")
            except Exception:
                pass
        # temps (C)
        if "Temp" in s or "temp" in s:
            out["temps"] = ";".join(tok for tok in s.split() if tok.endswith("C"))
        # power
        if "VDD" in s or "POM" in s or "POM_5V" in s:
            # collect all tokens with mW/mA/W
            out["power"] = " ".join(
                tok for tok in s.split() if any(u in tok for u in ("mW", "W ", "mA"))
            )
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/tegrastats.csv")
    ap.add_argument("--interval-ms", type=int, default=1000)
    ap.add_argument(
        "--duration-sec",
        type=int,
        default=0,
        help="0 to run until workload exits or Ctrl+C",
    )
    ap.add_argument(
        "--run", default="", help="optional workload command to run while logging"
    )
    args = ap.parse_args()

    if not has_tegrastats():
        print("[e] tegrastats not found in PATH", file=sys.stderr)
        sys.exit(2)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Start workload (optional)
    workload = None
    if args.run:
        print("[i] Starting workload:", args.run)
        workload = sp.Popen(args.run, shell=True, preexec_fn=os.setsid)

    # Start tegrastats
    cmd = ["tegrastats", "--interval", str(args.interval_ms)]
    proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True, bufsize=1)

    t0 = time.time()
    with open(args.out, "w") as f:
        f.write("ts_iso,cpu,gpu,ram,swap,emc,temps,power\n")
        try:
            for line in proc.stdout:  # type: ignore
                ts = datetime.utcnow().isoformat() + "Z"
                d = parse_line(line)
                f.write(
                    ",".join(
                        [
                            ts,
                            d.get("cpu", ""),
                            d.get("gpu", ""),
                            d.get("ram", ""),
                            d.get("swap", ""),
                            d.get("emc", ""),
                            '"' + d.get("temps", "").replace('"', "") + '"',
                            '"' + d.get("power", "").replace('"', "") + '"',
                        ]
                    )
                    + "\n"
                )
                f.flush()
                if args.duration_sec and (time.time() - t0) >= args.duration_sec:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            try:
                proc.terminate()
            except Exception:
                pass
            if workload is not None:
                try:
                    os.killpg(os.getpgid(workload.pid), signal.SIGTERM)
                except Exception:
                    pass
    print("[i] Wrote:", args.out)


if __name__ == "__main__":
    main()

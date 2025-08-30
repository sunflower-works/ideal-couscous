#!/usr/bin/env python3
"""Emit an environment report with tool/library versions and TRT/PyCUDA availability.
Writes to EDGE_ROOT/results/figures/env_report.txt by default.
"""
import os
import shutil
import subprocess
from datetime import datetime
from importlib import metadata as importlib_metadata


def find_trtexec():
    for c in (
        shutil.which("trtexec"),
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/src/tensorrt/samples/trtexec",
        "/opt/nvidia/tensorrt/bin/trtexec",
        "/usr/local/tensorrt/bin/trtexec",
    ):
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def pkg_ver(name):
    # Prefer reading dist metadata to avoid importing heavy modules
    try:
        return importlib_metadata.version(name)
    except Exception:
        try:
            m = __import__(name)
            return getattr(m, "__version__", "unknown")
        except Exception:
            return "MISSING"


def main():
    edge_root = os.environ.get("EDGE_ROOT", os.path.expanduser("~/edge-yolo"))
    out_dir = os.path.join(edge_root, "results/figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "env_report.txt")

    lines = []
    lines.append("# Edge environment report")
    lines.append(f"date: {datetime.now().isoformat(timespec='seconds')}")
    # Python version
    try:
        py_ver = subprocess.check_output(["python3", "-V"], text=True).strip()
    except Exception:
        py_ver = "python3 UNKNOWN"
    lines.append(f"python: {py_ver}")

    # trtexec details
    trt = find_trtexec()
    lines.append(f"trtexec: {trt or 'not found'}")
    if trt:
        try:
            ver = subprocess.check_output([trt, "--version"], text=True, stderr=subprocess.DEVNULL).strip()
            lines.append(f"trtexec_version: {ver}")
        except Exception:
            pass

    # Libraries
    lines.append(f"onnx {pkg_ver('onnx')}")
    lines.append(f"onnxruntime {pkg_ver('onnxruntime')}")
    lines.append(f"numpy {pkg_ver('numpy')}")
    lines.append(f"opencv-python {pkg_ver('opencv-python')}")
    lines.append(f"ultralytics {pkg_ver('ultralytics')}")
    lines.append(f"torch {pkg_ver('torch')}")
    lines.append(f"torchvision {pkg_ver('torchvision')}")
    lines.append(f"openvino {pkg_ver('openvino')}")
    # Availability flags
    try:
        import pkgutil  # noqa

        tensorrt_avail = bool(pkgutil.find_loader("tensorrt"))
        pycuda_avail = bool(pkgutil.find_loader("pycuda"))
    except Exception:
        tensorrt_avail = False
        pycuda_avail = False
    lines.append(f"tensorrt_available {tensorrt_avail}")
    lines.append(f"pycuda_available {pycuda_avail}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[i] Wrote {out_path}")


if __name__ == "__main__":
    main()

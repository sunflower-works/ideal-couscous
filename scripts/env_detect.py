#!/usr/bin/env python3
"""Environment / hardware detection utilities.

Provides lightweight GPU detection (Torch or CuPy) and helper to report backend info.
Falls back gracefully when dependencies absent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUInfo:
    available: bool
    backend: str
    device_count: int
    name: Optional[str] = None


def detect_gpu() -> GPUInfo:
    # Try PyTorch first
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            cnt = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0) if cnt else None
            return GPUInfo(True, "torch", cnt, name)
    except Exception:
        pass
    # Try CuPy
    try:
        import cupy  # type: ignore

        cnt = cupy.cuda.runtime.getDeviceCount()  # type: ignore
        if cnt > 0:
            name = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()  # type: ignore
            return GPUInfo(True, "cupy", cnt, name)
    except Exception:
        pass
    return GPUInfo(False, "none", 0, None)


if __name__ == "__main__":
    info = detect_gpu()
    print(info)

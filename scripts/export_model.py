#!/usr/bin/env python3
"""Static-shape YOLOv8 model exporter to ONNX.
Usage:
  python export_model.py --model yolov8n.pt --imgsz 416 --out yolov8n_416.onnx --dynamic false
Notes:
  - Uses Ultralytics YOLO export API.
  - Post-export renames produced ONNX to requested --out filename.
"""
import argparse
import os
import sys
from pathlib import Path

from ultralytics import YOLO


def str2bool(v: str) -> bool:
    return v.lower() in {"1", "true", "yes", "y"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt", help="Weights or model alias")
    ap.add_argument("--imgsz", type=int, default=416, help="Square inference size")
    ap.add_argument("--out", default="yolov8n.onnx", help="Output ONNX filename")
    ap.add_argument(
        "--dynamic", default="false", help="Enable dynamic axes (true/false)"
    )
    ap.add_argument("--opset", type=int, default=12, help="ONNX opset")
    ap.add_argument("--nms", default="false", help="Embed NMS into graph if supported (true/false)")
    ap.add_argument("--max-det", type=int, default=300, help="Max detections (TopK-like limit)")
    args = ap.parse_args()

    dyn = str2bool(args.dynamic)
    # Alias correction: map 'yolov11*.pt' -> 'yolo11*.pt' if file missing
    model_arg = args.model
    if not os.path.exists(model_arg):
        base = os.path.basename(model_arg)
        if base.startswith("yolov11"):
            fixed = base.replace("yolov11", "yolo11", 1)
            print(f"[w] Model '{base}' not found. Trying Ultralytics alias '{fixed}'.", file=sys.stderr)
            model_arg = fixed
    print(f"[i] Loading model: {model_arg}")
    model = YOLO(model_arg)
    print(
        f"[i] Exporting to ONNX (imgsz={args.imgsz}, dynamic={dyn}, opset={args.opset})"
    )
    want_nms = str2bool(args.nms)
    if want_nms:
        try:
            from torchvision.ops import nms as _check_nms  # noqa: F401
        except Exception as e:
            print(
                f"[w] torchvision.ops.nms not available ({e}); exporting without embedded NMS",
                file=sys.stderr,
            )
            want_nms = False
    model.export(
        format="onnx",
        imgsz=args.imgsz,
        half=False,
        dynamic=dyn,
        simplify=False,
        opset=args.opset,
        nms=want_nms,
        max_det=args.max_det,
    )
    # Ultralytics writes <model_stem>.onnx in CWD
    produced = f"{Path(args.model).stem}.onnx"
    if not os.path.exists(produced):
        # Fallback: pick the most recent .onnx in CWD
        cands = sorted(Path('.').glob('*.onnx'), key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            produced = str(cands[0])
    if os.path.exists(produced) and args.out != produced:
        os.replace(produced, args.out)
    elif not os.path.exists(args.out) and os.path.exists(produced):
        os.replace(produced, args.out)
    if not os.path.exists(args.out):
        print(f"[e] Expected output {args.out} not found", file=sys.stderr)
        sys.exit(1)
    print(f"[i] Exported -> {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()

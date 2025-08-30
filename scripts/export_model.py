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
    args = ap.parse_args()

    dyn = str2bool(args.dynamic)
    print(f"[i] Loading model: {args.model}")
    model = YOLO(args.model)
    print(
        f"[i] Exporting to ONNX (imgsz={args.imgsz}, dynamic={dyn}, opset={args.opset})"
    )
    model.export(
        format="onnx",
        imgsz=args.imgsz,
        half=False,
        dynamic=dyn,
        simplify=False,
        opset=args.opset,
    )
    produced = "yolov8n.onnx"
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

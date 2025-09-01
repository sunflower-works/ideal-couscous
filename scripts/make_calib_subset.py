#!/usr/bin/env python3
"""Create a small representative calibration image subset from COCO.

Selects N images (random or first N) from one or more COCO subsets (val2017/train2017)
 and copies or symlinks them into a target directory for INT8 calibration.

By default uses symlinks (fast, saves space). Use --copy to physically copy.
Optionally resize images to a max side (keeps aspect) to reduce calibration load.

Example:
  python make_calib_subset.py \
    --coco-root /data/coco \
    --subsets val2017 \
    --count 256 \
    --out-dir calib_images \
    --seed 42 --resize-max 800 --copy

On Jetson (from 10_jetson_yolov8_trt.sh auto path):
  AUTO: if PRECISION=int8 and CALIB_DIR is empty but AUTO_CALIB_COCO_ROOT is set,
  we call this script with --count=$CALIB_SAMPLES and output to auto folder.
"""
import argparse
import zipfile
import random
import shutil
from pathlib import Path
from typing import List

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def discover_images(root: Path, subset: str) -> List[Path]:
    subset_dir = root / subset
    if not subset_dir.exists():
        raise FileNotFoundError(f"Missing subset directory: {subset_dir}")
    return [p for p in subset_dir.iterdir() if p.suffix.lower() in IMG_EXTS]


def resize_if_needed(src: Path, dst: Path, resize_max: int):
    if resize_max <= 0:
        # direct copy
        shutil.copy2(src, dst)
        return
    try:
        from PIL import Image
    except Exception:
        shutil.copy2(src, dst)
        return
    with Image.open(src) as im:
        w, h = im.size
        scale = resize_max / max(w, h)
        if scale < 1.0:
            new_size = (int(w * scale), int(h * scale))
            im = im.resize(new_size, Image.LANCZOS)
        im.save(dst, quality=95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-root", required=True, type=Path)
    ap.add_argument("--subsets", nargs="+", default=["val2017"])
    ap.add_argument("--count", type=int, default=128)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy", action="store_true", help="Copy files instead of symlink")
    ap.add_argument(
        "--resize-max",
        type=int,
        default=0,
        help="If >0, resize so max side <= value (only when copying)",
    )
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="Take first N instead of random sample",
    )
    ap.add_argument(
        "--auto-extract",
        action="store_true",
        help="If subset dir missing, try to unzip subset.zip from COCO root or downloads/",
    )

    args = ap.parse_args()

    rng = random.Random(args.seed)
    imgs: List[Path] = []
    for subset in args.subsets:
        subset_dir = args.coco_root / subset
        if not subset_dir.exists() and args.auto_extract:
            # Try to extract from zip archives if present
            z1 = args.coco_root / f"{subset}.zip"
            z2 = args.coco_root / "downloads" / f"{subset}.zip"
            zsrc = None
            if z1.is_file():
                zsrc = z1
            elif z2.is_file():
                zsrc = z2
            if zsrc is not None:
                try:
                    print(f"[i] Auto-extracting {zsrc} -> {args.coco_root}")
                    with zipfile.ZipFile(zsrc, "r") as zf:
                        zf.extractall(args.coco_root)
                except Exception as e:
                    print(f"[w] Failed to extract {zsrc}: {e}")
            else:
                print(
                    f"[w] Subset {subset} missing and no zip found at {z1} or {z2}; continuing"
                )
        try:
            subset_imgs = discover_images(args.coco_root, subset)
        except FileNotFoundError as e:
            print(f"[w] {e}; skipping subset {subset}")
            continue
        imgs.extend(subset_imgs)
    if not imgs:
        raise SystemExit("[e] No images discovered - check coco root/subsets")

    if args.deterministic:
        selected = imgs[: args.count]
    else:
        if len(imgs) <= args.count:
            selected = imgs
        else:
            selected = rng.sample(imgs, args.count)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[i] Preparing {len(selected)} images -> {args.out_dir} (copy={args.copy} resize_max={args.resize_max})"
    )
    for p in selected:
        dst = args.out_dir / p.name
        if dst.exists():
            continue
        if args.copy:
            if args.resize_max > 0:
                try:
                    resize_if_needed(p, dst, args.resize_max)
                except Exception as e:
                    print(f"[w] resize failed for {p.name}: {e}; copying")
                    shutil.copy2(p, dst)
            else:
                shutil.copy2(p, dst)
        else:
            try:
                dst.symlink_to(p.resolve())
            except FileExistsError:
                pass
            except OSError:
                # fallback to copy if symlink not permitted
                shutil.copy2(p, dst)
    print(f"[i] Calibration subset ready: {args.out_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()

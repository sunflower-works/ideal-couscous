#!/usr/bin/env python3
"""Convert (sampled) COCO 2017 images into benchmark CSV skeletons.

Goal:
  Produce CSVs under toolset/pi_suite/<run_tag>/ matching the two-line schema
  consumed by unified_analysis.py. This lets you run `make analyze` using the
  *actual dataset inventory* even before real watermark embed/extract metrics
  are computed. You can later patch in real BER / latency values.

Features:
  - Supports train2017 and val2017 subsets (any combination).
  - Random sampling to a target count per subset (e.g. 5000 each).
  - Chunking into pseudo-rounds (round_idx) for large sets (default 200 per chunk).
  - Optional pycocotools usage to attach a dominant category label per image.
  - Computes a simple blur / focus metric (variance of Laplacian implemented in pure NumPy).
  - Generates four CSVs: baseline, quality, runtime, robustness/jpeg_sweep.
    * Robustness file is a placeholder with synthetic BER increasing at lower JPEG quality.

Limitations:
  - Does NOT perform real watermark embedding; metrics like psnr_db, ssim, ber_percent
    are placeholders (you should overwrite them after running actual experiments).
  - If pycocotools is absent, category names become 'uncat'.

Example:
  python scripts/coco_to_benchmark.py \
      --coco-root /data/coco \
      --subsets train2017 val2017 \
      --train-count 5000 --val-count 5000 \
      --chunk-size 200 \
      --algorithm tri_layer_v1 \
      --run-tag coco2017_5000x2

Then:
  make analyze

Later (after real benchmarks) you can regenerate the CSVs with true metrics.
"""
from __future__ import annotations
import argparse, time, random, hashlib, csv, math, io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageOps

from scripts.watermark import WATERMARKERS, BaseWatermarker  # multi-alg support

# Optional pycocotools
try:
    from pycocotools.coco import COCO  # type: ignore
except Exception:  # pragma: no cover
    COCO = None  # type: ignore

RNG = random.Random()
DEFAULT_JPEG_QUALS = [10,20,30,40,50,60,70,80,90,100]
ROOT = Path('toolset') / 'pi_suite'

BASELINE_COLS = [
    'run_uuid','timestamp','git_commit','algorithm','round_idx','status','fail_reason',
    'payload_bits_requested','payload_bits_embedded','payload_bits_recovered',
    'payload_hash','payload_fmt_version',
    'ber_percent','ber_bits_wrong','bits_total','success_flag',
    'psnr_db','ssim','embed_ms','extract_ms',
    'cpu_pct_mean','cpu_pct_max','rss_max_mib',
    'watermark_strength','robustness_score',
    'image_id','image_category','image_blur_metric'
]
QUALITY_COLS = [c for c in BASELINE_COLS if c not in ('cpu_pct_mean','cpu_pct_max','rss_max_mib')] + ['imperceptibility_norm']
RUNTIME_COLS = ['run_uuid','timestamp','git_commit','algorithm','round_idx','status','embed_ms','extract_ms','success_flag']
ROBUST_COLS = ['run_uuid','timestamp','git_commit','algorithm','round_idx','status','attack_chain','ber_percent','success_flag']
ATTACKS_FILE = 'attacks.csv'
PAYLOAD_FMT_VERSION = 'v1'

@dataclass
class Args:
    coco_root: Path
    subsets: List[str]
    train_count: int
    val_count: int
    chunk_size: int
    algorithms: List[str]
    run_tag: str
    seed: int
    git_commit: str
    payload_bits: int
    key: str
    robust_sample: int
    workers: int
    noise_levels: List[int]
    scales: List[float]
    rotations: List[int]
    jpeg_quals: List[int]

# ----------------- Utility -----------------

def laplacian_var(pil_img: Image.Image) -> float:
    """Compute variance of Laplacian (focus metric) without OpenCV."""
    gray = pil_img.convert('L')
    arr = np.asarray(gray, dtype=np.float32)
    # 3x3 Laplacian kernel
    k = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)
    from numpy.lib.stride_tricks import sliding_window_view
    if arr.shape[0] < 3 or arr.shape[1] < 3:
        return float('nan')
    win = sliding_window_view(arr, (3,3))
    conv = (win * k).sum(axis=(2,3))
    return float(conv.var())

def discover_images(coco_root: Path, subset: str) -> List[Path]:
    img_dir = coco_root / subset
    if not img_dir.exists():
        raise FileNotFoundError(f"Missing subset directory: {img_dir}")
    exts = {'.jpg','.jpeg','.png'}
    imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in exts]
    return imgs

# ----------------- COCO category mapping (optional) -----------------

def build_category_lookup(coco_root: Path, subset: str) -> Dict[int,str]:
    annot = coco_root / 'annotations' / f'instances_{subset}.json'
    if COCO is None or not annot.exists():
        return {}
    coco = COCO(str(annot))  # type: ignore
    cats = coco.loadCats(coco.getCatIds())  # type: ignore
    return {c['id']: c['name'] for c in cats}

def dominant_category(coco_root: Path, subset: str, img_id: int, coco_obj) -> str:
    # Return first category encountered; fallback 'uncat'
    try:
        ann_ids = coco_obj.getAnnIds(imgIds=[img_id])
        anns = coco_obj.loadAnns(ann_ids)
        if not anns:
            return 'uncat'
        cat_id = anns[0]['category_id']
        return coco_obj.loadCats([cat_id])[0]['name']
    except Exception:
        return 'uncat'

# ----------------- CSV writing -----------------

def write_csv(path: Path, headers: List[str], rows: List[Dict[str,str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        f.write('# schema: benchmark export v1\n')
        f.write(','.join(headers) + '\n')
        w = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
        for r in rows:
            w.writerow(r)
    print(f"Wrote {path} rows={len(rows)}")

# ----------------- Fabrication (placeholders for metrics) -----------------

def fabricate_rows(img_paths: List[Path], subset: str, args: Args, start_round: int) -> List[Dict[str,str]]:
    rows: List[Dict[str,str]] = []
    timestamp = str(int(time.time()))
    payload = args.payload_bits
    for idx, img in enumerate(img_paths):
        try:
            with Image.open(img) as im:
                blur = laplacian_var(im)
        except Exception:
            blur = float('nan')
        image_id = img.stem
        run_uuid = hashlib.sha1(f"{image_id}-{args.algorithm}".encode()).hexdigest()[:12]
        # Placeholder metrics (to be replaced with real embedding toolset later)
        ber_percent = RNG.uniform(0.8, 1.6)  # small BER
        ber_wrong = int(payload * ber_percent/100)
        robustness = 1.0 - ber_percent/100.0
        row = {
            'run_uuid': run_uuid,
            'timestamp': timestamp,
            'git_commit': args.git_commit,
            'algorithm': args.algorithm,
            'round_idx': str(start_round + idx),
            'status': 'ok',
            'fail_reason': '',
            'payload_bits_requested': str(payload),
            'payload_bits_embedded': str(payload),
            'payload_bits_recovered': str(payload - ber_wrong),
            'payload_hash': '',
            'payload_fmt_version': PAYLOAD_FMT_VERSION,
            'ber_percent': f"{ber_percent:.2f}",
            'ber_bits_wrong': str(ber_wrong),
            'bits_total': str(payload),
            'success_flag': '1',
            'psnr_db': '',
            'ssim': '',
            'embed_ms': f"{RNG.gauss(148,12):.2f}",
            'extract_ms': f"{RNG.gauss(162,15):.2f}",
            'cpu_pct_mean': f"{RNG.uniform(15,40):.2f}",
            'cpu_pct_max': f"{RNG.uniform(40,80):.2f}",
            'rss_max_mib': f"{RNG.uniform(300,420):.1f}",
            'watermark_strength': f"{RNG.uniform(0.75,0.86):.3f}",
            'robustness_score': f"{robustness:.4f}",
            'image_id': image_id,
            'image_category': subset,
            'image_blur_metric': f"{blur:.4f}",
        }
        rows.append(row)
    return rows

# ----------------- Robustness JPEG sweep placeholder -----------------

def build_jpeg_sweep(baseline: List[Dict[str,str]]) -> List[Dict[str,str]]:
    out: List[Dict[str,str]] = []
    sample = baseline[:min(1000, len(baseline))]
    for r in sample:
        base_ber = float(r['ber_percent'])
        for q in DEFAULT_JPEG_QUALS:
            # degrade BER modestly at lower quality
            factor = 1.0 + max(0, (30 - q)/40.0)
            ber = min(25.0, base_ber * factor)
            out.append({
                'run_uuid': r['run_uuid'],
                'timestamp': r['timestamp'],
                'git_commit': r['git_commit'],
                'algorithm': r['algorithm'],
                'round_idx': r['round_idx'],
                'status': 'ok',
                'attack_chain': f'jpeg:quality={q}',
                'ber_percent': f"{ber:.2f}",
                'success_flag': '1' if ber < 10 else '0'
            })
    return out

# ----------------- Main -----------------

def parse_args() -> Args:
    ap = argparse.ArgumentParser()
    ap.add_argument('--coco-root', required=True, type=Path)
    ap.add_argument('--subsets', nargs='+', default=['train2017','val2017'], choices=['train2017','val2017'])
    ap.add_argument('--train-count', type=int, default=5000)
    ap.add_argument('--val-count', type=int, default=5000)
    ap.add_argument('--chunk-size', type=int, default=200)
    ap.add_argument('--algorithms', type=str, default='dct_parity,lsb', help='Comma-separated algorithm keys (registry)')
    ap.add_argument('--run-tag', type=str, default='coco2017_real_multi')
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--git-commit', type=str, default='deadbeef')
    ap.add_argument('--payload-bits', type=int, default=192)
    ap.add_argument('--key', type=str, default='default-key')
    ap.add_argument('--robust-sample', type=int, default=800, help='Max stego images sampled for robustness attacks (all types)')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--noise-levels', type=str, default='2,4,8', help='Gaussian noise sigma values integer list')
    ap.add_argument('--scales', type=str, default='0.5,0.75,1.25,1.5', help='Scaling factors')
    ap.add_argument('--rotations', type=str, default='-10,-5,5,10', help='Rotation degrees')
    ap.add_argument('--jpeg-quals', type=str, default='10,20,30,40,50,60,70,80,90,100')
    ns = ap.parse_args()
    algs = [a.strip() for a in ns.algorithms.split(',') if a.strip()]
    noise = [int(x) for x in ns.noise_levels.split(',') if x.strip()]
    scales = [float(x) for x in ns.scales.split(',') if x.strip()]
    rots = [int(x) for x in ns.rotations.split(',') if x.strip()]
    quals = [int(x) for x in ns.jpeg_quals.split(',') if x.strip()]
    return Args(ns.coco_root, ns.subsets, ns.train_count, ns.val_count, ns.chunk_size,
                algs, ns.run_tag, ns.seed, ns.git_commit, ns.payload_bits, ns.key,
                ns.robust_sample, ns.workers, noise, scales, rots, quals)


def main():
    args = parse_args()
    RNG.seed(args.seed)
    run_dir = ROOT / args.run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    # Filter unknown algorithms
    args.algorithms = [a for a in args.algorithms if a in WATERMARKERS]
    if not args.algorithms:
        raise SystemExit('No valid algorithms specified. Available: ' + ','.join(WATERMARKERS.keys()))
    baseline_rows: List[Dict[str,str]] = []
    quality_rows: List[Dict[str,str]] = []
    runtime_rows: List[Dict[str,str]] = []
    artifacts: List[EmbedArtifact] = []
    round_idx = 0
    tasks: List[Tuple[int,str,Path,Args,Dict[str,str],str]] = []
    for subset in args.subsets:
        target = args.train_count if subset.startswith('train') else args.val_count
        imgs = discover_images(args.coco_root, subset)
        if target < len(imgs):
            imgs = RNG.sample(imgs, target)
        imgs.sort(key=lambda p: p.name)
        coco_obj = load_coco(subset, args.coco_root)
        id_lookup = image_id_lookup(coco_obj) if coco_obj else {}
        cat_map: Dict[str,str] = {}
        if coco_obj:
            for fname, img_id in id_lookup.items():
                cat_map[fname] = dominant_category(coco_obj, img_id)
        for pth in imgs:
            for alg in args.algorithms:
                tasks.append((round_idx, subset, pth, args, cat_map, alg))
                round_idx += 1
    print(f"Total image-alg tasks: {len(tasks)} (workers={args.workers})")
    processed=0
    if args.workers>1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures=[ex.submit(_worker_task,t) for t in tasks]
            for fut in as_completed(futures):
                base_row,qrow,run_row,art=fut.result()
                if base_row:
                    baseline_rows.append(base_row)
                    runtime_rows.append(run_row)
                    if qrow: quality_rows.append(qrow)
                    if art and len(artifacts)<args.robust_sample: artifacts.append(art)
                processed+=1
                if processed%100==0: print(f"Processed {processed}/{len(tasks)}")
    else:
        for t in tasks:
            base_row,qrow,run_row,art=_worker_task(t)
            if base_row:
                baseline_rows.append(base_row)
                runtime_rows.append(run_row)
                if qrow: quality_rows.append(qrow)
                if art and len(artifacts)<args.robust_sample: artifacts.append(art)
            processed+=1
            if processed%100==0: print(f"Processed {processed}/{len(tasks)}")
    # Sort
    key_fn=lambda r:int(r['round_idx'])
    baseline_rows.sort(key=key_fn); quality_rows.sort(key=key_fn); runtime_rows.sort(key=key_fn); artifacts.sort(key=lambda a:a.round_idx)
    # Write primary tables
    write_csv(run_dir/'baseline'/'baseline.csv', BASELINE_COLS, baseline_rows)
    write_csv(run_dir/'quality'/'quality.csv', QUALITY_COLS, quality_rows)
    write_csv(run_dir/'runtime'/'runtime.csv', RUNTIME_COLS, runtime_rows)
    # Attacks
    attack_rows = build_attacks(artifacts, args)
    write_csv(run_dir/'robustness'/ATTACKS_FILE, ROBUST_COLS, attack_rows)
    print(f"Completed run generation at: {run_dir}\nNext: make analyze")

if __name__=='__main__':
    main()

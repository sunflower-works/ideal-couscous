#!/usr/bin/env python3
"""Download COCO 2017 dataset for watermarking benchmarks.

Downloads train2017, val2017 images and annotations to specified directory.
Supports resume functionality and verification of downloads.
Adds flexible size verification (tolerance / allow larger / skip) to avoid
false negatives when upstream archives change slightly.
"""
import argparse
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

# COCO 2017 URLs and expected sizes
COCO_URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}

# Expected file sizes (bytes) for verification
EXPECTED_SIZES = {
    "train2017.zip": 18042884608,  # ~18GB (reference; may drift upstream)
    "val2017.zip": 778275200,  # ~778MB (reference)
    "annotations_trainval2017.zip": 252907541,  # ~253MB
}


def format_size(bytes_size: int) -> str:
    """Format bytes to human readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} TB"


def download_with_progress(url: str, filepath: Path, resume: bool = True) -> bool:
    """Download file with progress bar and resume capability."""
    headers = {}
    initial_pos = 0

    if resume and filepath.exists():
        initial_pos = filepath.stat().st_size
        headers["Range"] = f"bytes={initial_pos}-"
        print(f"Resuming download from {format_size(initial_pos)}")

    try:
        req = urllib.request.Request(url, headers=headers)
        response = urllib.request.urlopen(req)

        # Get total size
        if "content-range" in response.headers:
            total_size = int(response.headers["content-range"].split("/")[-1])
        elif "content-length" in response.headers:
            total_size = int(response.headers["content-length"]) + initial_pos
        else:
            total_size = None

        # Open file for writing
        mode = "ab" if resume and initial_pos > 0 else "wb"
        with open(filepath, mode) as f:
            downloaded = initial_pos
            chunk_size = 8192
            start_time = time.time()

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break

                f.write(chunk)
                downloaded += len(chunk)

                # Progress update every 1MB
                if downloaded % (1024 * 1024) == 0 or not chunk:
                    elapsed = time.time() - start_time
                    speed = downloaded / elapsed if elapsed > 0 else 0

                    if total_size:
                        progress = (downloaded / total_size) * 100
                        print(
                            f"\r{filepath.name}: {format_size(downloaded)}/{format_size(total_size)} "
                            f"({progress:.1f}%) - {format_size(speed)}/s",
                            end="",
                            flush=True,
                        )
                    else:
                        print(
                            f"\r{filepath.name}: {format_size(downloaded)} - {format_size(speed)}/s",
                            end="",
                            flush=True,
                        )

        print()  # New line after progress
        return True

    except urllib.error.HTTPError as e:
        if e.code == 416:  # Range not satisfiable - file already complete
            print(f"{filepath.name}: Already complete")
            return True
        else:
            print(f"\nError downloading {filepath.name}: {e}")
            return False
    except Exception as e:
        print(f"\nError downloading {filepath.name}: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """Extract zip file to destination directory."""
    try:
        import zipfile

        print(f"Extracting {zip_path.name}...")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Get total files for progress
            total_files = len(zip_ref.infolist())

            for i, member in enumerate(zip_ref.infolist()):
                zip_ref.extract(member, extract_to)
                if i % 1000 == 0:  # Progress every 1000 files
                    print(f"Extracted {i}/{total_files} files...", end="\r", flush=True)

            print(f"Extracted {total_files} files successfully")
        return True

    except Exception as e:
        print(f"Error extracting {zip_path.name}: {e}")
        return False


def verify_download(
    filepath: Path,
    expected_size: Optional[int] = None,
    tolerance_pct: float = 0.0,
    allow_larger: bool = False,
    skip: bool = False,
) -> bool:
    """Verify downloaded file size.

    Returns True if:
      - skip flag set, or
      - no expected size provided, or
      - actual size == expected, or
      - within +/- tolerance_pct percent of expected, or
      - (allow_larger and actual >= expected and not grossly larger (> +10%)).
    """
    if skip:
        return filepath.exists()
    if not filepath.exists():
        return False
    actual_size = filepath.stat().st_size
    if not expected_size:
        return True
    if actual_size == expected_size:
        return True
    # Compute relative diff
    diff = actual_size - expected_size
    rel = diff / expected_size
    if abs(rel) * 100.0 <= tolerance_pct:
        print(f"{filepath.name}: within tolerance (diff {rel*100:.2f}% )")
        return True
    if allow_larger and diff > 0 and rel <= 0.10:  # allow up to +10% growth
        print(f"{filepath.name}: accepting larger file (+{rel*100:.2f}%)")
        return True
    print(
        f"Size mismatch for {filepath.name}: expected {format_size(expected_size)}, "
        f"got {format_size(actual_size)} (diff {rel*100:.2f}%)"
    )
    return False


def main():
    parser = argparse.ArgumentParser(description="Download COCO 2017 dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="/data/coco",
        help="Directory to download COCO dataset (default: /data/coco)",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        choices=["train2017", "val2017", "annotations"],
        default=["train2017", "val2017", "annotations"],
        help="Which subsets to download",
    )
    parser.add_argument(
        "--no-extract", action="store_true", help="Download only, do not extract"
    )
    parser.add_argument(
        "--cleanup", action="store_true", help="Remove zip files after extraction"
    )
    parser.add_argument(
        "--size-tolerance-pct",
        type=float,
        default=3.0,
        help="Allowed +/- percent difference for size verification (default 3.0)",
    )
    parser.add_argument(
        "--allow-larger",
        action="store_true",
        help="Accept files larger than expected (up to +10%)",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip size verification entirely (trust download)",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir = args.output_dir / "downloads"
    downloads_dir.mkdir(exist_ok=True)

    print(f"Downloading COCO 2017 dataset to: {args.output_dir}")
    print(f"Subsets: {', '.join(args.subsets)}")

    # Calculate total download size
    total_size = sum(
        EXPECTED_SIZES.get(f"{subset}.zip", 0)
        for subset in args.subsets
        if subset != "annotations"
    ) + (
        EXPECTED_SIZES.get("annotations_trainval2017.zip", 0)
        if "annotations" in args.subsets
        else 0
    )

    print(f"Total download size: ~{format_size(total_size)}")

    # Download each subset
    for subset in args.subsets:
        if subset == "annotations":
            filename = "annotations_trainval2017.zip"
        else:
            filename = f"{subset}.zip"
        url = COCO_URLS[subset]
        filepath = downloads_dir / filename
        expected_size = EXPECTED_SIZES.get(filename)

        print(f"\n--- Downloading {subset} ---")

        if verify_download(
            filepath,
            expected_size,
            tolerance_pct=args.size_tolerance_pct,
            allow_larger=args.allow_larger,
            skip=args.skip_verify,
        ):
            print(f"{filename} already downloaded and verified/accepted")
        else:
            success = download_with_progress(url, filepath)
            if not success:
                print(f"Failed to download {subset}")
                continue
            if not verify_download(
                filepath,
                expected_size,
                tolerance_pct=args.size_tolerance_pct,
                allow_larger=args.allow_larger,
                skip=args.skip_verify,
            ):
                print(f"Download verification failed for {subset}")
                continue

        # Extract if requested
        if not args.no_extract:
            extract_success = extract_zip(filepath, args.output_dir)
            if extract_success and args.cleanup:
                print(f"Removing {filename}")
                filepath.unlink()

    # Summary
    print(f"\n--- Download Summary ---")

    # Check what was successfully downloaded/extracted
    for subset in args.subsets:
        if subset == "annotations":
            check_path = args.output_dir / "annotations"
        else:
            check_path = args.output_dir / subset

        if check_path.exists():
            if subset == "annotations":
                print(f"✓ {subset}: Available at {check_path}")
            else:
                image_count = len(list(check_path.glob("*.jpg")))
                print(f"✓ {subset}: {image_count} images at {check_path}")
        else:
            print(f"✗ {subset}: Not found")

    print(f"\nDataset ready at: {args.output_dir}")
    print("You can now run the COCO benchmark script!")


if __name__ == "__main__":
    main()

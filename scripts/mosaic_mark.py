#!/usr/bin/env python3
"""Create a single-page thumbnail mosaic of thesis.pdf and mark pages with embedded ZWSP marks.

Enhancements:
  * Prefer .wmidx sidecar (written by LaTeX) for authoritative mark positions & counts.
  * Fallback to self_verify.verify if index missing or disabled.
  * Optional --badge draws the number of marks inside the red dot (if >1).
  * Existing --grid / --paper / --frame options retained.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict

try:
    from PyPDF2 import PdfReader, PdfWriter
except ImportError:
    print("ERROR: PyPDF2 not installed. pip install PyPDF2", file=sys.stderr)
    sys.exit(1)

try:
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib.colors import red, white
except ImportError:
    print("ERROR: reportlab not installed. pip install reportlab", file=sys.stderr)
    sys.exit(1)

# Try to import self_verify (fallback for index absence)
try:
    import self_verify as self_verify  # when run from scripts/ dir directly
except Exception:
    import sys as _sys
    from pathlib import Path as _Path

    _root = _Path(__file__).resolve().parent.parent
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))
    try:
        import scripts.self_verify as self_verify  # type: ignore
    except Exception:
        self_verify = None  # degraded mode


def parse_index(pdf: Path) -> Dict[int, int]:
    """Parse <pdf>.wmidx returning {page: count}."""
    idx_path = pdf.with_suffix(".wmidx")
    marks: Dict[int, int] = {}
    if not idx_path.exists():
        return marks
    for line in idx_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line.startswith("WM|"):
            continue
        parts = line.split("|")
        if len(parts) < 5:
            continue
        page_s = [p for p in parts if p.startswith("page=")]
        if not page_s:
            continue
        try:
            page = int(page_s[0].split("=", 1)[1])
            marks[page] = marks.get(page, 0) + 1
        except ValueError:
            continue
    return marks


def choose_grid(pages: int) -> tuple[int, int]:
    best = (0, 0, 10**9, 10**9)
    limit = min(pages, 200)
    for r in range(1, limit + 1):
        c = (pages + r - 1) // r
        slots = c * r
        waste = slots - pages
        aspect = abs(c - r)
        if (waste < best[2]) or (waste == best[2] and aspect < best[3]):
            best = (c, r, waste, aspect)
            if waste == 0 and aspect <= 1:
                break
    return best[0], best[1]


def build_mosaic(
    src: Path, mosaic_raw: Path, cols: int, rows: int, paper: str, frame: bool
) -> None:
    if shutil.which("pdfjam") is None:
        print("ERROR: pdfjam not found (install texlive-extra-utils)", file=sys.stderr)
        sys.exit(1)
    # Range 1-N
    pages = len(PdfReader(str(src)).pages)
    cmd = [
        "pdfjam",
        str(src),
        f"1-{pages}",
        "--nup",
        f"{cols}x{rows}",
        "--paper",
        paper,
        "--outfile",
        str(mosaic_raw),
    ]
    if frame:
        cmd.insert(-2, "--frame")
        cmd.insert(-2, "true")
    subprocess.run(cmd, check=True)


def compute_cell_centers(mosaic_pdf: Path, cols: int, rows: int):
    reader = PdfReader(str(mosaic_pdf))
    page = reader.pages[0]
    w = float(page.mediabox.width)
    h = float(page.mediabox.height)
    cell_w = w / cols
    cell_h = h / rows
    centers = []
    for i in range(cols * rows):
        row = i // cols
        col = i % cols
        x = col * cell_w + cell_w / 2
        y = (rows - 1 - row) * cell_h + cell_h / 2
        centers.append((x, y))
    return centers, w, h


def collect_marked_pages(src: Path, use_index: bool) -> Dict[int, int]:
    if use_index:
        marks = parse_index(src)
        if marks:
            return marks
    # fallback self_verify
    if self_verify is None:
        return {}
    pages, _decl, _total = self_verify.verify(
        src, disclosure_page=None, disclosure_regex=self_verify.DISCLOSURE_REGEX_DEFAULT
    )
    result: Dict[int, int] = {}
    for p in pages:
        if p.combined > 0:
            result[p.index + 1] = p.combined
    return result


def main():
    ap = argparse.ArgumentParser(
        description="Create single-page mosaic with red dots on marked pages"
    )
    ap.add_argument(
        "--paper", default="a2paper", help="Target paper (a3paper, a2paper, a1paper)"
    )
    ap.add_argument(
        "--out", default="thesis_mosaic_marked.pdf", help="Output PDF filename"
    )
    ap.add_argument("--grid", help="Optional grid CxR; auto if omitted")
    ap.add_argument("--frame", action="store_true", help="Draw frames around tiles")
    ap.add_argument("--src", default="thesis.pdf", help="Source PDF")
    ap.add_argument(
        "--no-index", action="store_true", help="Ignore .wmidx sidecar even if present"
    )
    ap.add_argument(
        "--badge", action="store_true", help="Draw count badge (number inside dot)"
    )
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"ERROR: source PDF not found: {src}", file=sys.stderr)
        sys.exit(1)

    # Determine page count
    total_pages = len(PdfReader(str(src)).pages)

    marks = collect_marked_pages(src, use_index=not args.no_index)
    if not marks:
        print("[warn] No marks detected (index missing or empty).")

    if args.grid:
        if "x" not in args.grid:
            print("ERROR: --grid must be CxR", file=sys.stderr)
            sys.exit(1)
        cols, rows = map(int, args.grid.split("x", 1))
    else:
        cols, rows = choose_grid(total_pages)

    print(
        f"Mosaic: pages={total_pages} grid={cols}x{rows} paper={args.paper} marked_pages={len(marks)} total_marks={sum(marks.values())}"
    )

    with tempfile.TemporaryDirectory() as td:
        mosaic_raw = Path(td) / "mosaic_raw.pdf"
        build_mosaic(src, mosaic_raw, cols, rows, args.paper, args.frame)
        centers, w, h = compute_cell_centers(mosaic_raw, cols, rows)
        cell_w = w / cols
        cell_h = h / rows

        overlay_path = Path(td) / "overlay.pdf"
        c = Canvas(str(overlay_path), pagesize=(w, h))
        base_radius = max(
            3, int(min(cell_w, cell_h) * 0.08)
        )  # 8% of smaller cell dimension
        for page_num in range(1, cols * rows + 1):
            count = marks.get(page_num, 0)
            if count <= 0:
                continue
            x, y = centers[page_num - 1]
            radius = base_radius
            # If multiple marks, enlarge slightly
            if count > 1:
                radius = int(base_radius * (1 + min(count, 5) * 0.15))
            c.setFillColor(red)
            c.circle(x, y, radius, fill=1, stroke=0)
            if args.badge and count > 1:
                c.setFillColor(white)
                # Choose font size to fit inside circle
                fs = max(6, int(radius * 1.2))
                try:
                    c.setFont("Helvetica-Bold", fs)
                except Exception:
                    c.setFont("Helvetica", fs)
                txt = str(count)
                tw = c.stringWidth(txt, "Helvetica-Bold", fs)
                c.drawString(x - tw / 2, y - fs / 3, txt)
        c.showPage()
        c.save()

        # Merge overlay
        mosaic_reader = PdfReader(str(mosaic_raw))
        overlay_reader = PdfReader(str(overlay_path))
        base_page = mosaic_reader.pages[0]
        base_page.merge_page(overlay_reader.pages[0])
        writer = PdfWriter()
        writer.add_page(base_page)
        out_path = Path(args.out)
        with open(out_path, "wb") as f:
            writer.write(f)

    print(
        f"Written {args.out} (pages marked: {len(marks)}, total marks: {sum(marks.values())})"
    )


if __name__ == "__main__":
    main()

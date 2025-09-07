# --- scripts/mosaic_mark.py ---
import argparse
import shutil
import subprocess
import sys
import tempfile
from math import ceil, sqrt
from pathlib import Path
from typing import Dict

try:
    from PyPDF2 import PdfReader, PdfWriter
except ImportError:
    print("ERROR: PyPDF2 not installed. Run 'pip install PyPDF2'", file=sys.stderr)
    sys.exit(1)

try:
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib.colors import red, white
except ImportError:
    print(
        "ERROR: reportlab not installed. Run 'pip install reportlab'", file=sys.stderr
    )
    sys.exit(1)


def choose_grid(page_count: int) -> tuple[int, int]:
    """Determine an optimal CxR grid minimizing wasted slots and aspect distortion."""
    best = None
    for cols in range(1, int(ceil(sqrt(page_count))) + 2):
        rows = ceil(page_count / cols)
        wasted = cols * rows - page_count
        aspect_ratio_diff = abs(cols - rows)
        score = (wasted, aspect_ratio_diff)
        if best is None or score < best[0:2]:
            best = (*score, cols, rows)
    return best[2:]


def parse_grid(grid_spec: str) -> tuple[int, int]:
    """Parse explicit grid format CxR."""
    try:
        c, r = map(int, grid_spec.lower().replace("x", " ").split())
        if c < 1 or r < 1:
            raise ValueError
        return c, r
    except ValueError:
        print(
            f"ERROR: Invalid grid format '{grid_spec}'. Expected format: 'CxR'.",
            file=sys.stderr,
        )
        sys.exit(1)


def build_mosaic(
    src: Path, output_path: Path, cols: int, rows: int, paper: str, frame: bool = False
):
    """Build a mosaic PDF with given grid dimensions."""
    if shutil.which("pdfjam") is None:
        print(
            "ERROR: pdfjam not found. Install 'texlive-extra-utils'.", file=sys.stderr
        )
        sys.exit(1)

    pages = len(PdfReader(src).pages)
    cmd = [
        "pdfjam",
        str(src),
        f"1-{pages}",
        "--nup",
        f"{cols}x{rows}",
        "--paper",
        paper,
        "--outfile",
        str(output_path),
    ]
    if frame:
        cmd.insert(-2, "--frame")
        cmd.insert(-2, "true")
    subprocess.run(cmd, check=True)


def compute_cell_centers(mosaic: Path, cols: int, rows: int):
    """Compute the center coordinates of each grid cell."""
    reader = PdfReader(str(mosaic))
    page = reader.pages[0]
    w, h = float(page.mediabox.width), float(page.mediabox.height)
    cell_w, cell_h = w / cols, h / rows

    centers = [
        (col * cell_w + cell_w / 2, (rows - row - 1) * cell_h + cell_h / 2)
        for row in range(rows)
        for col in range(cols)
    ]
    return centers, cell_w, cell_h


def parse_watermark_index(pdf: Path) -> Dict[int, int]:
    """Parse watermark data from <pdf>.wmidx or return empty marks."""
    index_file = pdf.with_suffix(".wmidx")
    if not index_file.exists():
        return {}

    marks = {}
    with open(index_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("WM|") or "page=" not in line:
                continue
            try:
                page = int(line.split("page=", 1)[1].split("|", 1)[0])
                marks[page] = marks.get(page, 0) + 1
            except (ValueError, IndexError):
                pass
    return marks


def main():
    parser = argparse.ArgumentParser(
        description="Create a thumbnail mosaic with watermark marks."
    )
    parser.add_argument("--src", default="thesis.pdf", help="Source PDF file.")
    parser.add_argument(
        "--out", default="thesis_mosaic_marked.pdf", help="Output mosaic file."
    )
    parser.add_argument(
        "--grid", help="Grid layout CxR (e.g., 4x5). Auto-detected if not provided."
    )
    parser.add_argument(
        "--paper", default="a4paper", help="Target paper size (e.g., a4paper, a3paper)."
    )
    parser.add_argument(
        "--frame", action="store_true", help="Draw frames around mosaic tiles."
    )
    parser.add_argument(
        "--badge", action="store_true", help="Overlay count badges inside red dots."
    )
    args = parser.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"ERROR: Source PDF not found: {src}", file=sys.stderr)
        sys.exit(1)

    # Determine grid size
    page_count = len(PdfReader(src).pages)
    if args.grid:
        cols, rows = parse_grid(args.grid)
    else:
        cols, rows = choose_grid(page_count)

    # Parse watermark marks (if any)
    marks = parse_watermark_index(src)

    # Generate mosaic
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        mosaic_raw = tmpdir / "mosaic_raw.pdf"
        build_mosaic(src, mosaic_raw, cols, rows, args.paper, args.frame)

        # Compute cell centers
        centers, cell_w, cell_h = compute_cell_centers(mosaic_raw, cols, rows)

        # Create overlay with red dots
        # Use the actual mosaic page size for the overlay pagesize to ensure alignment.
        mosaic_reader = PdfReader(str(mosaic_raw))
        mosaic_page = mosaic_reader.pages[0]
        page_w, page_h = float(mosaic_page.mediabox.width), float(
            mosaic_page.mediabox.height
        )

        overlay_path = tmpdir / "overlay.pdf"
        # Guard: if centers empty (shouldn't happen), fallback to page size
        if centers:
            # create canvas sized exactly as the mosaic page
            canvas = Canvas(str(overlay_path), pagesize=(page_w, page_h))
        else:
            canvas = Canvas(str(overlay_path), pagesize=(page_w, page_h))

        base_radius = max(3, int(min(cell_w, cell_h) * 0.08))  # Relative radius
        for page, count in marks.items():
            if page > len(centers):
                continue
            x, y = centers[page - 1]
            r = base_radius * (1 + 0.15 * min(count - 1, 5))  # Scale for higher marks
            canvas.setFillColor(red)
            canvas.circle(x, y, r, fill=1, stroke=0)
            if args.badge and count > 1:
                canvas.setFillColor(white)
                font_size = max(6, int(r * 1.2))
                canvas.setFont("Helvetica-Bold", font_size)
                text_width = canvas.stringWidth(str(count), "Helvetica-Bold", font_size)
                canvas.drawString(x - text_width / 2, y - font_size / 3, str(count))
        canvas.showPage()
        canvas.save()

        # Merge overlay
        mosaic_pdf = PdfReader(str(mosaic_raw))
        overlay_pdf = PdfReader(str(overlay_path))
        mosaic_pdf.pages[0].merge_page(overlay_pdf.pages[0])

        # Correct usage of PdfWriter: create writer, add page(s), then call write()
        writer = PdfWriter()
        writer.add_page(mosaic_pdf.pages[0])
        with open(args.out, "wb") as f:
            writer.write(f)

    print(f"Generated mosaic saved at: {args.out}")


if __name__ == "__main__":
    main()

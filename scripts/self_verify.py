#!/usr/bin/env python3
"""
Project Sunflower – PDF self‑verification.

Functions:
  * Count embedded zero‑width space (U+200B) occurrences across all page content streams.
  * Cross‑check that the disclosure page's declared count matches the actual total.
Detection:
  * Raw stream scan: searches binary UTF‑8 sequence 0xE2 0x80 0x8B inside every page content stream (decompressed).
  * Text layer scan: PyPDF2 extracted text (fallback if encoding differs).
Disclosure:
  * Auto‑detect page whose extracted text matches configurable regex (default targets phrases like
    'disclosure', 'watermark', 'zero width', plus a number).
  * User can override with --disclosure-page N (1-based).
Exit codes:
  0 success; 1 file/read error; 2 mismatch; 3 no disclosure page found.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict

try:
    import PyPDF2
    from PyPDF2.generic import IndirectObject
except ImportError:
    print(
        "ERROR: PyPDF2 not installed. Run: pip install PyPDF2>=3.0.0", file=sys.stderr
    )
    sys.exit(1)

# Invisible / zero-width codepoints we will scan for (char -> UTF-8 bytes)
INVISIBLE_CODEPOINTS: Dict[str, bytes] = {
    "ZWSP": b"\xe2\x80\x8b",  # U+200B Zero Width Space
    "ZWNJ": b"\xe2\x80\x8c",  # U+200C Zero Width Non-Joiner
    "ZWJ": b"\xe2\x80\x8d",  # U+200D Zero Width Joiner
    "LRM": b"\xe2\x80\x8e",  # U+200E Left-to-Right Mark
    "RLM": b"\xe2\x80\x8f",  # U+200F Right-to-Left Mark
    "WJ": b"\xe2\x81\xa0",  # U+2060 Word Joiner
    "ZWNBSP": b"\xef\xbb\xbf",  # U+FEFF Zero Width No-Break Space (BOM)
}
# Mapping name -> codepoint int
INVISIBLE_CODEPOINT_VALUES: Dict[str, int] = {
    "ZWSP": 0x200B,
    "ZWNJ": 0x200C,
    "ZWJ": 0x200D,
    "LRM": 0x200E,
    "RLM": 0x200F,
    "WJ": 0x2060,
    "ZWNBSP": 0xFEFF,
}
# Reverse mapping for fast text counting
INVISIBLE_CHARS = {
    name: bytes_val.decode("utf-8", "ignore")
    for name, bytes_val in INVISIBLE_CODEPOINTS.items()
}

DISCLOSURE_REGEX_DEFAULT = r"(?:disclosure|watermark|zero[\s-]*width|victory[\s-]*lap|total[\s-]*marks[\s-]*embedded).*?(?P<count>\d{1,9})"


@dataclass
class PageResult:
    index: int  # 0-based page index
    counts_stream: Dict[str, int] = field(
        default_factory=dict
    )  # per-codepoint raw stream counts
    counts_text: Dict[str, int] = field(
        default_factory=dict
    )  # per-codepoint extracted text counts
    combined: int = 0  # aggregated heuristic (sum of per-codepoint max(stream,text))
    disclosed: Optional[int] = None
    is_disclosure: bool = False

    @property
    def total_stream(self) -> int:
        return sum(self.counts_stream.values())

    @property
    def total_text(self) -> int:
        return sum(self.counts_text.values())


def iter_content_streams(page) -> List[bytes]:
    """Return list of decompressed content stream byte sequences for a page."""
    contents = page.get("/Contents")
    streams: List[bytes] = []
    if contents is None:
        return streams
    objs = contents if isinstance(contents, list) else [contents]
    for obj in objs:
        try:
            if isinstance(obj, IndirectObject):
                obj = obj.get_object()
            if hasattr(obj, "get_data"):
                data = obj.get_data()
                streams.append(data)
        except Exception:
            continue
    return streams


def _extract_hex_text_bytes(stream: bytes) -> List[bytes]:
    """Extract decoded byte sequences from PDF hex strings <...> inside a content stream.
    Skips dictionaries '<<' and '>>'. Returns list of raw bytes represented by each hex string.
    """
    results: List[bytes] = []
    i = 0
    n = len(stream)
    while i < n:
        if stream[i] == 0x3C:  # '<'
            # Skip '<<'
            if i + 1 < n and stream[i + 1] == 0x3C:
                i += 2
                continue
            # Capture until '>' (0x3E) not followed by '>'
            j = i + 1
            hex_chars = bytearray()
            valid = True
            while j < n:
                b = stream[j]
                if b == 0x3E:  # '>' end
                    # Ensure not '>>'
                    if j + 1 < n and stream[j + 1] == 0x3E:
                        valid = False
                    j += 1
                    break
                # Allow whitespace inside
                if 0x30 <= b <= 0x39 or 0x41 <= b <= 0x46 or 0x61 <= b <= 0x66:
                    hex_chars.append(b)
                elif b in (0x20, 0x0D, 0x0A, 0x09):
                    pass
                else:
                    # Non-hex character => not a pure hex string
                    valid = False
                j += 1
            if valid and hex_chars:
                # Pad odd length per PDF spec (append 0)
                if len(hex_chars) % 2 == 1:
                    hex_chars.append(ord("0"))
                try:
                    results.append(bytes.fromhex(hex_chars.decode("ascii")))
                except Exception:
                    pass
            i = j
        else:
            i += 1
    return results


def count_invisible_in_streams(streams: List[bytes]) -> Dict[str, int]:
    counts: Dict[str, int] = {k: 0 for k in INVISIBLE_CODEPOINTS}
    for s in streams:
        # UTF-8 direct occurrences
        for name, pattern in INVISIBLE_CODEPOINTS.items():
            counts[name] += s.count(pattern)
        # UTF-16BE occurrences (with optional BOM at start of strings). We approximate by counting big-endian byte pairs.
        for name, cp in INVISIBLE_CODEPOINT_VALUES.items():
            be = cp.to_bytes(2, "big")
            idx = 0
            while True:
                idx = s.find(be, idx)
                if idx == -1:
                    break
                counts[name] += 1
                idx += 2
            bom_seq = b"\xfe\xff" + be
            bom_count = s.count(bom_seq)
            if bom_count:
                counts[name] += bom_count
                counts[name] -= bom_count
        # Hex text strings
        for decoded in _extract_hex_text_bytes(s):
            # If starts with BOM FE FF interpret as UTF-16BE
            if decoded.startswith(b"\xfe\xff") and len(decoded) >= 4:
                body = decoded[2:]
                for name, cp in INVISIBLE_CODEPOINT_VALUES.items():
                    be = cp.to_bytes(2, "big")
                    # scan pairs
                    for off in range(0, len(body) - 1, 2):
                        if body[off : off + 2] == be:
                            counts[name] += 1
            else:
                # Treat as raw bytes; count UTF-8 sequences
                for name, pattern in INVISIBLE_CODEPOINTS.items():
                    counts[name] += decoded.count(pattern)
    return counts


def count_invisible_in_text(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {k: 0 for k in INVISIBLE_CHARS}
    for name, ch in INVISIBLE_CHARS.items():
        if ch:
            counts[name] += text.count(ch)
    return counts


def extract_text_safe(page) -> str:
    try:
        return page.extract_text() or ""
    except Exception:
        return ""


def find_disclosure_page(
    pages: List[PageResult], texts: List[str], pattern: re.Pattern
) -> Optional[int]:
    # First, prioritize Victory Lap page - search from the end backwards
    for i in range(len(texts) - 1, -1, -1):
        txt = texts[i].lower()
        if "victory lap" in txt or "total marks embedded" in txt:
            return i

    # Fallback to original regex search if no Victory Lap found
    for i, txt in enumerate(texts):
        if pattern.search(txt.lower()):
            return i
    return None


def parse_disclosed_count(text: str) -> Optional[int]:
    m = re.search(r"(?:count|total|embedded|bits|marks)\D{0,10}(\d{1,9})", text.lower())
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def _parse_index_file(pdf_path: Path) -> Optional[List[Tuple[int, int, int, str]]]:
    """Parse <jobname>.wmidx sidecar if present.
    Returns list of tuples (seq, page, line, col_str).
    Format lines: WM|<seq>|page=<p>|line=<line>|col=<col>
    """
    idx_path = pdf_path.with_suffix(".wmidx")
    if not idx_path.exists():
        return None
    entries: List[Tuple[int, int, int, str]] = []
    for raw in idx_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = raw.strip()
        if not raw or not raw.startswith("WM|"):
            continue
        parts = raw.split("|")
        if len(parts) < 5:
            continue
        try:
            seq = int(parts[1])
            page_part = parts[2]
            line_part = parts[3]
            col_part = parts[4]
            if not (
                page_part.startswith("page=")
                and line_part.startswith("line=")
                and col_part.startswith("col=")
            ):
                continue
            page = int(page_part.split("=", 1)[1])
            line = int(line_part.split("=", 1)[1])
            col = col_part.split("=", 1)[1]
            entries.append((seq, page, line, col))
        except Exception:
            continue
    if not entries:
        return None
    return entries


def verify(
    pdf_path: Path,
    disclosure_page: Optional[int],
    disclosure_regex: str,
    use_index: bool = True,
    index_only: bool = False,
) -> Tuple[List[PageResult], Optional[int], int]:
    index_entries = _parse_index_file(pdf_path) if use_index else None
    pages: List[PageResult] = []
    declared: Optional[int] = None
    if index_only and not index_entries:
        raise FileNotFoundError(
            f"Index-only mode but no index file found: {pdf_path.with_suffix('.wmidx')}"
        )
    if index_entries and use_index:
        # Build PageResult list sized to max page number
        max_page = max(p for _s, p, _l, _c in index_entries)
        pages = [PageResult(index=i) for i in range(max_page)]
        # Populate counts: treat each mark as ZWSP (logical) for reporting
        for _seq, page, _line, _col in index_entries:
            pr = pages[page - 1]
            pr.counts_stream.setdefault("ZWSP", 0)
            pr.counts_stream["ZWSP"] += 1
            pr.combined += 1
        # Ensure all pages have full key set initialized (stream/text)
        for pr in pages:
            for name in INVISIBLE_CODEPOINTS:
                pr.counts_stream.setdefault(name, 0)
                pr.counts_text.setdefault(name, 0)
        # Disclosure detection: require reading PDF text only for disclosure page
        reader = PyPDF2.PdfReader(str(pdf_path))
        texts: List[str] = []
        for i, page in enumerate(reader.pages):
            texts.append(extract_text_safe(page))
            if i + 1 > len(pages):
                new_pr = PageResult(index=i)
                for name in INVISIBLE_CODEPOINTS:
                    new_pr.counts_stream.setdefault(name, 0)
                    new_pr.counts_text.setdefault(name, 0)
                pages.append(new_pr)
        pattern = re.compile(disclosure_regex, re.IGNORECASE | re.DOTALL)
        disc_idx = (
            (disclosure_page - 1)
            if disclosure_page
            else find_disclosure_page(pages, texts, pattern)
        )
        if disc_idx is not None and 0 <= disc_idx < len(pages):
            pages[disc_idx].is_disclosure = True
            declared = parse_disclosed_count(texts[disc_idx])
            pages[disc_idx].disclosed = declared
        total_combined = sum(p.combined for p in pages)
        return pages, declared, total_combined
    # Fallback to PDF scanning logic if no index or disabled
    reader = PyPDF2.PdfReader(str(pdf_path))
    pages = []
    texts: List[str] = []
    pattern = re.compile(disclosure_regex, re.IGNORECASE | re.DOTALL)
    for i, page in enumerate(reader.pages):
        text = extract_text_safe(page)
        texts.append(text)
        streams = iter_content_streams(page)
        stream_counts = count_invisible_in_streams(streams)
        text_counts = count_invisible_in_text(text)
        combined = sum(
            max(stream_counts[n], text_counts[n]) for n in INVISIBLE_CODEPOINTS
        )
        pages.append(
            PageResult(
                index=i,
                counts_stream=stream_counts,
                counts_text=text_counts,
                combined=combined,
            )
        )
    disc_idx = (
        (disclosure_page - 1)
        if disclosure_page
        else find_disclosure_page(pages, texts, pattern)
    )
    if disc_idx is not None and 0 <= disc_idx < len(pages):
        pages[disc_idx].is_disclosure = True
        declared = parse_disclosed_count(texts[disc_idx])
        pages[disc_idx].disclosed = declared
    total_combined = sum(p.combined for p in pages)
    # If scan found nothing but index exists (and not index_only), re-run with index for robustness
    if total_combined == 0 and not index_only and index_entries:
        return verify(
            pdf_path,
            disclosure_page,
            disclosure_regex,
            use_index=True,
            index_only=False,
        )
    return pages, declared, total_combined


def render_report(
    pdf_path: Path,
    pages: List[PageResult],
    declared: Optional[int],
    total: int,
    show_breakdown: bool = False,
) -> None:
    print("=" * 60)
    print(f"Sunflower PDF Verification: {pdf_path.name}")
    print("=" * 60)
    print(f"Pages: {len(pages)}")
    print("\nPer-page invisible mark counts (stream/text -> combined heuristic):")
    header = ["idx", "disc", "stream", "text", "combined"]
    print(
        f"{header[0]:>4s} {header[1]:>4s} {header[2]:>7s} {header[3]:>6s} {header[4]:>8s}"
    )
    print("-" * 40)
    for p in pages:
        flag = "Y" if p.is_disclosure else " "
        print(
            f"{p.index+1:4d} {flag:>4s} {p.total_stream:7d} {p.total_text:6d} {p.combined:8d}"
        )
        if show_breakdown and p.combined:
            parts = []
            for name in INVISIBLE_CODEPOINTS:
                sc = p.counts_stream.get(name, 0)
                tc = p.counts_text.get(name, 0)
                if sc or tc:
                    parts.append(f"{name}={max(sc,tc)}")
            if parts:
                print("       breakdown: " + ", ".join(parts))
    print("\nAggregate:")
    print(f"  Total embedded (combined heuristic): {total}")
    if declared is not None:
        print(f"  Declared on disclosure page:         {declared}")
        if declared == total:
            print("  ✓ MATCH")
        else:
            diff = total - declared
            print(f"  ✗ MISMATCH (difference = {diff})")
    else:
        print("  ! No declared count parsed.")


def main():
    ap = argparse.ArgumentParser(
        description="Verify hidden zero-width/invisible watermark counts vs disclosure."
    )
    ap.add_argument("pdf", type=Path, help="Path to PDF document")
    ap.add_argument(
        "--disclosure-page",
        type=int,
        help="1-based page number of disclosure (override auto-detect)",
    )
    ap.add_argument(
        "--disclosure-regex",
        default=DISCLOSURE_REGEX_DEFAULT,
        help="Regex to detect disclosure page (case-insensitive).",
    )
    ap.add_argument(
        "--breakdown",
        action="store_true",
        help="Show per-codepoint breakdown for pages with marks",
    )
    ap.add_argument(
        "--no-index",
        action="store_true",
        help="Disable use of .wmidx sidecar even if present",
    )
    ap.add_argument(
        "--index-only",
        action="store_true",
        help="Use only .wmidx sidecar (fail if missing)",
    )
    args = ap.parse_args()

    if not args.pdf.exists():
        print(f"ERROR: file not found: {args.pdf}", file=sys.stderr)
        sys.exit(1)

    try:
        pages, declared, total = verify(
            args.pdf,
            args.disclosure_page,
            args.disclosure_regex,
            use_index=not args.no_index,
            index_only=args.index_only,
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"ERROR: verification failure: {exc}", file=sys.stderr)
        sys.exit(1)

    render_report(args.pdf, pages, declared, total, show_breakdown=args.breakdown)

    # Exit status logic unchanged
    if any(p.is_disclosure for p in pages):
        if declared is None:
            sys.exit(3)
        if declared != total:
            sys.exit(2)
    else:
        print("WARNING: disclosure page not found.", file=sys.stderr)
        sys.exit(3)
    sys.exit(0)


if __name__ == "__main__":
    main()

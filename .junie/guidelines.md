# Project Sunflower – Developer Guidelines

This document captures build, configuration, testing, and development practices that are specific to this repository. It assumes familiarity with LaTeX toolchains and Python environments.

## Build and Configuration

The primary artifact is a LuaLaTeX-built PDF (thesis.pdf). The Makefile wraps latexmk with the correct flags and orchestrates optional analysis and verification steps.

- Core build toolchain
  - LuaLaTeX with shell-escape enabled (required for minted/pygments and externalized TikZ):
    - TeX Live packages: latexmk, lualatex, biber, biblatex, glossaries, minted, tikz/pgf, fontspec, csquotes, xpatch, hyperref, listings, etc.
    - System package examples (choose your distro’s equivalent):
      - openSUSE: zypper in texlive-scheme-full python3-Pygments ghostscript
      - Ubuntu: apt install texlive-full python3-pygments ghostscript
  - Pygments is required by minted; the Makefile already enables -shell-escape.

- Makefile targets (project-specific)
  - make build: latexmk -lualatex -shell-escape thesis.tex → thesis.pdf
  - make analyze: runs scripts/unified_analysis.py (no plots) then rebuilds PDF.
  - make final: runs a final guard (scripts/final_guard.py) then a rebuild. Use this before release.
  - make verify: builds then runs the self-verification script against thesis.pdf.
  - make mosaic GRID=8x8: generates a watermarked thumbnail mosaic (thesis_mosaic_marked.pdf). Requires reportlab.
  - make watch: latexmk -pvc auto-regeneration while editing.
  - make clean: removes latexmk artifacts and a few extra side files.

- Python environment (optional, for scripts)
  - requirements.txt pins heavy ML/benchmarking dependencies used by scripts/ (ultralytics, onnx, onnxruntime, torch, openvino, etc.). You only need this if you run the benchmarking/analysis pipeline.
  - Recommended: Python 3.10–3.12 virtualenv (3.13 works but some packages may lag). Example:
    - python3 -m venv .venv && source .venv/bin/activate
    - pip install --upgrade pip
    - pip install -r requirements.txt
  - Some Jetson/TensorRT paths rely on system packages; see comments in requirements.txt about tensorrt/pycuda being provided by the system on Jetson.

- External tools used by scripts (optional)
  - trtexec (TensorRT): auto-detected by scripts/env_report.py; not required for building the PDF.
  - reportlab (Python): needed for make mosaic. Install with: pip install reportlab
  - PyPDF2 (Python): used by scripts/self_verify.py (make verify). Install with: pip install PyPDF2

## Testing

There is no pytest-based test suite; testing is oriented around reproducible builds and environment verification.

### Running the existing self-checks

- Build and verify the PDF (requires PyPDF2 installed):
  - pip install PyPDF2
  - make verify
  - This will build thesis.pdf and run scripts/self_verify.py to validate invisible watermark disclosure counts; it exits non-zero on mismatch or missing disclosure.

- Environment report smoke test (no heavy deps required):
  - Purpose: capture versions/availability of optional backends without importing heavy libraries (reports "MISSING" instead).
  - Command:
    - python3 scripts/env_report.py
  - Output location: ${HOME}/edge-yolo/results/figures/env_report.txt
  - Clean-up: remove this file and its empty parent directories if you don’t want artifacts left outside the repo.

### Adding a new test (guideline)

When adding new tests, prefer self-contained scripts under scripts/ that:
- Use argparse and exit codes to signal pass/fail.
- Avoid importing heavy libraries unless genuinely needed; prefer importlib.metadata to query versions.
- Write outputs under a controlled directory, and document a cleanup step.

Example structure (pseudo):
- scripts/check_tex_packages.py
  - Checks presence of latexmk and lualatex via shutil.which and prints a brief report.
  - Exits 0 if present, non-zero otherwise.
- Integrate into Makefile as a phony target (e.g., make preflight) if it becomes a routine check.

### Demonstrated simple test (executed)

We executed the following smoke test to ensure it works on a minimal environment:
- Command run: python3 scripts/env_report.py
- Observed output path: ${HOME}/edge-yolo/results/figures/env_report.txt
- Sample content (example):
  - python: Python 3.13.7
  - trtexec: not found
  - numpy: 2.x or MISSING if not installed
  - onnx/onnxruntime/torch/openvino: MISSING if not installed
- Clean-up performed: we removed the generated env_report.txt and pruned now-empty directories.

You can replicate the test, inspect the file, and then clean it with:
- rm -f ${HOME}/edge-yolo/results/figures/env_report.txt
- rmdir --ignore-fail-on-non-empty ${HOME}/edge-yolo/results/figures ${HOME}/edge-yolo/results ${HOME}/edge-yolo || true

## Development Notes and Conventions

- LaTeX sources
  - thesis.tex is the main entry. Chapters live under chapters/*.tex and are included from thesis.tex.
  - Figures: TikZ sources under figures/; some generated .tex pieces live under toolset/* (e.g., toolset/unified/*.tex). Do not edit generated .tex manually; regenerate via scripts or Make targets that produce them.
  - minted is used; ensure Pygments is installed. The Makefile already passes -shell-escape.
  - Bibliography and glossaries are managed by latexmk (biber and makeglossaries are invoked automatically). Do not run them manually unless debugging.

- Python scripts
  - scripts/self_verify.py scans PDFs (and optional .wmidx index) to validate invisible watermark counts declared on a disclosure page.
    - It imports PyPDF2 at module import time; install PyPDF2 to run it outside of CI.
    - You can pass --disclosure-page N or override --disclosure-regex to locate the disclosure.
  - scripts/mosaic_mark.py can generate a watermarked mosaic PDF for quick visual QA; requires reportlab.
  - scripts/env_detect.py and scripts/env_report.py are lightweight and safe to run without optional deps; they degrade gracefully.
  - Where heavy toolchains are optional, scripts prefer metadata/version queries over importing full modules.

- Code style
  - Python: follow PEP 8. Prefer typing annotations and dataclasses for simple records (see env_detect.py). Keep CLI scripts idempotent and make them safe on machines lacking ML stacks.
  - Shell: keep scripts POSIX-compatible where practical; guard optional tools with helpful messages.
  - LaTeX: keep figures in .tikz/.tex under figures/. Generated assets should go to toolset/ subfolders. Use macros.tex and preamble.tex to centralize style.

- Reproducibility tips
  - Pin Python packages via requirements.txt if running the analysis pipeline; prefer a dedicated venv per machine.
  - For releases: use make final to run the final guard and rebuild the document.
  - Consider archiving tool versions (env_report.txt) alongside build artifacts when producing a reproducible report.

## Troubleshooting

- minted errors like "-shell-escape required": ensure the Makefile is used (it supplies -shell-escape) or pass it manually to latexmk.
- Missing PyPDF2 when running make verify: pip install PyPDF2.
- LaTeX build hangs on glossaries/biber: run make clean and rebuild; check that TeX Live is recent.
- Memory/timeouts on figures: TikZ-heavy figures can take long on first build; use make watch for incremental builds.

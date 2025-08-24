# ---------------------------------------------------------------------
# Project Sunflower – simplified latexmk build
# ---------------------------------------------------------------------
SRC       = thesis.tex
OUTPDF    = thesis.pdf
VERIFY    = scripts/self_verify.py
LATEXMK   = latexmk -lualatex -shell-escape -interaction=nonstopmode -halt-on-error -file-line-error

all: build

build:
	@echo "--- Building (latexmk auto-runs biber & glossaries) ---"
	$(LATEXMK) $(SRC)
	@echo "--- Build complete: $(OUTPDF) ---"

analyze:
	@echo "--- Running unified analysis (no plots) ---"
	python3 scripts/unified_analysis.py --no-plots || true
	@echo "--- Building PDF with refreshed metrics/figures ---"
	$(LATEXMK) $(SRC)
	@echo "--- Analyze + build complete: $(OUTPDF) ---"

final: analyze
	@echo "--- Running final build guard ---"
	python3 scripts/final_guard.py
	@echo "--- Final guard passed; rebuilding with FINAL plots toggle if desired ---"
	$(LATEXMK) $(SRC)
	@echo "--- Final build complete: $(OUTPDF) ---"

watch:
	$(LATEXMK) -pvc $(SRC)

verify: build
	@echo "--- Running Self-Verification Script ---"
	python3 $(VERIFY) $(OUTPDF)
	@echo "--- Verification Complete ---"

clean:
	@echo "--- Cleaning build artefacts ---"
	latexmk -C
	rm -f $(OUTPDF) thesis.gls thesis.glg thesis.bbl-SAVE-ERROR thesis.bcf-SAVE-ERROR

.PHONY: all build watch verify clean analyze final
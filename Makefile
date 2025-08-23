# ---------------------------------------------------------------------
# Project Sunflower – latexmk build system
# ---------------------------------------------------------------------
SRC     = thesis.tex
OUTPDF  = thesis.pdf
VERIFY  = scripts/self_verify.py
LATEXMK = latexmk -lualatex -shell-escape -interaction=nonstopmode

all: build verify

build:                          # full compile (latexmk handles biber, glossaries)
	@echo "--- Building PDF with latexmk ---"
	$(LATEXMK) $(SRC) >/dev/null
	@echo "--- PDF Build Complete: $(OUTPDF) ---"

verify:
	@echo "--- Running Self-Verification Script ---"
	python3 $(VERIFY) $(OUTPDF)
	@echo "--- Verification Complete ---"

clean:
	@echo "--- Cleaning up build artefacts ---"
	latexmk -C >/dev/null
	rm -f $(OUTPDF)

.PHONY: all build verify clean
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

mosaic: build
	@echo "--- Generating thumbnail mosaic with watermark marks ---"
	python3 scripts/mosaic_mark.py --paper a2paper --out thesis_mosaic_marked_a2.pdf || { echo 'Hint: pip install reportlab'; exit 1; }
	@echo "--- Mosaic ready: thesis_mosaic_marked_a2.pdf ---"

clean:
	@echo "--- Cleaning build artefacts ---"
	latexmk -C
	rm -f $(OUTPDF) thesis.gls thesis.glg thesis.bbl-SAVE-ERROR thesis.bcf-SAVE-ERROR

# ---------------------------------------------------------------------
# Added edge + Jetson automation targets
# Variables you can override on invocation, e.g.:
#   make coco-download COCO_ROOT=/data/coco SUBSETS="val2017 annotations"
#   make coco-calib-subset COCO_ROOT=/data/coco CALIB_OUT=calib_images CALIB_COUNT=256
#   make jetson-batch JETSON_HOST=192.168.1.79 IMG_SIZE=640 CALIB_COUNT=256 MAP=1 INT8_LAYER=1
#   make jetson-consolidate JETSON_HOST=192.168.1.79 IMG_SIZE=640
#   make precision-diff IMG_SIZE=640
# ---------------------------------------------------------------------
COCO_ROOT ?= /data/coco
SUBSETS ?= val2017 annotations
CALIB_OUT ?= calib_images
CALIB_COUNT ?= 256
CALIB_SEED ?= 42
IMG_SIZE ?= 640
JETSON_HOST ?= 192.168.1.79
JETSON_USER ?= orin
JETSON_SCRIPTS_DIR ?= edge_yolo_scripts
JETSON_ROOT ?= /home/$(JETSON_USER)/edge-yolo
MAP ?= 1
INT8_LAYER ?= 1

# COCO download (images + annotations) — large download if train2017 included.
coco-download:
	@echo "--- COCO download to $(COCO_ROOT) subsets=$(SUBSETS) ---"
	python3 scripts/download_coco.py --output-dir $(COCO_ROOT) --subsets $(SUBSETS)

# Create (or refresh) a calibration subset (symlinks by default; add COPY=1 for copies)
COPY ?= 0
RESIZE_MAX ?= 0
CALIB_SUBSETS ?= val2017
coco-calib-subset:
	@echo "--- Building calibration subset $(CALIB_OUT) from $(CALIB_SUBSETS) (N=$(CALIB_COUNT)) ---"
	python3 scripts/make_calib_subset.py \
	  --coco-root $(COCO_ROOT) \
	  --subsets $(CALIB_SUBSETS) \
	  --count $(CALIB_COUNT) \
	  --out-dir $(CALIB_OUT) \
	  --seed $(CALIB_SEED) \
	  $(if $(filter 1,$(COPY)),--copy,) $(if $(filter-out 0,$(RESIZE_MAX)),--resize-max $(RESIZE_MAX),) --deterministic || true

# Sync inference / benchmarking scripts to Jetson
jetson-sync:
	@echo "--- Sync scripts to Jetson $(JETSON_HOST) ---"
	JETSON_HOST=$(JETSON_HOST) JETSON_USER=$(JETSON_USER) bash scripts/stab.sh

# Run batch precisions (FP32/FP16/INT8) on Jetson with auto calib (if CALIB_OUT uploaded separately) and produce latency CSV.
jetson-batch: jetson-sync
	@echo "--- Jetson batch run (IMG_SIZE=$(IMG_SIZE)) ---"
	ssh $(JETSON_USER)@$(JETSON_HOST) 'cd $(JETSON_ROOT) && \\
		AUTO_CALIB_COCO_ROOT=$(COCO_ROOT) CALIB_SAMPLES=$(CALIB_COUNT) INT8_LAYER_REPORT=$(INT8_LAYER) MAP_EVAL=$(MAP) MAP_DATA=coco128.yaml \\
		bash ~/${JETSON_SCRIPTS_DIR}/11_jetson_batch_precisions.sh $(JETSON_ROOT) $(IMG_SIZE)'

# Consolidate edge report (latency+map+INT8 coverage) on Jetson then pull artefacts.
jetson-consolidate:
	@echo "--- Jetson consolidate (IMG_SIZE=$(IMG_SIZE)) ---"
	ssh $(JETSON_USER)@$(JETSON_HOST) 'cd $(JETSON_ROOT) && python consolidate_edge_report.py \
		--latency-csv latency_comparative_$(IMG_SIZE).csv \
		--map-glob "map_metrics_$(IMG_SIZE)_*.json" \
		--int8-layers int8_layers_$(IMG_SIZE).json \
		--device "Jetson Orin Nano" --imgsz $(IMG_SIZE) \
		--out-tex results/figures/edge_inference_report.tex \
		--out-json results/figures/edge_inference_report.json'
	@echo "--- Pulling artefacts back ---"
	scp $(JETSON_USER)@$(JETSON_HOST):$(JETSON_ROOT)/latency_comparative_$(IMG_SIZE).csv results/figures/ || true
	scp $(JETSON_USER)@$(JETSON_HOST):$(JETSON_ROOT)/results/figures/edge_inference_report.* results/figures/ || true
	scp $(JETSON_USER)@$(JETSON_HOST):$(JETSON_ROOT)/int8_layers_$(IMG_SIZE).json results/figures/ || true

# Generate LaTeX latency table locally from pulled CSV.
latency-latex:
	@echo "--- Generating latency LaTeX table (IMG_SIZE=$(IMG_SIZE)) ---"
	python3 scripts/generate_latency_table.py --csv results/figures/latency_comparative_$(IMG_SIZE).csv \
	  --out results/figures/latency_table_generated.tex --caption "YOLOv8n Latency ($(IMG_SIZE))" --label tab:lat_$(IMG_SIZE)

# Precision diff summary (needs latency CSV + map json files present locally)
precision-diff:
	@echo "--- Precision diff (IMG_SIZE=$(IMG_SIZE)) ---"
	python3 scripts/generate_precision_diff.py \
	  --latency-csv results/figures/latency_comparative_$(IMG_SIZE).csv \
	  --map-glob "map_metrics_$(IMG_SIZE)_*.json" \
	  --imgsz $(IMG_SIZE) \
	  --out-tex results/figures/precision_diff_summary.tex \
	  --out-json results/figures/precision_diff_summary.json || true

# Full edge pipeline: download (optional), calib subset, batch run on Jetson, consolidate + tables, then build thesis.
edge-all: coco-calib-subset jetson-batch jetson-consolidate latency-latex precision-diff build
	@echo "--- Edge full pipeline done (IMG_SIZE=$(IMG_SIZE)) ---"

.PHONY: all build watch verify clean analyze final mosaic
.PHONY += coco-download coco-calib-subset jetson-sync jetson-batch jetson-consolidate latency-latex precision-diff edge-all

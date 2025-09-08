# Scripts layout and entrypoints

This folder hosts all reproducible flows and helpers.

- Platform pipelines
  - Makefile.jetson: full Jetson (TensorRT) pipeline and one-shot `all`.
  - Makefile.rpi5: full Raspberry Pi 5 (ORT) pipeline and one-shot `all`.
- Benchmarks (model/perception)
  - 10_jetson_yolov8_trt.sh, 11_jetson_batch_precisions.sh
  - 10_rpi_yolov8_ort.sh, 10_rpi_yolov8_ov*.sh
  - bench_yolov8.py, bench_trtexec.py, aggregate_latency.py, build_int8_engine.py, list_int8_layers.py
- Watermarking suite
  - watermark.py, coco_to_benchmark.py, make_calib_subset.py, evaluate_map.py
- Report generators
  - consolidate_edge_report.py, generate_latency_table.py, generate_precision_diff.py,
    unified_analysis.py, generate_detector_macros.py, mosaic_mark.py, final_guard.py
- Utilities
  - download_coco.py, export_model.py, env_detect.py, env_report.py, stab.sh

One-shot entrypoints (recommended)
- Jetson (end-to-end):
  ```bash
  make -C report/scripts -f Makefile.jetson all EDGE_ROOT=$HOME/edge-yolo COCO_ROOT=/data/coco IMG_SIZE=640
  ```
- Raspberry Pi 5 (end-to-end):
  ```bash
  make -f report/scripts/Makefile.rpi5 all EDGE_ROOT=$HOME/edge-yolo COCO_ROOT=/data/coco IMG_SIZE=480
  ```
- LaTeX analysis/final build only:
  ```bash
  make -C report analyze
  make -C report final
  ```

Convenience hub
- Use the top-level scripts/Makefile `help` target to view organized targets and wrappers:
  ```bash
  make -C report/scripts help
  ```


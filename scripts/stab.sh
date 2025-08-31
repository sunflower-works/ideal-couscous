#!/bin/bash
set -euo pipefail

# Remote connection parameters
JETSON_HOST=${JETSON_HOST:-192.168.1.79}
JETSON_USER=${JETSON_USER:-orin}
REMOTE_DIR=${REMOTE_DIR:-edge_yolo_scripts}   # relative to ~ on remote
EDGE_ROOT=${EDGE_ROOT:-edge-yolo}              # remote edge project root (~/edge-yolo)
LOCAL_REPO=${LOCAL_REPO:-/home/beb/report/scripts}
# New: full project sync parameters
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "$LOCAL_REPO")" && pwd)}   # local project root (default parent of scripts)
REMOTE_PROJECT_DIR=${REMOTE_PROJECT_DIR:-report}                           # remote dir name for full project (~/<name>)
SYNC_PROJECT=${SYNC_PROJECT:-1}                                            # default now 1: sync entire project (set 0 for legacy scripts-only mode)
PROJECT_DELETE=${PROJECT_DELETE:-1}                                        # if 1, use --delete when syncing full project

# Behavior flags
CLEAN=${CLEAN:-0}
RSYNC=${RSYNC:-1}
DRYRUN=${DRYRUN:-0}
DUP_TO_EDGE_ROOT=${DUP_TO_EDGE_ROOT:-1}

# Validate LOCAL_REPO
if [ ! -d "$LOCAL_REPO" ]; then
  echo "[e] LOCAL_REPO does not exist: $LOCAL_REPO" >&2
  exit 2
fi

# Validate PROJECT_ROOT when full sync requested
if [ "$SYNC_PROJECT" = "1" ] && [ ! -d "$PROJECT_ROOT" ]; then
  echo "[e] PROJECT_ROOT does not exist: $PROJECT_ROOT" >&2
  exit 2
fi

# Warn if invoked via wrong relative path (common pitfall: inside scripts dir running 'bash scripts/stab.sh')
SCRIPT_BASENAME=$(basename "$0")
  echo "[w] Unexpected script invocation name: $SCRIPT_BASENAME. This may indicate incorrect script path resolution. Please run the script as './stab.sh' from the scripts directory." >&2
fi

# Files to sync (explicit allowlist)
FILES=(
  00_edge_common.sh
  10_jetson_yolov8_trt.sh
  10_rpi_yolov8_ort.sh
  11_jetson_batch_precisions.sh
  bench_yolov8.py
  bench_trtexec.py
  aggregate_latency.py
  build_int8_engine.py
  export_model.py
  make_calib_subset.py
  evaluate_map.py
  list_int8_layers.py
  generate_latency_table.py
  generate_precision_diff.py
  consolidate_edge_report.py
  download_coco.py
  Makefile
  Makefile.jetson
)

RUNTIME_DUP=(
  bench_yolov8.py
  bench_trtexec.py
  aggregate_latency.py
  build_int8_engine.py
  export_model.py
  make_calib_subset.py
  evaluate_map.py
  list_int8_layers.py
  generate_latency_table.py
  generate_precision_diff.py
  consolidate_edge_report.py
  download_coco.py
  10_jetson_yolov8_trt.sh
  11_jetson_batch_precisions.sh
  00_edge_common.sh
)

# Prevent local expansion of ~ by quoting
REMOTE_ABS="~/${REMOTE_DIR}"
EDGE_ROOT_ABS="~/${EDGE_ROOT}"
PROJECT_REMOTE_ABS="~/${REMOTE_PROJECT_DIR}"

if [ "$SYNC_PROJECT" = "1" ]; then
  echo "[i] Jetson full project sync start host=${JETSON_HOST} user=${JETSON_USER} local_project=${PROJECT_ROOT} -> ${PROJECT_REMOTE_ABS} dryrun=${DRYRUN} delete=${PROJECT_DELETE}";
else
  echo "[i] Jetson sync start host=${JETSON_HOST} user=${JETSON_USER} dir=${REMOTE_ABS} clean=${CLEAN} rsync=${RSYNC} dryrun=${DRYRUN}";
fi

# Ensure remote base directories (do not expand ~ locally)
if [ "$SYNC_PROJECT" = "1" ]; then
  ssh "${JETSON_USER}@${JETSON_HOST}" "mkdir -p ${PROJECT_REMOTE_ABS}" >/dev/null || { echo "[e] SSH mkdir project failed"; exit 3; }
else
  ssh "${JETSON_USER}@${JETSON_HOST}" "mkdir -p ${REMOTE_ABS} ${EDGE_ROOT_ABS}" >/dev/null || { echo "[e] SSH mkdir failed"; exit 3; }
fi

# Optional clean only applies to scripts mode
if [ "$SYNC_PROJECT" = "0" ] && [ "${CLEAN}" = "1" ]; then
  echo "[i] Cleaning remote directory ${REMOTE_ABS}";
  ssh "${JETSON_USER}@${JETSON_HOST}" "rm -rf ${REMOTE_ABS} && mkdir -p ${REMOTE_ABS}" || { echo "[e] Remote clean failed"; exit 4; }
fi

if [ "$SYNC_PROJECT" = "1" ]; then
  RSYNC_FLAGS=(-av)
  [ "$PROJECT_DELETE" = "1" ] && RSYNC_FLAGS+=(--delete)
  [ "$DRYRUN" = "1" ] && RSYNC_FLAGS+=(--dry-run)
  # Exclude patterns can be appended via RSYNC_EXCLUDES env var (space separated)
  if [ -n "${RSYNC_EXCLUDES:-}" ]; then
    # shellcheck disable=SC2206
    EX_ARR=($RSYNC_EXCLUDES)
    for ex in "${EX_ARR[@]}"; do
      RSYNC_FLAGS+=(--exclude "$ex")
    done
  fi
  echo "[i] Using rsync for full project (flags: ${RSYNC_FLAGS[*]})"
  # Trailing slash on source to copy contents into target dir
  rsync "${RSYNC_FLAGS[@]}" "${PROJECT_ROOT}/" "${JETSON_USER}@${JETSON_HOST}:${PROJECT_REMOTE_ABS}/"
else
  if [ "${RSYNC}" = "1" ]; then
    RSYNC_FLAGS=(-av --chmod=Fu=rw,Fu+x,Du=rwx --delete)
    [ "${DRYRUN}" = "1" ] && RSYNC_FLAGS+=(--dry-run)
    echo "[i] Using rsync (flags: ${RSYNC_FLAGS[*]})"
    for f in "${FILES[@]}"; do
      if [ ! -f "${LOCAL_REPO}/$f" ]; then
        echo "[w] Local file missing: $f" >&2
      fi
    done
    # shellcheck disable=SC2086
    rsync "${RSYNC_FLAGS[@]}" ${FILES[@]/#/${LOCAL_REPO}/} "${JETSON_USER}@${JETSON_HOST}:${REMOTE_ABS}/"
  else
    echo "[i] Using scp fallback (no deletion of extraneous remote files)"
    echo "[i] Copying ${#FILES[@]} files"
    scp "${FILES[@]/#/${LOCAL_REPO}/}" "${JETSON_USER}@${JETSON_HOST}:${REMOTE_ABS}/"
  fi
  ssh "${JETSON_USER}@${JETSON_HOST}" "chmod +x ${REMOTE_ABS}/*.sh 2>/dev/null || true"
fi

if [ "$SYNC_PROJECT" = "0" ] && [ "${DUP_TO_EDGE_ROOT}" = "1" ]; then
  echo "[i] Duplicating runtime scripts into ${EDGE_ROOT_ABS}"
  for f in "${RUNTIME_DUP[@]}"; do
    if [ -f "${LOCAL_REPO}/$f" ]; then
      scp "${LOCAL_REPO}/$f" "${JETSON_USER}@${JETSON_HOST}:${EDGE_ROOT_ABS}/" >/dev/null || echo "[w] dup failed: $f"
    fi
  done
  ssh "${JETSON_USER}@${JETSON_HOST}" "chmod +x ${EDGE_ROOT_ABS}/*.sh 2>/dev/null || true"
fi

if [ "${DRYRUN}" = "0" ]; then
  if [ "$SYNC_PROJECT" = "1" ]; then
    echo "[i] Remote listing (project root top level):"
    ssh "${JETSON_USER}@${JETSON_HOST}" "ls -1 ${PROJECT_REMOTE_ABS} | head -n 50" || echo "[w] listing failed"
  else
    echo "[i] Remote listing (scripts dir):"
    ssh "${JETSON_USER}@${JETSON_HOST}" "ls -1 ${REMOTE_ABS}" || echo "[w] listing failed"
  fi
fi

echo "[i] Sync complete"

echo "[hint] Usage examples:"
echo "  CLEAN=1 bash scripts/stab.sh                        # clean scripts dir sync"
echo "  RSYNC=1 DRYRUN=1 bash scripts/stab.sh               # dry run preview (scripts mode)"
echo "  DUP_TO_EDGE_ROOT=0 bash scripts/stab.sh             # skip duplication to edge root"
echo "  LOCAL_REPO=./scripts bash scripts/stab.sh           # override local scripts path"
echo "  SYNC_PROJECT=1 bash scripts/stab.sh                 # sync entire project into ~/${REMOTE_PROJECT_DIR}"
echo "  SYNC_PROJECT=1 RSYNC_EXCLUDES='*.pdf *.aux .git' bash scripts/stab.sh  # full project with excludes"

#!/usr/bin/env bash
# poster.sh – enlarge a PDF via pdfposter OR tile all pages onto one sheet (mosaic).
# Usage:
#   ./poster.sh [options] [input.pdf [output.pdf]]
# Modes:
#   (default) Poster enlargement using pdfposter.
#   --mosaic  Tile all pages into a single page (A1 by default) using pdfjam.
# Env overrides:
#   BASE_MEDIA   (poster mode base sheet, default: A4)
#   POSTER_SIZE  (poster target size, default: A1)
#   SCALE        (poster scale factor; overrides POSTER_SIZE)
#   MOSAIC_PAPER (mosaic target paper, default: a1paper)
#   MOSAIC_GRID  (e.g. 8x8). If unset, picks near-optimal grid covering pages.
#   MOSAIC_FRAME (1 to draw frames)
#
# Examples:
#   ./poster.sh thesis.pdf poster.pdf --size A0
#   ./poster.sh --scale 2 thesis.pdf big.pdf
#   ./poster.sh --mosaic thesis.pdf allpages.pdf
#   MOSAIC_GRID=9x7 ./poster.sh --mosaic thesis.pdf allpages.pdf
set -euo pipefail

BASE_MEDIA="${BASE_MEDIA:-A4}"
POSTER_SIZE="${POSTER_SIZE:-A1}"
SCALE="${SCALE:-}"
MOSAIC_PAPER="${MOSAIC_PAPER:-a1paper}"
MOSAIC_GRID="${MOSAIC_GRID:-}"
MOSAIC_FRAME="${MOSAIC_FRAME:-0}"
MODE="poster" # or mosaic

INPUT="thesis.pdf"
OUTPUT="poster.pdf"

usage() {
  sed -n '1,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//'
  echo "\nOptions:\n  --scale N         Poster: scale factor (e.g. 2, 3.5). Overrides --size/POSTER_SIZE.\n  --size Ax         Poster: target poster size (A0..A4 etc). Ignored if --scale given.\n  -m MEDIA          Poster: base media sheet size (default ${BASE_MEDIA}).\n  --mosaic          Mosaic mode: tile all pages on one sheet.\n  --grid CxR        Mosaic grid override (e.g. 8x8). Auto if omitted.\n  --paper PAPER     Mosaic paper (default ${MOSAIC_PAPER}).\n  --frame           Mosaic: draw thin frames around tiles.\n  -o FILE           Output PDF filename (default poster.pdf).\n  -h                Show this help.\n"
}

err() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[poster] $*"; }
trap 'err "Failed (line $LINENO)."' ERR

# Parse positional + options
positional=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scale) [[ ${2:-} ]] || err "--scale requires a value"; SCALE="$2"; shift 2 ;;
    --size)  [[ ${2:-} ]] || err "--size requires a value"; POSTER_SIZE="$2"; shift 2 ;;
    -m)      [[ ${2:-} ]] || err "-m requires a value"; BASE_MEDIA="$2"; shift 2 ;;
    --mosaic) MODE="mosaic"; shift ;;
    --grid)  [[ ${2:-} ]] || err "--grid requires a value (CxR)"; MOSAIC_GRID="$2"; shift 2 ;;
    --paper) [[ ${2:-} ]] || err "--paper requires a value"; MOSAIC_PAPER="$2"; shift 2 ;;
    --frame) MOSAIC_FRAME=1; shift ;;
    -o)      [[ ${2:-} ]] || err "-o requires a value"; OUTPUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    -*) err "Unknown option: $1" ;;
    *) positional+=("$1"); shift ;;
  esac
done

# Remaining positional (if any)
if [[ ${#positional[@]} -gt 0 ]]; then INPUT="${positional[0]}"; fi
if [[ ${#positional[@]} -gt 1 ]]; then OUTPUT="${positional[1]}"; fi

[[ -f "$INPUT" ]] || err "Input PDF not found: $INPUT"
[[ -w "$(dirname "$(readlink -f "$OUTPUT")")" ]] || err "Cannot write to output directory: $(dirname "$OUTPUT")"

if [[ "$MODE" == poster ]]; then
  if ! command -v pdfposter >/dev/null 2>&1; then
    err "pdfposter not found in PATH. Install e.g.: sudo apt-get install pdfposter"
  fi
  if [[ -n "$SCALE" && "$SCALE" =~ ^(--|-) ]]; then
    err "SCALE value looks like an option: $SCALE"
  fi
  if [[ -n "$SCALE" && "$POSTER_SIZE" != "" ]]; then
    log "Scale mode active; ignoring POSTER_SIZE=${POSTER_SIZE}"
  fi
  log "Mode: poster"
  log "Input: $INPUT"; log "Output: $OUTPUT"; log "Base media: $BASE_MEDIA";
  set -x
  if [[ -n "$SCALE" ]]; then
    pdfposter -m"$BASE_MEDIA" -s"$SCALE" "$INPUT" "$OUTPUT"
  else
    pdfposter -m"$BASE_MEDIA" -p"$POSTER_SIZE" "$INPUT" "$OUTPUT"
  fi
  set +x
else
  # Mosaic mode
  if ! command -v pdfjam >/dev/null 2>&1; then
    err "pdfjam not found (apt install pdfjam)"
  fi
  if ! command -v pdfinfo >/dev/null 2>&1; then
    err "pdfinfo not found (apt install poppler-utils)"
  fi
  PAGES=$(pdfinfo "$INPUT" | awk '/^Pages:/ {print $2}')
  [[ $PAGES -gt 0 ]] || err "Could not determine page count"
  if [[ -z "$MOSAIC_GRID" ]]; then
    # Search for near-optimal grid (cols*rows >= pages) minimizing wasted slots then aspect diff
    # Limit search to reasonable rows to avoid huge loops; rows up to pages or 200 whichever smaller
    LIMIT=$(( PAGES < 200 ? PAGES : 200 ))
    BEST_COLS=0; BEST_ROWS=0; BEST_WASTE=999999; BEST_ASPECT=999999
    for ((r=1; r<=LIMIT; r++)); do
      c=$(( (PAGES + r - 1) / r )) # ceil division
      slots=$(( c * r ))
      waste=$(( slots - PAGES ))
      aspect_diff=$(( (c>r?c-r:r-c) ))
      if (( waste < BEST_WASTE || (waste == BEST_WASTE && aspect_diff < BEST_ASPECT) )); then
        BEST_WASTE=$waste; BEST_ASPECT=$aspect_diff; BEST_COLS=$c; BEST_ROWS=$r
        [[ $waste -eq 0 && $aspect_diff -le 1 ]] && break # good enough early exit
      fi
    done
    COLS=$BEST_COLS; ROWS=$BEST_ROWS
  else
    if [[ ! "$MOSAIC_GRID" =~ ^[0-9]+x[0-9]+$ ]]; then
      err "--grid must be CxR (e.g. 8x8)"
    fi
    COLS=${MOSAIC_GRID%x*}; ROWS=${MOSAIC_GRID#*x}
  fi
  TOTAL=$((COLS * ROWS))
  if (( TOTAL < PAGES )); then
    err "Grid ${COLS}x${ROWS} has only $TOTAL slots < $PAGES pages"
  fi
  RANGE="1-$PAGES"
  log "Mode: mosaic"
  log "Input: $INPUT"; log "Pages: $PAGES"; log "Grid: ${COLS}x${ROWS} (slots=$TOTAL, waste=$((TOTAL-PAGES)))"; log "Paper: $MOSAIC_PAPER"; log "Output: $OUTPUT";
  EXTRA=()
  if [[ $MOSAIC_FRAME -eq 1 ]]; then EXTRA+=(--frame true); fi
  set -x
  pdfjam "$INPUT" $RANGE --nup ${COLS}x${ROWS} --paper "$MOSAIC_PAPER" "${EXTRA[@]}" --outfile "$OUTPUT"
  set +x
fi

if [[ -f "$OUTPUT" && -s "$OUTPUT" ]]; then
  log "Created: $OUTPUT (size $(stat -c%s "$OUTPUT") bytes)"
else
  err "Finished but output not found or empty: $OUTPUT"
fi

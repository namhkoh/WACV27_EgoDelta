#!/usr/bin/env bash
set -euo pipefail

: "${PAIR_MANIFEST:?Set PAIR_MANIFEST=/path/to/pairs_fast/<scene>/tomato_pair/manifest.json}"
: "${OUT:=outputs/harvest_scene_001}"
: "${SAM_BACKEND:=native_sam3}"

python -m sam_aria_change_baseline.run_pipeline \
  --task harvesting \
  --mode extracted_frames \
  --pair-manifest "${PAIR_MANIFEST}" \
  --rgb-stream-id "${RGB_STREAM_ID:-214-1}" \
  --frame-stride "${FRAME_STRIDE:-1}" \
  --sam-backend "${SAM_BACKEND}" \
  --use-aux-prompts \
  --out "${OUT}"

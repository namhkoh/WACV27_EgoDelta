#!/usr/bin/env bash
set -euo pipefail

python -m yolo_aria_harvest_poc.render_projected_results \
  --out "${OUT_DIR:-outputs/scene_001}" \
  --vrs "${VRS_PATH:-data/prepost.vrs}" \
  --mps "${MPS_PATH:-data/mps_prepost_vrs}" \
  --rgb-stream-id "${RGB_STREAM_ID:-214-1}" \
  --image-view "${IMAGE_VIEW:-raw}" \
  --aria-camera-label "${ARIA_CAMERA_LABEL:-camera-rgb}" \
  --rect-size "${RECT_SIZE:-720}" \
  --rect-focal "${RECT_FOCAL:-380.0}" \
  --max-frames-per-split "${MAX_FRAMES_PER_SPLIT:-20}"

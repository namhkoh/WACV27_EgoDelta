#!/usr/bin/env bash
set -euo pipefail

python -m yolo_aria_harvest_poc.run_pipeline \
  --mode single_recording \
  --vrs "${VRS_PATH:?Set VRS_PATH}" \
  --mps "${MPS_PATH:?Set MPS_PATH}" \
  --pre-start-s "${PRE_START_S:-0}" \
  --pre-end-s "${PRE_END_S:-22}" \
  --post-start-s "${POST_START_S:-33}" \
  --post-end-s "${POST_END_S:-52}" \
  --model "${MODEL_PATH:-models/best.pt}" \
  --rgb-stream-id "${RGB_STREAM_ID:-214-1}" \
  --image-view "${IMAGE_VIEW:-raw}" \
  --aria-camera-label "${ARIA_CAMERA_LABEL:-camera-rgb}" \
  --rect-size "${RECT_SIZE:-720}" \
  --rect-focal "${RECT_FOCAL:-380.0}" \
  --frame-stride "${FRAME_STRIDE:-10}" \
  --out "${OUT_DIR:-outputs/scene_001}" \
  --world-up "${WORLD_UP:-0,0,1}"

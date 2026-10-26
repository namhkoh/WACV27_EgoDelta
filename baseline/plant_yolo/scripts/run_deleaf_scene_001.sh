#!/usr/bin/env bash
set -euo pipefail

python -m yolo_aria_harvest_poc.run_pipeline \
  --task deleaf \
  --task-config configs/deleaf.yaml \
  --mode single_recording \
  --vrs "${VRS_PATH:-/path/to/scene.vrs}" \
  --mps "${MPS_PATH:-/path/to/mps}" \
  --pre-start-s "${PRE_START_S:-0}" \
  --pre-end-s "${PRE_END_S:-22}" \
  --post-start-s "${POST_START_S:-33}" \
  --post-end-s "${POST_END_S:-52}" \
  --segmentation-backend "${SEGMENTATION_BACKEND:-yoloe_prompt}" \
  --yoloe-model "${YOLOE_MODEL:-yoloe-11s-seg.pt}" \
  --rgb-stream-id "${RGB_STREAM_ID:-214-1}" \
  --image-view "${IMAGE_VIEW:-raw}" \
  --aria-camera-label "${ARIA_CAMERA_LABEL:-camera-rgb}" \
  --rect-size "${RECT_SIZE:-720}" \
  --rect-focal "${RECT_FOCAL:-380.0}" \
  --frame-stride "${FRAME_STRIDE:-10}" \
  --imgsz "${IMGSZ:-1024}" \
  --world-up "${WORLD_UP:-0,0,1}" \
  --out "${OUT_DIR:-outputs/scene_001_deleaf}" \
  --debug

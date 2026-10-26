#!/usr/bin/env bash
set -euo pipefail

: "${PAIR_MANIFEST:?Set PAIR_MANIFEST=/path/to/pairs_fast/<scene>/leaf_pair/manifest.json}"
: "${OUT:=outputs/deleaf_scene_001}"
: "${SAM_BACKEND:=native_sam3}"

extra_args=()
if [[ "${USE_AUX_PROMPTS:-0}" == "1" ]]; then
  extra_args+=(--use-aux-prompts)
fi
if [[ "${DISABLE_DELEAFING_PRIMARY_AXIS_FILTER:-0}" == "1" ]]; then
  extra_args+=(--disable-deleafing-primary-axis-filter)
fi

python -m sam_aria_change_baseline.run_pipeline \
  --task deleafing \
  --mode extracted_frames \
  --pair-manifest "${PAIR_MANIFEST}" \
  --rgb-stream-id "${RGB_STREAM_ID:-214-1}" \
  --frame-stride "${FRAME_STRIDE:-1}" \
  --sam-backend "${SAM_BACKEND}" \
  --deleafing-sam-score-thres "${DELEAFING_SAM_SCORE_THRES:-0.12}" \
  --deleafing-min-mask-area-px "${DELEAFING_MIN_MASK_AREA_PX:-50}" \
  --deleafing-max-mask-area-frac "${DELEAFING_MAX_MASK_AREA_FRAC:-0.03}" \
  --deleafing-max-primary-masks-per-frame "${DELEAFING_MAX_PRIMARY_MASKS_PER_FRAME:-12}" \
  --deleafing-pipe-like-long-side-frac "${DELEAFING_PIPE_LIKE_LONG_SIDE_FRAC:-0.35}" \
  --deleafing-pipe-like-aspect-ratio "${DELEAFING_PIPE_LIKE_ASPECT_RATIO:-6.0}" \
  --deleafing-primary-vertical-axis-thres "${DELEAFING_PRIMARY_VERTICAL_AXIS_THRES:-0.90}" \
  --deleafing-primary-vertical-axis-max-thinness "${DELEAFING_PRIMARY_VERTICAL_AXIS_MAX_THINNESS:-0.45}" \
  --deleafing-primary-vertical-axis-image-direction "${DELEAFING_PRIMARY_VERTICAL_AXIS_IMAGE_DIRECTION:-image_vertical}" \
  --petiole-min-points-inside-mask "${PETIOLE_MIN_POINTS_INSIDE_MASK:-4}" \
  --petiole-dbscan-min-samples "${PETIOLE_DBSCAN_MIN_SAMPLES:-3}" \
  --petiole-instance-min-frames "${PETIOLE_INSTANCE_MIN_FRAMES:-1}" \
  --petiole-min-3d-major-extent-m "${PETIOLE_MIN_3D_MAJOR_EXTENT_M:-0.04}" \
  --petiole-max-3d-major-extent-m "${PETIOLE_MAX_3D_MAJOR_EXTENT_M:-0.55}" \
  --deleafing-aux-sam-score-thres "${DELEAFING_AUX_SAM_SCORE_THRES:-0.35}" \
  --deleafing-aux-min-mask-area-px "${DELEAFING_AUX_MIN_MASK_AREA_PX:-80}" \
  --deleafing-main-stem-min-verticality "${DELEAFING_MAIN_STEM_MIN_VERTICALITY:-0.82}" \
  --deleafing-main-stem-min-height-frac "${DELEAFING_MAIN_STEM_MIN_HEIGHT_FRAC:-0.18}" \
  --deleafing-main-stem-max-width-frac "${DELEAFING_MAIN_STEM_MAX_WIDTH_FRAC:-0.22}" \
  --deleafing-main-stem-min-aspect-ratio "${DELEAFING_MAIN_STEM_MIN_ASPECT_RATIO:-1.8}" \
  --deleafing-aux-min-vertical-extent-m "${DELEAFING_AUX_MIN_VERTICAL_EXTENT_M:-0.08}" \
  --deleafing-aux-min-vertical-ratio "${DELEAFING_AUX_MIN_VERTICAL_RATIO:-1.25}" \
  --deleafing-max-aux-masks-per-frame "${DELEAFING_MAX_AUX_MASKS_PER_FRAME:-8}" \
  "${extra_args[@]}" \
  --out "${OUT}"

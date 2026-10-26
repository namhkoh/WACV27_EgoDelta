#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:-pose}"
DEFAULT_COPY_TO="models/peduncle_${TASK}.pt"

cmd=(
  python -m yolo_aria_harvest_poc.train_yolo_peduncle
  --task "${TASK}"
  --data "${DATA_YAML:-data/peduncle_${TASK}.yaml}"
  --model-size "${YOLO_MODEL_SIZE:-s}"
  --epochs "${EPOCHS:-200}"
  --batch "${BATCH:-8}"
  --imgsz "${IMGSZ:-1024}"
  --project "${TRAIN_PROJECT:-training_runs}"
  --name "${TRAIN_NAME:-peduncle_${TASK}_yolo11${YOLO_MODEL_SIZE:-s}}"
  --copy-to "${MODEL_PATH:-${DEFAULT_COPY_TO}}"
)

if [[ -n "${DEVICE:-}" ]]; then
  cmd+=(--device "${DEVICE}")
fi

if [[ -n "${PRETRAINED:-}" ]]; then
  cmd+=(--pretrained "${PRETRAINED}")
fi

if [[ "${CACHE:-0}" == "1" ]]; then
  cmd+=(--cache)
fi

"${cmd[@]}"


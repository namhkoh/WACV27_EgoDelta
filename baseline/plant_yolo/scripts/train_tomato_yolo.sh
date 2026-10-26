#!/usr/bin/env bash
set -euo pipefail

cmd=(
  python -m yolo_aria_harvest_poc.train_yolo_tomato
  --data "${DATA_YAML:-data/laboro_tomato.yaml}"
  --model "${YOLO_MODEL_SIZE:-x}"
  --epochs "${EPOCHS:-100}"
  --batch "${BATCH:-4}"
  --imgsz "${IMGSZ:-640}"
  --project "${TRAIN_PROJECT:-training_runs}"
  --name "${TRAIN_NAME:-tomato_yolo11_seg}"
  --copy-to "${MODEL_PATH:-models/best.pt}"
)

if [[ -n "${DEVICE:-}" ]]; then
  cmd+=(--device "${DEVICE}")
fi

"${cmd[@]}"

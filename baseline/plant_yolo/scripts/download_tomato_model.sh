#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-models/best.pt}"
MODEL_URL="https://github.com/Fonyuy45/yolo11-tomato-segmentation/releases/download/v1.0/best.pt"

mkdir -p "$(dirname "${MODEL_PATH}")"

if wget -O "${MODEL_PATH}" "${MODEL_URL}"; then
  exit 0
fi

rm -f "${MODEL_PATH}"
cat >&2 <<EOF

Could not download ${MODEL_URL}.

The upstream README documents this URL, but the GitHub repository currently has
no published releases, so the checkpoint is not available from that location.

Prepare a replacement checkpoint from the public Laboro Tomato dataset:

  export ROBOFLOW_API_KEY=...
  python scripts/download_laboro_tomato_dataset.py
  bash scripts/train_tomato_yolo.sh

The training script writes ${MODEL_PATH} by default.
EOF

exit 1

#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-sam3_ultra_aria}"

eval "$(conda shell.bash hook)"
conda create -n "${ENV_NAME}" python=3.12 -y
conda activate "${ENV_NAME}"

python -m pip install -U ultralytics
python -m pip install -r requirements.txt
python -m pip install -e .

cat <<'MSG'
Ultralytics SAM 3 environment is ready.
This adapter requires an ultralytics version that exposes SAM 3 text-prompt prediction.
MSG


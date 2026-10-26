#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade --force-reinstall -r requirements-training-cu121.txt

python - <<'PY'
import torch
import torchvision
from ultralytics import YOLO

print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("cuda_available", torch.cuda.is_available())
print("ultralytics YOLO import OK", YOLO)
PY


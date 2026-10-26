#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-2}"
ITERATION="${2:-30000}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${REPO_ROOT}/baseline/scene_change/scripts/env_paths.sh"
VALIDATION_ROOT="${REPO_ROOT}/baseline/scene_change/runs/paper_validation/gsdiff/Cantina/Instance_1"
SOURCE="${REPO_ROOT}/.external/MV3DCD/data/PASLCD/Cantina/Instance_1"
OUTPUT="${VALIDATION_ROOT}/predictions"

export CUDA_VISIBLE_DEVICES="${GPU}"
cd "${REPO_ROOT}"
"${EGOSCENE_COMMON_ENV}/bin/python" baseline/scene_change/scripts/run_gsdiff_paper.py \
  --root "${VALIDATION_ROOT}" \
  --source "${SOURCE}" \
  --output "${OUTPUT}" \
  --iteration "${ITERATION}" \
  --device cuda \
  --threshold 0.5

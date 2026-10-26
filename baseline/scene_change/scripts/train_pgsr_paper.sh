#!/usr/bin/env bash
set -euo pipefail

SPLIT="${1:?Usage: train_pgsr_paper.sh reference|inference GPU [ITERATIONS]}"
GPU="${2:?Usage: train_pgsr_paper.sh reference|inference GPU [ITERATIONS]}"
ITERATIONS="${3:-30000}"
case "${SPLIT}" in
  reference) PORT=6020 ;;
  inference) PORT=6021 ;;
  *) echo "Split must be reference or inference" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${REPO_ROOT}/baseline/scene_change/scripts/env_paths.sh"
PYTHON="${EGOSCENE_PGSR_ENV}/bin/python"
UPSTREAM="${REPO_ROOT}/.external/PGSR"
VALIDATION_ROOT="${REPO_ROOT}/baseline/scene_change/runs/paper_validation/gsdiff/Cantina/Instance_1"
DATA="${VALIDATION_ROOT}/data/${SPLIT}"
MODEL="${VALIDATION_ROOT}/models/${SPLIT}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export CUDA_HOME="${EGOSCENE_PGSR_ENV}"
export PATH="${EGOSCENE_PGSR_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${EGOSCENE_PGSR_ENV}/lib:${EGOSCENE_PGSR_ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

test -x "${PYTHON}"
test -f "${DATA}/sparse/cameras.txt"
test -f "${DATA}/sparse/images.txt"
test -f "${DATA}/sparse/points3D.txt"
mkdir -p "${MODEL}"
cd "${UPSTREAM}"

# GS-DIFF specifies PGSR but does not publish its training command.  These are
# the official PGSR custom-dataset defaults plus the two robustness flags that
# its README explicitly recommends for custom scenes.
"${PYTHON}" train.py \
  -s "${DATA}" \
  -m "${MODEL}" \
  --iterations "${ITERATIONS}" \
  --save_iterations "${ITERATIONS}" \
  --test_iterations "${ITERATIONS}" \
  --max_abs_split_points 0 \
  --opacity_cull_threshold 0.05 \
  --port "${PORT}"

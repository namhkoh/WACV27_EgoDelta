#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-Cantina}"
INSTANCE="${2:-Instance_1}"
GPU="${3:-1}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${REPO_ROOT}/baseline/scene_change/scripts/env_paths.sh"
PYTHON="${EGOSCENE_COMMON_ENV}/bin/python"
UPSTREAM="${REPO_ROOT}/.external/MV3DCD"
DATA="${UPSTREAM}/data/PASLCD/${DATASET}/${INSTANCE}"
MODEL="${REPO_ROOT}/baseline/scene_change/runs/paper_validation/mv3dcd/${DATASET}/${INSTANCE}"
RESOLUTION=8
THRESHOLD=0.5
PORT=6012
export CUDA_VISIBLE_DEVICES="${GPU}"

test -x "${PYTHON}"
test -d "${DATA}/images"
test -d "${DATA}/gt_mask"
test -d "${DATA}/sparse/0"
mkdir -p "${MODEL}"
cd "${UPSTREAM}"

if [[ ! -f "${MODEL}/point_cloud/iteration_7000/point_cloud.ply" ]]; then
  "${PYTHON}" train.py -s "${DATA}" -m "${MODEL}" --iterations 7000 \
    --change "${INSTANCE}" --resolution "${RESOLUTION}" \
    --checkpoint_iterations 7000 --save_iterations 7000 --port "${PORT}"
fi

"${PYTHON}" render_viewpoints.py -s "${DATA}" -m "${MODEL}" --iterations 7000 \
  --change "${INSTANCE}" --resolution "${RESOLUTION}" --port "${PORT}"
"${PYTHON}" create_mask.py --t "${THRESHOLD}" --input_folder "${MODEL}"

if [[ ! -f "${MODEL}/point_cloud/iteration_10000/point_cloud.ply" ]]; then
  "${PYTHON}" train_masks.py -s "${DATA}" -m "${MODEL}" --iterations 10000 \
    --change "${INSTANCE}" --checkpoint_iterations 10000 \
    --start_checkpoint "${MODEL}/chkpnt7000.pth" --resolution "${RESOLUTION}" \
    --port "${PORT}"
fi

"${PYTHON}" render_viewpoints.py -s "${DATA}" -m "${MODEL}" --iterations 10000 \
  --change "${INSTANCE}" --resolution "${RESOLUTION}" --aug True --port "${PORT}"
"${PYTHON}" create_mask.py --t "${THRESHOLD}" --input_folder "${MODEL}"

if [[ ! -f "${MODEL}/point_cloud/iteration_13000/point_cloud.ply" ]]; then
  "${PYTHON}" train_masks.py -s "${DATA}" -m "${MODEL}" --iterations 13000 \
    --change "${INSTANCE}" --checkpoint_iterations 13000 \
    --start_checkpoint "${MODEL}/chkpnt10000.pth" --resolution "${RESOLUTION}" \
    --augment True --port "${PORT}"
fi

"${PYTHON}" render_viewpoints.py -s "${DATA}" -m "${MODEL}" --iterations 13000 \
  --change "${INSTANCE}" --resolution "${RESOLUTION}" --mask True --port "${PORT}"
"${PYTHON}" evaluate.py --gt "${DATA}/gt_mask" \
  --pred_binary "${MODEL}/renders/binary_masks"

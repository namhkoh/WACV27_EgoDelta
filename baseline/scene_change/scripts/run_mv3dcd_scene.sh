#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 SCENE_ID PREPARED_DATA_ROOT MODEL_ROOT PREDICTIONS_ROOT [GPU] [RESOLUTION]" >&2
  exit 2
fi

SCENE_ID="$1"
PREPARED_ROOT="$(realpath -m "$2")"
MODEL_ROOT="$(realpath -m "$3")"
PREDICTIONS_ROOT="$(realpath -m "$4")"
GPU="${5:-2}"
RESOLUTION="${6:-2}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${REPO_ROOT}/baseline/scene_change/scripts/env_paths.sh"
PYTHON="${EGOSCENE_COMMON_ENV}/bin/python"
UPSTREAM="${REPO_ROOT}/.external/MV3DCD"
DATA="${PREPARED_ROOT}/${SCENE_ID}"
MODEL="${MODEL_ROOT}/${SCENE_ID}"
export CUDA_VISIBLE_DEVICES="${GPU}"

test -x "${PYTHON}"
test -f "${DATA}/egodelta_split.json"
mkdir -p "${MODEL}"
cd "${UPSTREAM}"

if [[ ! -f "${MODEL}/point_cloud/iteration_7000/point_cloud.ply" ]]; then
  "${PYTHON}" train.py -s "${DATA}" -m "${MODEL}" --iterations 7000 --change "${SCENE_ID}" \
    --resolution "${RESOLUTION}" --checkpoint_iterations 7000 --save_iterations 7000 --port 6011
fi

"${PYTHON}" render_viewpoints.py -s "${DATA}" -m "${MODEL}" --iterations 7000 \
  --change "${SCENE_ID}" --resolution "${RESOLUTION}" --port 6011
"${PYTHON}" create_mask.py --t 0.5 --input_folder "${MODEL}"

if [[ ! -f "${MODEL}/point_cloud/iteration_10000/point_cloud.ply" ]]; then
  "${PYTHON}" train_masks.py -s "${DATA}" -m "${MODEL}" --iterations 10000 --change "${SCENE_ID}" \
    --checkpoint_iterations 10000 --start_checkpoint "${MODEL}/chkpnt7000.pth" \
    --resolution "${RESOLUTION}" --port 6011
fi

"${PYTHON}" render_viewpoints.py -s "${DATA}" -m "${MODEL}" --iterations 10000 \
  --change "${SCENE_ID}" --resolution "${RESOLUTION}" --aug True --port 6011
"${PYTHON}" create_mask.py --t 0.5 --input_folder "${MODEL}"

if [[ ! -f "${MODEL}/point_cloud/iteration_13000/point_cloud.ply" ]]; then
  "${PYTHON}" train_masks.py -s "${DATA}" -m "${MODEL}" --iterations 13000 --change "${SCENE_ID}" \
    --checkpoint_iterations 13000 --start_checkpoint "${MODEL}/chkpnt10000.pth" \
    --resolution "${RESOLUTION}" --augment True --port 6011
fi

# Preserve the exact pre/post view split alongside the trained artifact.
cp "${DATA}/egodelta_split.json" "${MODEL}/egodelta_split.json"

cd "${REPO_ROOT}"
"${PYTHON}" -m egoscene.cli render-mv3dcd --scene "${SCENE_ID}" \
  --model-root "${MODEL_ROOT}" --output-root "${PREDICTIONS_ROOT}" --upstream-root "${UPSTREAM}"

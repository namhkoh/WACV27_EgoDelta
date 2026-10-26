#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-2}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${REPO_ROOT}/baseline/scene_change/scripts/env_paths.sh"
PYTHON="${EGOSCENE_3DGSCD_ENV}/bin/python"
UPSTREAM="${REPO_ROOT}/.external/3DGS-CD"
DATA="${UPSTREAM}/data/Mustard"

export CUDA_VISIBLE_DEVICES="${GPU}"
export CUDA_HOME="${EGOSCENE_3DGSCD_ENV}"
export PATH="${EGOSCENE_3DGSCD_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${EGOSCENE_3DGSCD_ENV}/lib:${EGOSCENE_3DGSCD_ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

test -x "${PYTHON}"
test -f "${DATA}/config.yml"
test -f "${DATA}/configs.json"
test -f "${DATA}/transforms.json"
test -f "${DATA}/nerfstudio_models/step-000029999.ckpt"
# The official Mustard Drive folder contains 147 reference views.  The public
# folder UI initially exposed only its first 50 entries, so keep this exact
# count as a guard against silently validating on that truncated download.
test "$(find "${DATA}/rgb" -maxdepth 1 -type f | wc -l)" -eq 147
test "$(find "${DATA}/rgb_new" -maxdepth 1 -type f | wc -l)" -eq 8

cd "${UPSTREAM}"
"${PYTHON}" nerfstudio/scripts/change_det.py \
  --config "${DATA}/config.yml" \
  --transform "${DATA}/transforms.json" \
  --output "${DATA}" \
  --ckpt "${DATA}/nerfstudio_models/"

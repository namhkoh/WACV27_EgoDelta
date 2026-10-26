#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${REPO_ROOT}/baseline/scene_change/scripts/env_paths.sh"
ENV_DIR="${EGOSCENE_3DGSCD_ENV}"
LOCAL_ROOT="/tmp/egodelta-3dgscd-final720-compute-node-2"
SEED_ROOT="${LOCAL_ROOT}/torch_extension_seed"
SEED_EXTENSION="${SEED_ROOT}/gsplat_cuda"
SMOKE_DATA="${REPO_ROOT}/baseline/scene_change/runs/data/3dgscd/egodelta_040bc88b3760_leaf/transforms_pretrain.json"
SESSION="wacv-3dgscd-compute-node-2"

test -x "${ENV_DIR}/bin/ns-train"
test -f "${SMOKE_DATA}"
if (( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) < 2 )); then
  echo "compute-node-2 launcher requires two visible GPUs" >&2
  exit 2
fi
mkdir -p "${LOCAL_ROOT}" "${SEED_ROOT}"

export CUDA_HOME="${ENV_DIR}"
export PATH="${ENV_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${ENV_DIR}/lib:${ENV_DIR}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="9.0+PTX"

if [[ ! -f "${SEED_EXTENSION}/gsplat_cuda.so" ]]; then
  echo "Running the two-step Blackwell/CUDA compatibility preflight on GPU 0..."
  export CUDA_VISIBLE_DEVICES=0
  export TORCH_EXTENSIONS_DIR="${SEED_ROOT}"
  cd "${REPO_ROOT}/.external/3DGS-CD"
  "${ENV_DIR}/bin/ns-train" splatfacto --vis tensorboard \
    --experiment-name compute-node-2_blackwell_preflight \
    --output-dir "${LOCAL_ROOT}/preflight" --timestamp egodelta \
    --steps_per_eval_batch 1000000 --steps_per_eval_image 1000000 \
    --steps_per_eval_all_images 1000000 --logging.steps-per-log 100 \
    --pipeline.model.cull_alpha_thresh 0.005 \
    --pipeline.model.continue_cull_post_densification=False \
    --max-num-iterations 2 --machine.num-devices 1 \
    nerfstudio-data --data "${SMOKE_DATA}" \
    --auto-scale-poses=False --center-method none --orientation-method none --load-3D-points True
  cd "${REPO_ROOT}"
fi
test -f "${SEED_EXTENSION}/gsplat_cuda.so"
unset CUDA_VISIBLE_DEVICES TORCH_EXTENSIONS_DIR

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "${SESSION} is already running; refusing to launch a duplicate fleet" >&2
  exit 2
fi

tmux new-session -d -s "${SESSION}" -n fleet \
  "/usr/bin/bash -lc 'cd ${REPO_ROOT} && set -o pipefail && exec env THREEDGSCD_GPU_COUNT=2 THREEDGSCD_LOCAL_ROOT=${LOCAL_ROOT} THREEDGSCD_SEED_EXTENSION=${SEED_EXTENSION} TORCH_CUDA_ARCH_LIST=9.0+PTX bash baseline/scene_change/scripts/run_3dgscd_fleet.sh 24 16 40 2>&1 | tee baseline/scene_change/runs/logs/final_720_20260817/fleet_orchestrator_compute-node-2.log'"

echo "compute-node-2 fleet launched: workers 24-39 of 40, eight workers per GPU."
echo "Monitor with: tmux attach -t ${SESSION}"
echo "GPU status:   watch -n 2 nvidia-smi"

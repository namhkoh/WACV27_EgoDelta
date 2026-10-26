#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 SCENE_ID PREPARED_DATA_ROOT MODEL_ROOT PREDICTIONS_ROOT [GPU]" >&2
  exit 2
fi

SCENE_ID="$1"
PREPARED_ROOT="$(realpath -m "$2")"
MODEL_ROOT="$(realpath -m "$3")"
PREDICTIONS_ROOT="$(realpath -m "$4")"
GPU="${5:-2}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${REPO_ROOT}/baseline/scene_change/scripts/env_paths.sh"
ENV_DIR="${EGOSCENE_3DGSCD_ENV}"
COMMON_PYTHON="${EGOSCENE_COMMON_ENV}/bin/python"
UPSTREAM="${REPO_ROOT}/.external/3DGS-CD"
DATA="${PREPARED_ROOT}/${SCENE_ID}"
DURABLE_CONFIG="${MODEL_ROOT}/${SCENE_ID}/splatfacto/egodelta/config.yml"
DURABLE_CHECKPOINTS="${MODEL_ROOT}/${SCENE_ID}/splatfacto/egodelta/nerfstudio_models"
DURABLE_FINAL_CHECKPOINT="${DURABLE_CHECKPOINTS}/step-000029999.ckpt"
TRAIN_ROOT="$(realpath -m "${THREEDGSCD_TRAIN_ROOT:-${MODEL_ROOT}}")"
TRAIN_CONFIG="${TRAIN_ROOT}/${SCENE_ID}/splatfacto/egodelta/config.yml"
TRAIN_CHECKPOINTS="${TRAIN_ROOT}/${SCENE_ID}/splatfacto/egodelta/nerfstudio_models"
TRAIN_FINAL_CHECKPOINT="${TRAIN_CHECKPOINTS}/step-000029999.ckpt"
GPU_LOCK_ROOT="${THREEDGSCD_GPU_LOCK_ROOT:-/tmp/egodelta-3dgscd-gpu-locks}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export CUDA_HOME="${ENV_DIR}"
export PATH="${ENV_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${ENV_DIR}/lib:${ENV_DIR}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
# torch.utils.cpp_extension has a shared lock-file race when several independent
# scenes import gsplat simultaneously. Give every GPU shard a durable cache so
# compilation/loading can proceed concurrently without touching another shard's
# baton file.
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/path/to/workspace/.cache/egodelta-scene-change/torch_extensions/3dgscd_gpu${GPU}}"
export MAX_JOBS="${MAX_JOBS:-6}"

test -x "${ENV_DIR}/bin/python"
test -f "${DATA}/transforms_pretrain.json"
mkdir -p "${TRAIN_ROOT}" "${TORCH_EXTENSIONS_DIR}" "${GPU_LOCK_ROOT}"
cd "${UPSTREAM}"

if [[ -f "${DURABLE_FINAL_CHECKPOINT}" ]]; then
  CONFIG="${DURABLE_CONFIG}"
  CHECKPOINTS="${DURABLE_CHECKPOINTS}"
else
  # Resume a durable partial checkpoint in the fast workspace when one exists.
  if [[ ! -d "${TRAIN_CHECKPOINTS}" ]]; then
    mkdir -p "${TRAIN_CHECKPOINTS}"
    LATEST_DURABLE="$(find "${DURABLE_CHECKPOINTS}" -maxdepth 1 -type f -name 'step-*.ckpt' 2>/dev/null | sort | tail -1 || true)"
    if [[ -n "${LATEST_DURABLE}" ]]; then
      cp "${LATEST_DURABLE}" "${TRAIN_CHECKPOINTS}/"
    fi
  fi

  if [[ ! -f "${TRAIN_FINAL_CHECKPOINT}" ]]; then
    LATEST_LOCAL="$(find "${TRAIN_CHECKPOINTS}" -maxdepth 1 -type f -name 'step-*.ckpt' 2>/dev/null | sort | tail -1 || true)"
    REMAINING_STEPS=30000
    LOAD_ARGS=()
    if [[ -n "${LATEST_LOCAL}" ]]; then
      LATEST_NAME="$(basename "${LATEST_LOCAL}")"
      LATEST_STEP="${LATEST_NAME#step-}"
      LATEST_STEP="${LATEST_STEP%.ckpt}"
      LATEST_STEP="$((10#${LATEST_STEP}))"
      REMAINING_STEPS="$((30000 - LATEST_STEP - 1))"
      LOAD_ARGS=(--load-dir "${TRAIN_CHECKPOINTS}")
      echo "  resuming local training from step ${LATEST_STEP}; ${REMAINING_STEPS} steps remain"
    fi
  # Full-image evaluation is diagnostic and extremely expensive; place it
  # beyond the 30k optimization horizon without changing any training step.
    "${ENV_DIR}/bin/ns-train" splatfacto --vis tensorboard \
      --experiment-name "${SCENE_ID}" --output-dir "${TRAIN_ROOT}" --timestamp egodelta \
      --steps_per_eval_batch 1000000 --steps_per_eval_image 1000000 \
      --steps_per_eval_all_images 1000000 --logging.steps-per-log 100 \
      --pipeline.model.cull_alpha_thresh 0.005 \
      --pipeline.model.continue_cull_post_densification=False \
      --max-num-iterations "${REMAINING_STEPS}" "${LOAD_ARGS[@]}" \
      --machine.num-devices 1 nerfstudio-data --data "${DATA}/transforms_pretrain.json" \
      --auto-scale-poses=False --center-method none --orientation-method none --load-3D-points True
  fi
  test -f "${TRAIN_FINAL_CHECKPOINT}"
  CONFIG="${TRAIN_CONFIG}"
  CHECKPOINTS="${TRAIN_CHECKPOINTS}"

  # Archive only the reproducibility-critical final config and checkpoint.
  # TensorBoard events and intermediate checkpoints remain on local NVMe.
  if [[ "${TRAIN_ROOT}" != "${MODEL_ROOT}" ]]; then
    mkdir -p "$(dirname "${DURABLE_CONFIG}")" "${DURABLE_CHECKPOINTS}"
    cp "${TRAIN_CONFIG}" "${DURABLE_CONFIG}.tmp.$$"
    mv "${DURABLE_CONFIG}.tmp.$$" "${DURABLE_CONFIG}"
    cp "${TRAIN_FINAL_CHECKPOINT}" "${DURABLE_FINAL_CHECKPOINT}.tmp.$$"
    mv "${DURABLE_FINAL_CHECKPOINT}.tmp.$$" "${DURABLE_FINAL_CHECKPOINT}"
  fi
fi
test -f "${CONFIG}"
test -f "${CHECKPOINTS}/step-000029999.ckpt"

flock -x "${GPU_LOCK_ROOT}/gpu${GPU}.lock" \
  "${ENV_DIR}/bin/python" "${REPO_ROOT}/baseline/scene_change/scripts/run_3dgscd_known_poses.py" \
    --upstream "${UPSTREAM}" --config "${CONFIG}" \
    --transform "${DATA}/transforms.json" --output "${DATA}" --ckpt "${CHECKPOINTS}"

cd "${REPO_ROOT}"
"${COMMON_PYTHON}" -m egoscene.cli collect-3dgscd --scene "${SCENE_ID}" \
  --official-output-root "${PREPARED_ROOT}" --output-root "${PREDICTIONS_ROOT}"

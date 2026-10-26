#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 START_WORKER WORKER_COUNT TOTAL_WORKERS" >&2
  exit 2
fi

START_WORKER="$1"
WORKER_COUNT="$2"
TOTAL_WORKERS="$3"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOCAL_ROOT="${THREEDGSCD_LOCAL_ROOT:-/tmp/egodelta-3dgscd-final720}"
LOCAL_MODELS="${LOCAL_ROOT}/models"
LOCAL_LOGS="${LOCAL_ROOT}/logs"
DURABLE_LOGS="${REPO_ROOT}/baseline/scene_change/runs/logs/final_720_20260817/fleet"
SEED_EXTENSION="${THREEDGSCD_SEED_EXTENSION:-/path/to/workspace/.cache/torch_extensions/py38_cu118/gsplat_cuda}"
GPU_COUNT="${THREEDGSCD_GPU_COUNT:-4}"

if (( START_WORKER < 0 || WORKER_COUNT < 1 || TOTAL_WORKERS < 1 || GPU_COUNT < 1 )); then
  echo "Invalid fleet range" >&2
  exit 2
fi
if (( START_WORKER + WORKER_COUNT > TOTAL_WORKERS )); then
  echo "Fleet range exceeds total worker count" >&2
  exit 2
fi
test -f "${SEED_EXTENSION}/gsplat_cuda.so"
mkdir -p "${LOCAL_MODELS}" "${LOCAL_LOGS}" "${DURABLE_LOGS}"

PIDS=()
WORKERS=()
for ((WORKER = START_WORKER; WORKER < START_WORKER + WORKER_COUNT; WORKER++)); do
  GPU="$((WORKER % GPU_COUNT))"
  EXTENSION_ROOT="${LOCAL_ROOT}/torch_extensions/worker_${WORKER}"
  if [[ ! -f "${EXTENSION_ROOT}/gsplat_cuda/gsplat_cuda.so" ]]; then
    mkdir -p "${EXTENSION_ROOT}"
    cp -a "${SEED_EXTENSION}" "${EXTENSION_ROOT}/"
  fi
  LOCAL_LOG="${LOCAL_LOGS}/worker_${WORKER}_gpu${GPU}.log"
  DURABLE_LOG="${DURABLE_LOGS}/worker_${WORKER}_gpu${GPU}.log"
  echo "launch worker ${WORKER}/${TOTAL_WORKERS} on GPU ${GPU}"
  (
    set +e
    env \
      OMP_NUM_THREADS=2 \
      MKL_NUM_THREADS=2 \
      MAX_JOBS=2 \
      PYTHONUNBUFFERED=1 \
      THREEDGSCD_TRAIN_ROOT="${LOCAL_MODELS}" \
      TORCH_EXTENSIONS_DIR="${EXTENSION_ROOT}" \
      bash "${REPO_ROOT}/baseline/scene_change/scripts/run_scene_shard.sh" \
        3dgscd "${GPU}" "${WORKER}" "${TOTAL_WORKERS}" \
        > "${LOCAL_LOG}" 2>&1
    STATUS="$?"
    cp "${LOCAL_LOG}" "${DURABLE_LOG}.tmp.$$"
    mv "${DURABLE_LOG}.tmp.$$" "${DURABLE_LOG}"
    exit "${STATUS}"
  ) &
  PIDS+=("$!")
  WORKERS+=("${WORKER}")
done

FAILED=0
for INDEX in "${!PIDS[@]}"; do
  if wait "${PIDS[$INDEX]}"; then
    echo "worker ${WORKERS[$INDEX]} complete"
  else
    echo "worker ${WORKERS[$INDEX]} failed; inspect its log" >&2
    FAILED=1
  fi
done
exit "${FAILED}"

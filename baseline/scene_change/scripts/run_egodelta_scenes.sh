#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 GPU [SCENE_ID ...]" >&2
  exit 2
fi

GPU="$1"
shift
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${REPO_ROOT}/baseline/scene_change/scripts/env_paths.sh"
PYTHON="${EGOSCENE_COMMON_ENV}/bin/python"
DATA_ROOT="${DATA_ROOT:-/path/to/data/mps/egodelta_benchmark_data}"
RUNS="${RUNS:-${REPO_ROOT}/baseline/scene_change/runs}"
CACHE_ROOT="${CACHE_ROOT:-${RUNS}/models/egodelta/stage2_cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUNS}/predictions/egodelta}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/annotation_runs/wacv2027_masks_12views/manifest.json}"
# The final benchmark uses every fixed query. Set ANNOTATOR to a concrete ID
# only for an explicitly scoped diagnostic run.
ANNOTATOR="${ANNOTATOR:-all}"
export PYTHONPATH="${REPO_ROOT}/baseline/scene_change:${REPO_ROOT}"
test -x "${PYTHON}"

if [[ $# -gt 0 ]]; then
  SCENES=("$@")
else
  mapfile -t SCENES < <("${PYTHON}" -c '
from egoscene.cli import DEFAULT_DATA, DEFAULT_MANIFEST
from egoscene.benchmark import discover_scenes
for x in discover_scenes(DEFAULT_MANIFEST, pairs_root=DEFAULT_DATA/"pairs_fast", pre_models_root=DEFAULT_DATA/"pre_models", post_models_root=DEFAULT_DATA/"post_models"):
    print(x.scene_id)
')
fi

mkdir -p "${CACHE_ROOT}" "${OUTPUT_ROOT}"
ANNOTATOR_ARGS=()
if [[ -n "${ANNOTATOR}" && "${ANNOTATOR}" != "all" ]]; then
  ANNOTATOR_ARGS=(--annotator "${ANNOTATOR}")
fi
for INDEX in "${!SCENES[@]}"; do
  SCENE="${SCENES[$INDEX]}"
  echo "[$((INDEX + 1))/${#SCENES[@]}] egodelta: ${SCENE}"
  if [[ -f "${OUTPUT_ROOT}/${SCENE}/metadata.json" ]]; then
    echo "  complete metadata exists; skipping"
    continue
  fi
  if [[ ! -f "${CACHE_ROOT}/${SCENE}.npz" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
      "${REPO_ROOT}/target_extraction/old/code/scripts/run_eval.py" cache \
      --pre-root "${DATA_ROOT}/pre_models" \
      --post-root "${DATA_ROOT}/post_models" \
      --config "${REPO_ROOT}/target_extraction/old/code/configs/objectness_full.yaml" \
      --cache-root "${CACHE_ROOT}" \
      --device cuda:0 \
      --scene "${SCENE}" \
      --skip-existing
  fi
  "${PYTHON}" -m egoscene.cli egodelta \
    --manifest "${MANIFEST}" \
    --data-root "${DATA_ROOT}" \
    --scene "${SCENE}" \
    --cache-root "${CACHE_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    "${ANNOTATOR_ARGS[@]}"
done

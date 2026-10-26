#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 {gsdiff|mv3dcd|3dgscd} GPU [SCENE_ID ...]" >&2
  exit 2
fi

METHOD="$1"
GPU="$2"
shift 2
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${REPO_ROOT}/baseline/scene_change/scripts/env_paths.sh"
PYTHON="${EGOSCENE_COMMON_ENV}/bin/python"
RUNS="${REPO_ROOT}/baseline/scene_change/runs"
POST_VIEWS="${POST_VIEWS:-10}"
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

for INDEX in "${!SCENES[@]}"; do
  SCENE="${SCENES[$INDEX]}"
  echo "[$((INDEX + 1))/${#SCENES[@]}] ${METHOD}: ${SCENE}"
  if [[ -f "${RUNS}/predictions/${METHOD}/${SCENE}/metadata.json" ]]; then
    echo "  complete metadata exists; skipping"
    continue
  fi
  case "${METHOD}" in
    gsdiff)
      CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m egoscene.cli gsdiff \
        --scene "${SCENE}" --output-root "${RUNS}/predictions/gsdiff"
      ;;
    mv3dcd)
      "${PYTHON}" -m egoscene.cli prepare --baseline mv3dcd --scene "${SCENE}" \
        --output-root "${RUNS}/data/mv3dcd" --post-views "${POST_VIEWS}"
      bash "${REPO_ROOT}/baseline/scene_change/scripts/run_mv3dcd_scene.sh" \
        "${SCENE}" "${RUNS}/data/mv3dcd" "${RUNS}/models/mv3dcd" \
        "${RUNS}/predictions/mv3dcd" "${GPU}"
      ;;
    3dgscd)
      "${PYTHON}" -m egoscene.cli prepare --baseline 3dgscd --scene "${SCENE}" \
        --output-root "${RUNS}/data/3dgscd" --post-views "${POST_VIEWS}"
      bash "${REPO_ROOT}/baseline/scene_change/scripts/run_3dgscd_scene.sh" \
        "${SCENE}" "${RUNS}/data/3dgscd" "${RUNS}/models/3dgscd" \
        "${RUNS}/predictions/3dgscd" "${GPU}"
      ;;
    *)
      echo "Unknown method: ${METHOD}" >&2
      exit 2
      ;;
  esac
done

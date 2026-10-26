#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 {3dgscd|gsdiff|mv3dcd|egodelta} GPU SHARD_INDEX SHARD_COUNT" >&2
  exit 2
fi

METHOD="$1"
GPU="$2"
SHARD_INDEX="$3"
SHARD_COUNT="$4"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${REPO_ROOT}/baseline/scene_change/scripts/env_paths.sh"
PYTHON="${EGOSCENE_COMMON_ENV}/bin/python"

if (( SHARD_COUNT < 1 || SHARD_INDEX < 0 || SHARD_INDEX >= SHARD_COUNT )); then
  echo "Invalid shard ${SHARD_INDEX}/${SHARD_COUNT}" >&2
  exit 2
fi

mapfile -t ALL_SCENES < <("${PYTHON}" -c '
from egoscene.cli import DEFAULT_DATA, DEFAULT_MANIFEST
from egoscene.benchmark import discover_scenes
for scene in discover_scenes(
    DEFAULT_MANIFEST,
    pairs_root=DEFAULT_DATA / "pairs_fast",
    pre_models_root=DEFAULT_DATA / "pre_models",
    post_models_root=DEFAULT_DATA / "post_models",
):
    print(scene.scene_id)
')

SCENES=()
for INDEX in "${!ALL_SCENES[@]}"; do
  if (( INDEX % SHARD_COUNT == SHARD_INDEX )); then
    SCENES+=("${ALL_SCENES[$INDEX]}")
  fi
done

echo "${METHOD}: shard ${SHARD_INDEX}/${SHARD_COUNT}; ${#SCENES[@]} scenes; GPU ${GPU}"
if [[ "${METHOD}" == "egodelta" ]]; then
  exec bash "${REPO_ROOT}/baseline/scene_change/scripts/run_egodelta_scenes.sh" \
    "${GPU}" "${SCENES[@]}"
fi
exec bash "${REPO_ROOT}/baseline/scene_change/scripts/run_all_scenes.sh" \
  "${METHOD}" "${GPU}" "${SCENES[@]}"

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXTERNAL_ROOT="${REPO_ROOT}/.external"
mkdir -p "${EXTERNAL_ROOT}"

clone_at() {
  local name="$1"
  local url="$2"
  local revision="$3"
  if [[ ! -d "${EXTERNAL_ROOT}/${name}/.git" ]]; then
    git clone --recursive "${url}" "${EXTERNAL_ROOT}/${name}"
  fi
  git -C "${EXTERNAL_ROOT}/${name}" fetch --all --tags
  git -C "${EXTERNAL_ROOT}/${name}" checkout --detach "${revision}"
  git -C "${EXTERNAL_ROOT}/${name}" submodule update --init --recursive
  test "$(git -C "${EXTERNAL_ROOT}/${name}" rev-parse HEAD)" = "${revision}"
}

clone_at "3DGS-CD" "https://github.com/520xyxyzq/3DGS-CD.git" "5ca372651e8426b9652ce757671a914be93590b4"
clone_at "MV3DCD" "https://github.com/Chumsy0725/MV3DCD.git" "b45606c83ee277207b2f73be8fda366760b12921"
clone_at "GS-DIFF" "https://github.com/Chumsy0725/GS-DIFF.git" "b3e4c596d84ea86f8aad4d115d1a52a41f35561d"
clone_at "PGSR" "https://github.com/zju3dv/PGSR.git" "de24f1a38b350387e8d8fe381b2cd70c1ae946e7"
clone_at "EfficientSAM" "https://github.com/yformer/EfficientSAM.git" "c9408a74b1db85e7831977c66e9462c6f4891729"

# The official function accepts refine_pose=False but leaves cams_eval
# undefined on that path. Apply the minimal control-flow fix needed to use
# EgoDelta's known Aria MPS evaluation poses without reading evaluation RGB.
for PATCH_NAME in 3dgscd_known_eval_pose.patch 3dgscd_empty_image_diff.patch; do
  PATCH_PATH="${REPO_ROOT}/baseline/scene_change/patches/${PATCH_NAME}"
  if git -C "${EXTERNAL_ROOT}/3DGS-CD" apply --reverse --check "${PATCH_PATH}" 2>/dev/null; then
    : # already applied
  else
    git -C "${EXTERNAL_ROOT}/3DGS-CD" apply --check "${PATCH_PATH}"
    git -C "${EXTERNAL_ROOT}/3DGS-CD" apply "${PATCH_PATH}"
  fi
done

if [[ ! -f "${EXTERNAL_ROOT}/EfficientSAM/weights/efficient_sam_vits.pt" ]]; then
  unzip -o "${EXTERNAL_ROOT}/EfficientSAM/weights/efficient_sam_vits.pt.zip" -d "${EXTERNAL_ROOT}/EfficientSAM/weights"
fi
if [[ ! -e "${EXTERNAL_ROOT}/3DGS-CD/weights" ]]; then
  ln -s "${EXTERNAL_ROOT}/EfficientSAM/weights" "${EXTERNAL_ROOT}/3DGS-CD/weights"
fi

echo "Pinned upstream repositories are ready under ${EXTERNAL_ROOT}"

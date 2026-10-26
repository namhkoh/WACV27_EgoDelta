#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${REPO_ROOT}/baseline/scene_change/scripts/env_paths.sh"
COMMON_ENV="${EGOSCENE_COMMON_ENV}"
GSCD_ENV="${EGOSCENE_3DGSCD_ENV}"
PGSR_ENV="${EGOSCENE_PGSR_ENV}"
mkdir -p "${EGOSCENE_ENV_ROOT}"
EGOSCENE_ORIGINAL_PATH="${PATH}"
EGOSCENE_ORIGINAL_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

use_build_environment() {
  local environment="$1"
  export CUDA_HOME="${environment}"
  export PATH="${environment}/bin:${EGOSCENE_ORIGINAL_PATH}"
  export LD_LIBRARY_PATH="${environment}/lib:${environment}/targets/x86_64-linux/lib:${EGOSCENE_ORIGINAL_LD_LIBRARY_PATH}"
}

setup_common() {
  if [[ ! -x "${COMMON_ENV}/bin/python" ]]; then
    conda create -y -p "${COMMON_ENV}" python=3.10 pip=24.2
  fi
  conda install -y -p "${COMMON_ENV}" -c "nvidia/label/cuda-12.4.0" cuda-toolkit
  use_build_environment "${COMMON_ENV}"
  "${COMMON_ENV}/bin/pip" install \
    torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124
  "${COMMON_ENV}/bin/pip" install \
    "git+https://github.com/nerfstudio-project/gsplat.git@4d3a3b69db4de0326f983ccf7b7b255271a17b01" \
    "numpy<2" "opencv-python<4.12" ninja plyfile timm matplotlib \
    scikit-learn torchmetrics scipy pillow pytest
  "${COMMON_ENV}/bin/pip" install "${REPO_ROOT}/.external/MV3DCD/submodules/simple-knn"
  "${COMMON_ENV}/bin/pip" install -e "${REPO_ROOT}/baseline/scene_change"
}

setup_3dgscd() {
  if [[ ! -x "${GSCD_ENV}/bin/python" ]]; then
    conda create -y -p "${GSCD_ENV}" python=3.8 pip=23.3.1
  fi
  conda install -y -p "${GSCD_ENV}" -c "nvidia/label/cuda-11.8.0" cuda-toolkit
  use_build_environment "${GSCD_ENV}"
  "${GSCD_ENV}/bin/pip" install \
    torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118
  "${GSCD_ENV}/bin/pip" install --upgrade setuptools wheel ninja
  "${GSCD_ENV}/bin/pip" install \
    "git+https://github.com/NVlabs/tiny-cuda-nn.git@749dd70c5afc5a9dadb85e5652ed65d55e0ba187#subdirectory=bindings/torch"
  "${GSCD_ENV}/bin/pip" install -e "${REPO_ROOT}/.external/3DGS-CD"
  "${GSCD_ENV}/bin/pip" install -e "${REPO_ROOT}/.external/EfficientSAM"
  # Keep these as separate installs, matching the upstream instructions.  HLoc
  # declares an unpinned LightGlue VCS dependency; asking pip to resolve that
  # together with our pinned revision produces a false ResolutionImpossible.
  "${GSCD_ENV}/bin/pip" install \
    "git+https://github.com/cvg/Hierarchical-Localization.git@73a3cb0f59659306eb6c15c7213137b2196c5ceb"
  "${GSCD_ENV}/bin/pip" install pycolmap==0.4.0
  "${GSCD_ENV}/bin/pip" install --no-deps --force-reinstall \
    "git+https://github.com/cvg/LightGlue@035612541779b17897aa06d6ff19cb4060111616"
}

setup_pgsr() {
  if [[ ! -x "${PGSR_ENV}/bin/python" ]]; then
    conda create -y -p "${PGSR_ENV}" python=3.8 pip=23.3.1
  fi
  conda install -y -p "${PGSR_ENV}" -c "nvidia/label/cuda-11.8.0" cuda-toolkit
  use_build_environment "${PGSR_ENV}"
  "${PGSR_ENV}/bin/pip" install \
    torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118
  "${PGSR_ENV}/bin/pip" install --upgrade setuptools wheel ninja
  "${PGSR_ENV}/bin/pip" install "numpy<2" plyfile opencv-python tqdm pillow
  "${PGSR_ENV}/bin/pip" install \
    "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
  "${PGSR_ENV}/bin/pip" install "${REPO_ROOT}/.external/PGSR/submodules/diff-plane-rasterization"
  "${PGSR_ENV}/bin/pip" install "${REPO_ROOT}/.external/PGSR/submodules/simple-knn"
  # PGSR's official environment is Python 3.8, whereas the independent
  # egoscene adapter uses Python >=3.10.  Training imports only PGSR and its
  # CUDA extensions, so deliberately keep the environments isolated.
}

case "${1:-all}" in
  common) setup_common ;;
  3dgscd) setup_3dgscd ;;
  pgsr) setup_pgsr ;;
  all) setup_common; setup_3dgscd; setup_pgsr ;;
  *) echo "Usage: $0 [common|3dgscd|pgsr|all]" >&2; exit 2 ;;
esac

echo "Environment setup complete."

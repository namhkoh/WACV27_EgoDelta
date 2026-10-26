#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-sam3_aria}"
SAM3_ROOT="${SAM3_ROOT:-/path/to/workspace/sam3}"
SAM3_DIR="${SAM3_DIR:-${SAM3_ROOT}/repo}"
HF_HOME="${HF_HOME:-${SAM3_ROOT}/hf_home}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE

eval "$(conda shell.bash hook)"
conda create -n "${ENV_NAME}" python=3.12 -y
conda activate "${ENV_NAME}"

if [[ -n "${TORCH_INSTALL_CMD:-}" ]]; then
  eval "${TORCH_INSTALL_CMD}"
elif command -v nvidia-smi >/dev/null 2>&1; then
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
  python -m pip install torch torchvision
fi

# The current SAM 3 package imports pkg_resources from setuptools, and imports
# einops at runtime even though it is listed as an optional dependency upstream.
# SAM 3 also declares numpy<2; keep that constraint after installing project deps.
python -m pip install "setuptools<81" "numpy>=1.26,<2" einops pycocotools

mkdir -p "$(dirname "${SAM3_DIR}")" "${HF_HOME}" "${HF_HUB_CACHE}"
if [[ ! -d "${SAM3_DIR}/.git" ]]; then
  git clone https://github.com/facebookresearch/sam3.git "${SAM3_DIR}"
fi

python -m pip install -e "${SAM3_DIR}"
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install "setuptools<81" "numpy>=1.26,<2" einops pycocotools

cat <<'MSG'
Native SAM 3 environment is ready.
SAM 3 repo path:
  ${SAM3_DIR}
Hugging Face cache:
  ${HF_HUB_CACHE}
You may need Hugging Face authentication and SAM 3 checkpoint access before inference:
  huggingface-cli login
MSG

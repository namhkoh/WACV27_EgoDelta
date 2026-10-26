#!/usr/bin/env bash
# Source this before running native SAM 3 inference to keep large checkpoints
# and Hugging Face cache files on the external disk.

export SAM3_ROOT="${SAM3_ROOT:-/path/to/workspace/sam3}"
export SAM3_DIR="${SAM3_DIR:-${SAM3_ROOT}/repo}"
export HF_HOME="${HF_HOME:-${SAM3_ROOT}/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"

mkdir -p "${SAM3_ROOT}" "${HF_HOME}" "${HF_HUB_CACHE}"


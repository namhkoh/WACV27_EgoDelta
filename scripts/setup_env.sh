#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${EGO_DELTA_ENV:-surfel_splatting}"
ENV_FILE="${EGO_DELTA_ENV_FILE:-environment.yml}"

if [ -f .gitmodules ]; then
    git submodule update --init --recursive
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required. Install Miniconda or Mambaforge, then rerun this script." >&2
    exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
    conda env create -f "$ENV_FILE"
fi

conda activate "$ENV_NAME"

# Build CUDA extensions from the pinned submodule sources for this machine.
python -m pip install submodules/diff-surfel-rasterization
python -m pip install submodules/simple-knn

python - <<'PY'
import importlib

importlib.import_module("diff_surfel_rasterization")
importlib.import_module("simple_knn._C")
importlib.import_module("projectaria_tools")
print("EgoDelta environment is ready.")
PY

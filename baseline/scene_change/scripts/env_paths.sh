#!/usr/bin/env bash

# Keep large Conda environments off the benchmark data mount. Override this
# location when needed, e.g. EGOSCENE_ENV_ROOT=/fast/local/disk/envs.
EGOSCENE_ENV_ROOT="${EGOSCENE_ENV_ROOT:-${HOME}/.cache/egodelta-scene-change/envs}"
EGOSCENE_COMMON_ENV="${EGOSCENE_ENV_ROOT}/scenechange"
EGOSCENE_3DGSCD_ENV="${EGOSCENE_ENV_ROOT}/3dgscd"
EGOSCENE_PGSR_ENV="${EGOSCENE_ENV_ROOT}/pgsr"

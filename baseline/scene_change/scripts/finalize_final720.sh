#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${REPO_ROOT}/baseline/scene_change/scripts/env_paths.sh"
PYTHON="${EGOSCENE_COMMON_ENV}/bin/python"
EXPECTED_SCENES="${EXPECTED_SCENES:-60}"
POLL_SECONDS="${POLL_SECONDS:-60}"
GT_EXPORT="${GT_EXPORT:-${REPO_ROOT}/annotation_runs/image_masks_gt_wacv2027_human_final_720_20260817}"
RUNS="${REPO_ROOT}/baseline/scene_change/runs"
EVALUATIONS="${RUNS}/evaluations/final_720_20260817"
FIGURES="${RUNS}/figures/final_720_20260817"
THREEDGSCD="${RUNS}/predictions/3dgscd"
MV3DCD="${RUNS}/predictions/mv3dcd"
GSDIFF="${RUNS}/predictions/gsdiff"
EGODELTA="${RUNS}/final_720_20260817/predictions/egodelta"
export PYTHONPATH="${REPO_ROOT}/baseline/scene_change:${REPO_ROOT}/annotation_tool:${REPO_ROOT}:${PYTHONPATH:-}"

while true; do
  COMPLETE="$(find "${THREEDGSCD}" -mindepth 2 -maxdepth 2 -type f -name metadata.json | wc -l)"
  printf '%s 3DGS-CD scene coverage: %s/%s\n' "$(date -Is)" "${COMPLETE}" "${EXPECTED_SCENES}"
  if (( COMPLETE >= EXPECTED_SCENES )); then
    break
  fi
  sleep "${POLL_SECONDS}"
done

mkdir -p "${EVALUATIONS}" "${FIGURES}"
for METHOD in 3dgscd mv3dcd gsdiff egodelta; do
  case "${METHOD}" in
    3dgscd) ROOT="${THREEDGSCD}" ;;
    mv3dcd) ROOT="${MV3DCD}" ;;
    gsdiff) ROOT="${GSDIFF}" ;;
    egodelta) ROOT="${EGODELTA}" ;;
  esac
  "${PYTHON}" -m egoscene.cli evaluate \
    --gt-export "${GT_EXPORT}" \
    --predictions-root "${ROOT}" \
    --output-json "${EVALUATIONS}/${METHOD}.json" \
    --output-csv "${EVALUATIONS}/${METHOD}.csv"
done

"${PYTHON}" "${REPO_ROOT}/baseline/scene_change/scripts/summarize_results.py" \
  --evaluation-dir "${EVALUATIONS}" \
  --output-json "${EVALUATIONS}/summary.json" \
  --output-csv "${EVALUATIONS}/summary.csv"

"${PYTHON}" "${REPO_ROOT}/baseline/scene_change/scripts/make_figure3.py" \
  --gt-export "${GT_EXPORT}" \
  --3dgscd-root "${THREEDGSCD}" \
  --mv3dcd-root "${MV3DCD}" \
  --gsdiff-root "${GSDIFF}" \
  --egodelta-root "${EGODELTA}" \
  --output-stem "${FIGURES}/figure3_comparison"

(
  cd "${RUNS}"
  sha256sum \
    evaluations/final_720_20260817/*.json \
    evaluations/final_720_20260817/*.csv \
    figures/final_720_20260817/figure3_comparison.* \
    > final_720_20260817/artifacts.sha256
)
printf '%s final evaluation and qualitative figure complete\n' "$(date -Is)"

#!/usr/bin/env bash
set -euo pipefail

SHARD_INDEX="${1:?usage: run_condor_eval_shard.sh SHARD_INDEX}"

PROJECT_ROOT="/home/user/plant_yolo"
PYTHON_BIN="/home/user/miniconda3/envs/aria-tomato-change/bin/python"
OUT_ROOT="/path/to/data/plant_yolo/eval_pairs_fast_yoloe26s_allframes_imgsz720_sharded"

cd "${PROJECT_ROOT}"

echo "host=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "shard_index=${SHARD_INDEX}"
echo "assigned_gpus=${_CONDOR_AssignedGPUs:-unset}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

"${PYTHON_BIN}" scripts/run_eval_benchmark.py \
  --labels /path/to/data/mps/egodelta_benchmark_splitted/3d_points_labels.txt \
  --full-root /path/to/data/mps/egodelta_benchmark \
  --pairs-root /path/to/data/mps/egodelta_benchmark_data/pairs_fast \
  --out "${OUT_ROOT}/shard_${SHARD_INDEX}" \
  --tasks harvesting,deleafing \
  --yoloe-model yoloe-26s-seg.pt \
  --frame-stride 1 \
  --harvest-imgsz 720 \
  --deleaf-imgsz 720 \
  --max-det 300 \
  --segmentation-batch-size 32 \
  --visualization-stride 30 \
  --num-shards 10 \
  --shard-index "${SHARD_INDEX}" \
  --skip-coordinate-check

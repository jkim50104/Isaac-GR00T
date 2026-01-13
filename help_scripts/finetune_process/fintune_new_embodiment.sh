#!/usr/bin/env bash
set -euo pipefail

SERVER="$(hostname -s)"   # or: hostname

# Defaults (fallback)
NUM_GPUS=1
BATCH_SIZE=128
ARM_ONLY=true
USE_WRIST=true
ACTION_REP=REL

case "${SERVER}" in
  *pearl*)
    NUM_GPUS=4
    BATCH_SIZE=512
    ARM_ONLY=false
    USE_WRIST=false
    ACTION_REP=REL
    ulimit -n 65535
    ;;
  *turing*)
    NUM_GPUS=4
    BATCH_SIZE=256
    ARM_ONLY=true
    USE_WRIST=true
    ACTION_REP=REL
    ;;
  *lunar*)
    NUM_GPUS=1
    BATCH_SIZE=128
    ARM_ONLY=true
    USE_WRIST=true
    ACTION_REP=ABS
    ;;
  *)
    echo "Unknown server hostname '${SERVER}', using defaults."
    ;;
esac

BASE_MODEL="nvidia/GR00T-N1.6-3B"
DATASET_NAME="ffw_sg2_rev1_base_drive_test"
DATASET_PATH="./data/jkim50104/$DATASET_NAME" # ffw_sg2_rev1_clear_item, ffw_sg2_rev1_base_drive_test
EMBODIMENT_TAG="NEW_EMBODIMENT"

CONFIG="ai_worker_config.py"
HYPER_PARAMS="G${NUM_GPUS}_B${BATCH_SIZE}_${ACTION_REP}"

if [[ "${ARM_ONLY}" == "true" ]]; then
  CONFIG="ai_worker_arm_only_config.py"
  HYPER_PARAMS="${HYPER_PARAMS}_AO"
fi

if [[ "${USE_WRIST}" == "true" ]]; then
  HYPER_PARAMS="${HYPER_PARAMS}_WR"
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <cuda_visible_devices>"
  echo "Example: $0 0"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$1"

# python 
torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \
  gr00t/experiment/launch_finetune.py \
  --base-model-path "${BASE_MODEL}" \
  --dataset-path "${DATASET_PATH}" \
  --embodiment-tag "${EMBODIMENT_TAG}" \
  --modality-config-path "./help_scripts/data_config/${CONFIG}" \
  --num-gpus "${NUM_GPUS}" \
  --output-dir "./output/$DATASET_NAME/${HYPER_PARAMS}" \
  --save-total-limit 3 \
  --save-steps 5000 \
  --max-steps 10000 \
  --use-wandb \
  --global-batch-size "${BATCH_SIZE}" \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --dataloader-num-workers 4

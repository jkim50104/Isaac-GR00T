#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

SERVER="$(hostname -s)"

# Global finetune settings
USE_WRIST_VIEW=true
ARM_ONLY=true
ACTION_REP=REL  # "ABS" or "REL"

# ---- args ----
DATASET_PREFIX="ffw_sg2_rev1_"
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <task_name> [cuda_visible_devices]"
  echo "Example: $0 pick_item          # GPUs default to 0,1,2,3"
  echo "Example: $0 pick_item 0        # single GPU"
  exit 1
fi
DATASET_NAME="${DATASET_PREFIX}$1"
CUDA_DEVICES="${2:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

# Count devices from comma-separated list (pure bash)
IFS=',' read -r -a DEV_ARR <<< "${CUDA_DEVICES}"
DEVICE_COUNT="${#DEV_ARR[@]}"

# Defaults by server (can still set batch size etc.)
case "${SERVER}" in
  *pearl*)
    ulimit -n 65535 || true
    BATCH_SIZE=512
    ;;
  *turing*|*rosenblatt*)
    BATCH_SIZE=256
    ;;
  *lunar*)
    BATCH_SIZE=128
    ;;
  *)
    echo "Unknown server hostname '${SERVER}', using defaults."
    BATCH_SIZE=128
    ;;
esac

# Use the number of visible devices for torchrun
NUM_GPUS="${DEVICE_COUNT}"

# If USE_WRIST_VIEW is false, scale batch size by 1.5x
if [[ "${USE_WRIST_VIEW}" == "false" ]]; then
  BATCH_SIZE=$(( BATCH_SIZE * 9 / 5 ))   # 1.8x (floors)
fi

# Make sure BATCH_SIZE is divisible by NUM_GPUS (round down to nearest multiple)
if (( BATCH_SIZE % NUM_GPUS != 0 )); then
  OLD_BS="${BATCH_SIZE}"
  BATCH_SIZE=$(( (BATCH_SIZE / NUM_GPUS) * NUM_GPUS ))
  if (( BATCH_SIZE == 0 )); then
    echo "ERROR: Batch size became 0 after making it divisible by NUM_GPUS=${NUM_GPUS} (old=${OLD_BS})."
    exit 1
  fi
  echo "WARN: Adjusted BATCH_SIZE ${OLD_BS} -> ${BATCH_SIZE} to be divisible by NUM_GPUS=${NUM_GPUS}"
fi


BASE_MODEL="nvidia/GR00T-N1.6-3B"
DATASET_PATH="./data/jkim50104/$DATASET_NAME"

if [[ ! -d "${DATASET_PATH}" ]]; then
  echo "ERROR: Dataset not found: ${DATASET_PATH}"
  echo "Available datasets:"
  ls -1 ./data/jkim50104/ 2>/dev/null | grep "^${DATASET_PREFIX}" | sed "s/^${DATASET_PREFIX}/  /"
  exit 1
fi
EMBODIMENT_TAG="NEW_EMBODIMENT"

CONFIG="ai_worker_config.py"
HYPER_PARAMS="G${NUM_GPUS}_B${BATCH_SIZE}_${ACTION_REP}"

if [[ "${ARM_ONLY}" == "true" ]]; then
  HYPER_PARAMS="${HYPER_PARAMS}_AO"
fi
if [[ "${USE_WRIST_VIEW}" == "true" ]]; then
  HYPER_PARAMS="${HYPER_PARAMS}_WR"
fi

# make flags visible to all torchrun ranks
export GR00T_ARM_ONLY="${ARM_ONLY}"
export GR00T_USE_WRIST_VIEW="${USE_WRIST_VIEW}"
export GR00T_ACTION_REP="${ACTION_REP}"

# Config check
bash help_scripts/data_config/config_check.sh "./help_scripts/data_config/${CONFIG}" --strict

# Auto-link modality.json if missing
if [[ ! -f "${DATASET_PATH}/meta/modality.json" ]]; then
  echo "[INFO] modality.json not found in ${DATASET_PATH}/meta/, running link_modality.py ..."
  python help_scripts/data_config/link_modality.py
fi

echo "================= FINETUNE NEW EMBODIMENT ================="
echo "SERVER=${SERVER}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NUM_GPUS=${NUM_GPUS}"
REF_BATCH=256
REF_MAX_STEPS=30000
REF_SAVE_STEPS=5000
MAX_STEPS=$(( REF_MAX_STEPS * REF_BATCH / BATCH_SIZE ))
SAVE_STEPS=$(( REF_SAVE_STEPS * REF_BATCH / BATCH_SIZE ))

echo "GLOBAL_BATCH_SIZE=${BATCH_SIZE}"
echo "MAX_STEPS=${MAX_STEPS}  SAVE_STEPS=${SAVE_STEPS}"
echo "OUT=./output/${DATASET_NAME}/${HYPER_PARAMS}"

torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" \
  gr00t/experiment/launch_finetune.py \
  --base-model-path "${BASE_MODEL}" \
  --dataset-path "${DATASET_PATH}" \
  --embodiment-tag "${EMBODIMENT_TAG}" \
  --modality-config-path "./help_scripts/data_config/${CONFIG}" \
  --num-gpus "${NUM_GPUS}" \
  --output-dir "./output/${DATASET_NAME}/${HYPER_PARAMS}" \
  --save-total-limit 2 \
  --save-steps "${SAVE_STEPS}" \
  --max-steps "${MAX_STEPS}" \
  --use-wandb \
  --global-batch-size "${BATCH_SIZE}" \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --dataloader-num-workers 4

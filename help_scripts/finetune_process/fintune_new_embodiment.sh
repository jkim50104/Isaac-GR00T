#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

SERVER="$(hostname -s)"

# Global finetune settings
USE_WRIST_VIEW=true
ARM_ONLY=false
ACTION_REP=REL  # "ABS" or "REL"

# ---- args ----
SIM_MODE=false
DEBUG_MODE=false
POSITIONAL=()
for arg in "$@"; do
  case "$arg" in
    --sim)   SIM_MODE=true ;;
    --debug) DEBUG_MODE=true ;;
    *)       POSITIONAL+=("$arg") ;;
  esac
done
set -- "${POSITIONAL[@]}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 [--sim] <task_name> [cuda_visible_devices]"
  echo "Example: $0 pick_item               # real data, GPUs default to 0,1,2,3"
  echo "Example: $0 pick_item 0             # single GPU"
  echo "Example: $0 --sim table_pnp         # sim data (ACS_ROBI)"
  exit 1
fi

if [[ "${SIM_MODE}" == "true" ]]; then
  DATASET_PREFIX="sim_ffw_sg2_"
  DATA_ROOT="./data/ACS_ROBI"
else
  DATASET_PREFIX="ffw_sg2_rev1_"
  DATA_ROOT="./data/jkim50104"
fi

TASK_NAME="$1"
DATASET_NAME="${DATASET_PREFIX}${TASK_NAME}"
CUDA_DEVICES="${2:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

# Count devices from comma-separated list (pure bash)
IFS=',' read -r -a DEV_ARR <<< "${CUDA_DEVICES}"
DEVICE_COUNT="${#DEV_ARR[@]}"

# Per-GPU batch calibrated to GPU VRAM (baseline: with wrist view)
case "${SERVER}" in
  *pearl*)
    ulimit -n 65535 || true
    GPU_VRAM=82
    PER_GPU_BATCH=336   # 4x NVIDIA A100 ~82GB
    ;;
  *turing*|*rosenblatt*)
    GPU_VRAM=50
    PER_GPU_BATCH=80    # 4x NVIDIA RTX A6000 ~50GB
    ;;
  *lunar*)
    GPU_VRAM=98
    PER_GPU_BATCH=360   # 1x NVIDIA RTX PRO 6000 Blackwell ~98GB
    ;;
  *)
    echo "Unknown server '${SERVER}', using conservative defaults."
    GPU_VRAM=50
    PER_GPU_BATCH=64
    ;;
esac

NUM_GPUS="${DEVICE_COUNT}"

# Scale per-GPU batch when no wrist view (one fewer camera = less memory per sample)
if [[ "${USE_WRIST_VIEW}" == "false" ]]; then
  PER_GPU_BATCH=$(( PER_GPU_BATCH * 9 / 5 ))
fi

# Global batch scales with GPU count; steps scale inversely — total samples seen stays constant
BATCH_SIZE=$(( PER_GPU_BATCH * NUM_GPUS ))
TOTAL_SAMPLES=7680000   # fixed training budget (30000 steps × 256 ref batch)
if [[ "${DEBUG_MODE}" == "true" ]]; then
  TOTAL_SAMPLES=$(( TOTAL_SAMPLES * 10 ))
fi
N_CHECKPOINTS=6
SAVE_STEPS=$(( TOTAL_SAMPLES / BATCH_SIZE / N_CHECKPOINTS ))
MAX_STEPS=$(( SAVE_STEPS * N_CHECKPOINTS ))


BASE_MODEL="nvidia/GR00T-N1.7-3B"
DATASET_PATH="${DATA_ROOT}/${DATASET_NAME}"

if [[ ! -d "${DATASET_PATH}" ]]; then
  echo "ERROR: Dataset not found: ${DATASET_PATH}"
  echo "Available datasets:"
  ls -1 "${DATA_ROOT}/" 2>/dev/null | grep "^${DATASET_PREFIX}" | sed "s/^${DATASET_PREFIX}/  /"
  exit 1
fi
EMBODIMENT_TAG="NEW_EMBODIMENT"

CONFIG="ai_worker_config.py"
RUN_MODE="$( [[ "${SIM_MODE}" == "true" ]] && echo "sim" || echo "real" )"
DATE="$(date +%Y%m%d)"
HYPER_PARAMS="B${BATCH_SIZE}_${ACTION_REP}"
if [[ "${ARM_ONLY}" == "true" ]]; then
  HYPER_PARAMS="${HYPER_PARAMS}_AO"
fi
if [[ "${USE_WRIST_VIEW}" == "true" ]]; then
  HYPER_PARAMS="${HYPER_PARAMS}_WR"
fi
if [[ "${DEBUG_MODE}" == "true" ]]; then
  HYPER_PARAMS="DEBUG_${HYPER_PARAMS}"
fi
OUTPUT_DIR="./output/v1.7/${DATASET_NAME}/${DATE}_${HYPER_PARAMS}"

# make flags visible to all torchrun ranks
export GR00T_ARM_ONLY="${ARM_ONLY}"
export GR00T_USE_WRIST_VIEW="${USE_WRIST_VIEW}"
export GR00T_ACTION_REP="${ACTION_REP}"

# Config check
bash help_scripts/data_config/config_check.sh "./help_scripts/data_config/${CONFIG}" --strict

# Auto-link modality.json if missing
MODALITY_LINK="${DATASET_PATH}/meta/modality.json"
MODALITY_TARGET="$(pwd)/help_scripts/data_config/ai_worker_modality.json"
if [[ ! -f "${MODALITY_LINK}" ]]; then
  echo "[INFO] modality.json not found in ${DATASET_PATH}/meta/, creating symlink ..."
  ln -sf "${MODALITY_TARGET}" "${MODALITY_LINK}"
fi

echo "================= FINETUNE NEW EMBODIMENT ================="
echo "SERVER=${SERVER}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NUM_GPUS=${NUM_GPUS}  GPU_VRAM=${GPU_VRAM}GB  PER_GPU_BATCH=${PER_GPU_BATCH}"
echo "GLOBAL_BATCH_SIZE=${BATCH_SIZE}  TOTAL_SAMPLES=${TOTAL_SAMPLES}"
echo "MAX_STEPS=${MAX_STEPS}  SAVE_STEPS=${SAVE_STEPS}"
echo "OUT=${OUTPUT_DIR}"

torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" \
  gr00t/experiment/launch_finetune.py \
  --base-model-path "${BASE_MODEL}" \
  --dataset-path "${DATASET_PATH}" \
  --embodiment-tag "${EMBODIMENT_TAG}" \
  --modality-config-path "./help_scripts/data_config/${CONFIG}" \
  --num-gpus "${NUM_GPUS}" \
  --output-dir "${OUTPUT_DIR}" \
  --save-total-limit 2 \
  --save-steps "${SAVE_STEPS}" \
  --max-steps "${MAX_STEPS}" \
  --use-wandb \
  --global-batch-size "${BATCH_SIZE}" \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --dataloader-num-workers 4

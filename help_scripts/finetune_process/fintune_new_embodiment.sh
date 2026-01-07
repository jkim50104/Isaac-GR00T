#!/bin/bash

BATCH_SIZE=160
ARM_ONLY=true

# Default values
HYPER_PARAMS="B${BATCH_SIZE}"
CONFIG="ai_worker_config.py"

# Apply ARM_ONLY overrides
if [ "$ARM_ONLY" = true ]; then
  HYPER_PARAMS="B${BATCH_SIZE}_AO"
  CONFIG="ai_worker_config_arm_only.py"
fi

# Run finetuning process with the new embodiment tag
# Configure for single GPU
export NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=$1

python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path ./data/jkim50104/ffw_sg2_rev1_clear_item \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path ./data/jkim50104/$CONFIG \
    --num-gpus $NUM_GPUS \
    --output-dir ./output/ai_worker_${HYPER_PARAMS}_REL_LUNAR \
    --save-total-limit 5 \
    --save-steps 5000 \
    --max-steps 20000 \
    --use-wandb \
    --global-batch-size $BATCH_SIZE \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 4

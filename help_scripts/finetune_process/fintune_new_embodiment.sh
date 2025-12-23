#!/bin/bash

# Run finetuning process with the new embodiment tag
# Configure for single GPU
export NUM_GPUS=2
export CUDA_VISIBLE_DEVICES="2,3"
python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path ./data/ai_worker/ffw_sg2_rev1_clear_item \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path ./data/ai_worker/ai_worker_config.py \
    --num-gpus $NUM_GPUS \
    --output-dir ./output/ai_worker_160BT_2GPU \
    --save-total-limit 5 \
    --save-steps 2000 \
    --max-steps 10000 \
    --use-wandb \
    --global-batch-size 160 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 4
# Deploy robot
CONFIG_PATH="help_scripts/data_config/ai_worker_arm_only_config.py" # ai_worker_config, ai_worker_arm_only_config

python gr00t/eval/real_robot/ai_worker/eval_ai_worker.py \
    --embodiment_config_path $CONFIG_PATH \
    --use_compressed_rgb True \
    --viz_rgb True 
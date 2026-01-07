CKPT="./output/ai_worker_B160_AO_REL_LUNAR/checkpoint-20000"

uv run python gr00t/eval/open_loop_eval.py \
    --dataset-path ./data/jkim50104/ffw_sg2_rev1_clear_item \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path $CKPT \
    --traj-ids 0 \
    --action-horizon 16 \
    --steps 1000 \
    --modality-keys left_arm left_gripper right_arm right_gripper \
    --log-path $CKPT \
    --save_plot_path $CKPT

# head lift base
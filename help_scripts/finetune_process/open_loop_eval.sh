CKPT="./output/ai_worker/checkpoint-8000"

python gr00t/eval/open_loop_eval.py \
    --dataset-path ./data/ai_worker/ffw_sg2_rev1_clear_item \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path $CKPT \
    --traj-ids 0 \
    --action-horizon 16 \
    --steps 400 \
    --modality-keys left_arm left_gripper right_arm right_gripper head lift base \
    --log-path $CKPT \
    --save_plot_path $CKPT
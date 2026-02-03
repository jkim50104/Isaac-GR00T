CKPT="output/ffw_sg2_rev1_base_drive_test/G4_B512_REL/checkpoint-10000"

python gr00t/eval/open_loop_eval.py \
    --dataset-path ./data/jkim50104/ffw_sg2_rev1_base_drive_test \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path $CKPT \
    --traj-ids 0 30 50 \
    --action-horizon 32 \
    --steps 1000 \
    --modality-keys left_arm left_gripper right_arm right_gripper head lift base \
    --log-path $CKPT \
    --save_plot_path $CKPT

# ffw_sg2_rev1_clear_item, ffw_sg2_rev1_base_drive_test
# head lift base
# Start inference vla server with trained checkpoint
# python gr00t/eval/run_gr00t_server.py --embodiment-tag GR1 --model-path nvidia/GR00T-N1.6-3B

# Default: nvidia/GR00T-N1.6-3B
MODEL_PATH="output/ai_worker_G4_B512_REL_AO_WR/checkpoint-30000" 

python gr00t/eval/run_gr00t_server.py --embodiment-tag NEW_EMBODIMENT --model-path $MODEL_PATH 
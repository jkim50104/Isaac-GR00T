# Run inference server for GR1 embodiment and GR00T-N1.6-3B model
python gr00t/eval/run_gr00t_server.py --embodiment-tag GR1 --model-path nvidia/GR00T-N1.6-3B

# At different terminal, run inference client test script
python help_scripts/inference_test/inference_client_test.py
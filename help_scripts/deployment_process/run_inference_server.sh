#!/usr/bin/env bash
# Run the GR00T inference server.
#
# Usage:
#   bash help_scripts/deployment_process/run_inference_server.sh

# ======================== CONFIG ========================
# Shared with deploy_robot.sh — it sources this file for MODEL_PATH/POLICY_*.
# MODEL_PATH="output/v1.6/real/clean_the_table/20260430_B512_REL_AO_WR/checkpoint-5000"
# MODEL_PATH="output/v1.6/real/pick_item/20260430_B576_REL_AO/checkpoint-13333"
# MODEL_PATH="output/v1.6/sim/table_pnp/20260430_B576_REL_AO/checkpoint-13333"
MODEL_PATH="output/v1.7/sim_ffw_sg2_138_417_table_pnp/20260430_B288_REL_AO/checkpoint-26666"

POLICY_HOST="localhost"
POLICY_PORT=5555
# ========================================================

# Exit here if this file was sourced (by deploy_robot.sh) rather than executed.
(return 0 2>/dev/null) && return 0

set -euo pipefail
source .venv/bin/activate

python gr00t/eval/run_gr00t_server.py \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path "$MODEL_PATH"

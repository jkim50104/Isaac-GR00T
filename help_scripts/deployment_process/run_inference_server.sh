#!/usr/bin/env bash
# Run the GR00T inference server.
#
# Usage:
#   bash help_scripts/deployment_process/run_inference_server.sh

# ======================== CONFIG ========================
# Shared with deploy_robot.sh — it sources this file for MODEL_PATH/POLICY_*.
# MODEL_PATH="output/ffw_sg2_rev1_clean_the_table/G4_B512_REL_AO_WR/checkpoint-5000"
# MODEL_PATH="output/ffw_sg2_rev1_clean_the_table/G4_B256_REL_AO_WR/checkpoint-10000"
# MODEL_PATH="output/ffw_sg2_rev1_pick_item/G4_B256_REL_AO_WR/checkpoint-30000"
MODEL_PATH="output/sim_ffw_sg2_138_417_table_pnp/G4_B460_REL_AO/checkpoint-16695"

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

#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# GR00T simulation inference launcher
#
# What this does:
# 1) (optional) starts GR00T policy server
# 2) validates ROS2 topics/config compatibility
# 3) runs ai_worker_eval policy bridge for closed-loop inference
#
# Usage:
#   bash help_scripts/deployment_process/deploy_sim_inference.sh
#
# Optional overrides (env vars):
#   START_SERVER=False
#   MODEL_PATH=output/...
#   CONFIG_PATH=help_scripts/data_config/ai_worker_sim_ego_arm_only_config.py
#   LANG_INSTRUCTION="pick the blue bowl"
# -----------------------------------------------------------------------------

as_bool() {
    local value="${1:-False}"
    case "${value}" in
        True|true|1|yes|YES|y|Y) echo "True" ;;
        *) echo "False" ;;
    esac
}

# Server / model
MODEL_PATH="${MODEL_PATH:-output/sim_pick_pringles/G4_B920_REL_AO/checkpoint-20000}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-NEW_EMBODIMENT}"
POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-5555}"
START_SERVER="${START_SERVER:-True}"
START_SERVER="$(as_bool "$START_SERVER")"

# Bridge / runtime
CONFIG_PATH="${CONFIG_PATH:-help_scripts/data_config/ai_worker_sim_ego_arm_only_config.py}"
LANG_INSTRUCTION="${LANG_INSTRUCTION:-pick the blue bowl}"
ACTION_HORIZON="${ACTION_HORIZON:-16}"
USE_COMPRESSED_RGB="${USE_COMPRESSED_RGB:-True}"
VIZ_RGB="${VIZ_RGB:-True}"
REQUIRE_CMD_VEL="${REQUIRE_CMD_VEL:-False}"
CHECK_ONLY="${CHECK_ONLY:-False}"

USE_COMPRESSED_RGB="$(as_bool "$USE_COMPRESSED_RGB")"
VIZ_RGB="$(as_bool "$VIZ_RGB")"
REQUIRE_CMD_VEL="$(as_bool "$REQUIRE_CMD_VEL")"
CHECK_ONLY="$(as_bool "$CHECK_ONLY")"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "[ERROR] Config file not found: $CONFIG_PATH"
    exit 1
fi

if [[ "$START_SERVER" == "True" && ! -e "$MODEL_PATH" && "${MODEL_PATH:0:1}" == "/" ]]; then
    echo "[ERROR] MODEL_PATH does not exist: $MODEL_PATH"
    exit 1
fi

echo "[INFO] Simulation inference configuration"
echo "  START_SERVER:       $START_SERVER"
echo "  MODEL_PATH:         $MODEL_PATH"
echo "  EMBODIMENT_TAG:     $EMBODIMENT_TAG"
echo "  POLICY_HOST:        $POLICY_HOST"
echo "  POLICY_PORT:        $POLICY_PORT"
echo "  CONFIG_PATH:        $CONFIG_PATH"
echo "  LANG_INSTRUCTION:   $LANG_INSTRUCTION"
echo "  ACTION_HORIZON:     $ACTION_HORIZON"
echo "  USE_COMPRESSED_RGB: $USE_COMPRESSED_RGB"
echo "  VIZ_RGB:            $VIZ_RGB"
echo "  REQUIRE_CMD_VEL:    $REQUIRE_CMD_VEL"

check_server_flag="False"
if [[ "$START_SERVER" == "False" ]]; then
    check_server_flag="True"
fi

CONFIG_PATH="$CONFIG_PATH" \
USE_COMPRESSED_RGB="$USE_COMPRESSED_RGB" \
REQUIRE_CMD_VEL="$REQUIRE_CMD_VEL" \
POLICY_HOST="$POLICY_HOST" \
POLICY_PORT="$POLICY_PORT" \
CHECK_POLICY_SERVER="$check_server_flag" \
bash help_scripts/deployment_process/check_sim_inference_ready.sh

if [[ "$CHECK_ONLY" == "True" ]]; then
    echo "[INFO] CHECK_ONLY=True. Exiting without starting inference."
    exit 0
fi

wait_for_port() {
    local host="$1"
    local port="$2"
    python - "$host" "$port" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
deadline = time.time() + 30.0

while time.time() < deadline:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.5)
    try:
        sock.connect((host, port))
        sock.close()
        sys.exit(0)
    except OSError:
        sock.close()
        time.sleep(0.5)

sys.exit(1)
PY
}

server_pid=""
cleanup() {
    if [[ -n "$server_pid" ]]; then
        if kill -0 "$server_pid" >/dev/null 2>&1; then
            echo "[INFO] Stopping policy server (pid=$server_pid)"
            kill "$server_pid" || true
            wait "$server_pid" 2>/dev/null || true
        fi
    fi
}
trap cleanup EXIT INT TERM

if [[ "$START_SERVER" == "True" ]]; then
    echo "[INFO] Starting GR00T policy server..."
    python gr00t/eval/run_gr00t_server.py \
        --embodiment-tag "$EMBODIMENT_TAG" \
        --model-path "$MODEL_PATH" \
        --host "$POLICY_HOST" \
        --port "$POLICY_PORT" &
    server_pid=$!

    if ! wait_for_port "$POLICY_HOST" "$POLICY_PORT"; then
        echo "[ERROR] Policy server did not become ready on ${POLICY_HOST}:${POLICY_PORT}"
        exit 1
    fi
    echo "[INFO] Policy server is ready."
fi

echo "[INFO] Starting simulation inference bridge..."
python gr00t/eval/real_robot/ai_worker/ai_worker_eval.py \
    --policy_host "$POLICY_HOST" \
    --policy_port "$POLICY_PORT" \
    --embodiment_config_path "$CONFIG_PATH" \
    --lang_instruction "$LANG_INSTRUCTION" \
    --action_horizon "$ACTION_HORIZON" \
    --require_cmd_vel "$REQUIRE_CMD_VEL" \
    --use_compressed_rgb "$USE_COMPRESSED_RGB" \
    --viz_rgb "$VIZ_RGB"

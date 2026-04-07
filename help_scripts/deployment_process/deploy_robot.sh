#!/usr/bin/env bash
# Deploy robot: inference server (auto-start if needed) + eval (GUI or headless)
#
# Usage:
#   bash help_scripts/deployment_process/deploy_robot.sh              # headless
#   bash help_scripts/deployment_process/deploy_robot.sh --gui        # GUI mode

set -euo pipefail
source .venv/bin/activate

# ======================== CONFIG ========================
MODEL_PATH="output/ffw_sg2_rev1_pick_item/G4_B256_REL_AO_WR/checkpoint-20000"
POLICY_HOST="localhost"
POLICY_PORT=5555
LANG_INSTRUCTION=""  # empty = auto-read from dataset
# ========================================================

USE_GUI=false
for arg in "$@"; do
    case "$arg" in
        --gui) USE_GUI=true ;;
    esac
done

# Check if inference server is already running
server_running() {
    python3 -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(1.0)
try:
    s.connect(('${POLICY_HOST}', ${POLICY_PORT}))
    sys.exit(0)
except OSError:
    sys.exit(1)
finally:
    s.close()
" 2>/dev/null
}

STARTED_SERVER=false
if server_running; then
    echo "[OK] Inference server already running at ${POLICY_HOST}:${POLICY_PORT}"
else
    echo "[INFO] Inference server not found at ${POLICY_HOST}:${POLICY_PORT}, starting..."
    python gr00t/eval/run_gr00t_server.py \
        --embodiment-tag NEW_EMBODIMENT \
        --model-path "$MODEL_PATH" &
    SERVER_PID=$!
    STARTED_SERVER=true

    echo "[INFO] Waiting for server (PID $SERVER_PID) to be ready..."
    for i in $(seq 1 60); do
        if server_running; then
            echo "[OK] Server ready after ${i}s"
            break
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[ERROR] Server process died"
            exit 1
        fi
        sleep 1
    done

    if ! server_running; then
        echo "[ERROR] Server did not become ready in 60s"
        kill "$SERVER_PID" 2>/dev/null || true
        exit 1
    fi
fi

# Cleanup handler
cleanup() {
    if [[ "$STARTED_SERVER" == "true" ]]; then
        echo "[INFO] Shutting down inference server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    echo "Done."
}
trap cleanup EXIT

# Run eval
if [[ "$USE_GUI" == "true" ]]; then
    echo "[INFO] Launching GUI..."
    python gr00t/eval/real_robot/ai_worker/ai_worker_eval_gui.py \
        --checkpoint "$MODEL_PATH" --host "$POLICY_HOST" --port "$POLICY_PORT"
else
    echo "[INFO] Launching headless eval..."
    python gr00t/eval/real_robot/ai_worker/ai_worker_eval.py \
        --checkpoint_path "$MODEL_PATH" \
        --lang_instruction "$LANG_INSTRUCTION" \
        --action_horizon 32 \
        --use_compressed_rgb True
fi

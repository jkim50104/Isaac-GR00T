#!/usr/bin/env bash
# Deploy robot eval (GUI or headless). Requires the inference server to
# already be running — start it separately with run_inference_server.sh.
#
# Usage:
#   bash help_scripts/deployment_process/deploy_robot.sh              # headless
#   bash help_scripts/deployment_process/deploy_robot.sh --gui        # GUI mode
#   bash help_scripts/deployment_process/deploy_robot.sh --dummy      # open-loop eval on dataset (no robot)

set -euo pipefail
source .venv/bin/activate
source "$(dirname "$0")/run_inference_server.sh"  # MODEL_PATH, POLICY_HOST, POLICY_PORT

USE_GUI=false
USE_DUMMY=false
for arg in "$@"; do
    case "$arg" in
        --gui) USE_GUI=true ;;
        --dummy) USE_DUMMY=true ;;
    esac
done

# Check that inference server is running
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

if ! server_running; then
    echo "[ERROR] Inference server not reachable at ${POLICY_HOST}:${POLICY_PORT}"
    echo "        Start it first with:"
    echo "          bash help_scripts/deployment_process/run_inference_server.sh"
    exit 1
fi
echo "[OK] Inference server reachable at ${POLICY_HOST}:${POLICY_PORT}"

# Run eval
if [[ "$USE_GUI" == "true" ]]; then
    GUI_ARGS=(--checkpoint "$MODEL_PATH" --host "$POLICY_HOST" --port "$POLICY_PORT")
    if [[ "$USE_DUMMY" == "true" ]]; then
        echo "[INFO] Launching GUI in DUMMY mode (no robot)..."
        GUI_ARGS+=(--dummy)
    else
        echo "[INFO] Launching GUI..."
    fi
    python gr00t/eval/real_robot/ai_worker/ai_worker_eval_gui.py "${GUI_ARGS[@]}"
else
    EXTRA_ARGS=()
    if [[ "$USE_DUMMY" == "true" ]]; then
        echo "[INFO] Launching headless eval in DUMMY mode (no robot)..."
        EXTRA_ARGS+=(--dummy True)
    else
        echo "[INFO] Launching headless eval..."
    fi
    python gr00t/eval/real_robot/ai_worker/ai_worker_eval.py \
        --checkpoint_path "$MODEL_PATH" \
        --action_horizon 32 \
        --use_compressed_rgb True \
        "${EXTRA_ARGS[@]}"
fi

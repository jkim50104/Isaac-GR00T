#!/usr/bin/env bash
# Deploy robot eval (GUI or headless). Requires the inference server to
# already be running — start it separately with run_inference_server.sh.
#
# Usage:
#   bash help_scripts/deployment_process/deploy_robot.sh              # headless
#   bash help_scripts/deployment_process/deploy_robot.sh --gui        # GUI mode

set -euo pipefail
source .venv/bin/activate
source "$(dirname "$0")/run_inference_server.sh"  # MODEL_PATH, POLICY_HOST, POLICY_PORT

LANG_INSTRUCTION=""  # empty = auto-read from dataset

USE_GUI=false
for arg in "$@"; do
    case "$arg" in
        --gui) USE_GUI=true ;;
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

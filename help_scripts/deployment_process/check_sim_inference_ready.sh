#!/usr/bin/env bash
set -euo pipefail

# Config (override with env vars if needed)
CONFIG_PATH="${CONFIG_PATH:-help_scripts/data_config/ai_worker_sim_ego_arm_only_config.py}"
USE_COMPRESSED_RGB="${USE_COMPRESSED_RGB:-True}"
REQUIRE_CMD_VEL="${REQUIRE_CMD_VEL:-False}"
POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-5555}"
CHECK_POLICY_SERVER="${CHECK_POLICY_SERVER:-False}"

as_bool() {
    local value="${1:-False}"
    case "${value}" in
        True|true|1|yes|YES|y|Y) echo "True" ;;
        *) echo "False" ;;
    esac
}

USE_COMPRESSED_RGB="$(as_bool "$USE_COMPRESSED_RGB")"
REQUIRE_CMD_VEL="$(as_bool "$REQUIRE_CMD_VEL")"
CHECK_POLICY_SERVER="$(as_bool "$CHECK_POLICY_SERVER")"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "[ERROR] Config file not found: $CONFIG_PATH"
    exit 1
fi

if ! command -v ros2 >/dev/null 2>&1; then
    echo "[ERROR] 'ros2' command not found. Source ROS2 workspace first."
    exit 1
fi

mapfile -t _cfg_lines < <(
    python - "$CONFIG_PATH" <<'PY'
import importlib.util
import sys

cfg_path = sys.argv[1]
spec = importlib.util.spec_from_file_location("_ai_worker_cfg", cfg_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load config: {cfg_path}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if not hasattr(module, "ai_worker"):
    raise RuntimeError(f"{cfg_path} does not define 'ai_worker'")
cfg = module.ai_worker
video_keys = list(cfg["video"].modality_keys)
state_keys = list(cfg["state"].modality_keys)
action_keys = list(cfg["action"].modality_keys)
print("video:" + " ".join(video_keys))
print("state:" + " ".join(state_keys))
print("action:" + " ".join(action_keys))
PY
)

video_keys="${_cfg_lines[0]#video:}"
state_keys="${_cfg_lines[1]#state:}"
action_keys="${_cfg_lines[2]#action:}"

declare -A CAMERA_KEY_TO_TOPIC=(
    ["ego_view"]="/zed/zed_node/left/image_rect_color"
    ["left_wrist_view"]="/camera_left/camera_left/color/image_rect_raw"
    ["right_wrist_view"]="/camera_right/camera_right/color/image_rect_raw"
)

required_topics=("/joint_states")
if [[ "$REQUIRE_CMD_VEL" == "True" ]]; then
    required_topics+=("/cmd_vel")
fi

for k in $video_keys; do
    topic="${CAMERA_KEY_TO_TOPIC[$k]:-}"
    if [[ -z "$topic" ]]; then
        echo "[ERROR] Unknown video modality key in config: '$k'"
        echo "        Add mapping in ai_worker_eval.py and check_sim_inference_ready.sh"
        exit 1
    fi
    if [[ "$USE_COMPRESSED_RGB" == "True" ]]; then
        topic="${topic}/compressed"
    fi
    required_topics+=("$topic")
done

available_topics="$(ros2 topic list 2>/dev/null || true)"
if [[ -z "$available_topics" ]]; then
    echo "[ERROR] No ROS2 topics found. Is the simulation/ROS graph running?"
    exit 1
fi

missing=()
for t in "${required_topics[@]}"; do
    if ! grep -qx "$t" <<<"$available_topics"; then
        missing+=("$t")
    fi
done

if [[ "${#missing[@]}" -gt 0 ]]; then
    echo "[ERROR] Missing required ROS2 topics:"
    for t in "${missing[@]}"; do
        echo "  - $t"
    done
    echo
    echo "Config summary:"
    echo "  config_path:         $CONFIG_PATH"
    echo "  video_keys:          $video_keys"
    echo "  state_keys:          $state_keys"
    echo "  action_keys:         $action_keys"
    echo "  use_compressed_rgb:  $USE_COMPRESSED_RGB"
    echo "  require_cmd_vel:     $REQUIRE_CMD_VEL"
    exit 1
fi

if [[ "$CHECK_POLICY_SERVER" == "True" ]]; then
    if ! python - "$POLICY_HOST" "$POLICY_PORT" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(1.0)
try:
    s.connect((host, port))
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
    then
        echo "[ERROR] Policy server is not reachable at ${POLICY_HOST}:${POLICY_PORT}"
        exit 1
    fi
fi

echo "[OK] Simulation inference precheck passed."
echo "  config_path:         $CONFIG_PATH"
echo "  video_keys:          $video_keys"
echo "  state_keys:          $state_keys"
echo "  action_keys:         $action_keys"
echo "  use_compressed_rgb:  $USE_COMPRESSED_RGB"
echo "  require_cmd_vel:     $REQUIRE_CMD_VEL"
if [[ "$CHECK_POLICY_SERVER" == "True" ]]; then
    echo "  policy_server:       ${POLICY_HOST}:${POLICY_PORT} reachable"
fi

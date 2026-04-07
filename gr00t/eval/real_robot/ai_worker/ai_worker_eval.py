#!/usr/bin/env python3
"""
ai_worker ROS2 Observation Collector + GR00T Adapter + Eval Loop (single-file)

What it does
------------
1) Subscribes to:
   - RGB image:   "/zed/zed_node/left/image_rect_color"   (sensor_msgs/Image)
   - Joint state: /joint_states                                  (sensor_msgs/JointState)
   - Base cmd:    /cmd_vel                                       (geometry_msgs/Twist)

2) Builds ai_worker obs dict:
   {
     "rgb": np.uint8 (H,W,3) RGB
     "state_vec": np.float32 (22,) in fixed layout:
        0..6   left_arm (7)
        7      left_gripper (1)
        8..14  right_arm (7)
        15     right_gripper (1)
        16..17 head (2)
        18     lift (1)
        19..21 base (3)  = [cmd_vel.linear.x, cmd_vel.linear.y, cmd_vel.angular.z]
     "lang": str
   }

3) Converts obs -> GR00T VLA model input format (video/state/language) with (B=1,T=1).

4) Runs an eval loop:
   obs -> policy -> multi-step action chunk -> stream actions

5) Sends actions to robot using:
   - JointTrajectory:
        left:  /leader/joint_trajectory_command_broadcaster_left/joint_trajectory
        right: /leader/joint_trajectory_command_broadcaster_right/joint_trajectory
        head:  /leader/joystick_controller_left/joint_trajectory
        lift:  /leader/joystick_controller_right/joint_trajectory
   - Twist:
        base:  /cmd_vel

Assumptions
-----------
- Policy returns action blocks with keys:
    left_arm(7), left_gripper(1), right_arm(7), right_gripper(1),
    head(2), lift(1), base(3=[lin.x, lin.y, ang.z])
- Joint values are absolute position targets in radians (and lift in its unit), except left_arm and right_arm,
  base is velocity command.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration  # add this import
import cv2

from sensor_msgs.msg import Image, JointState, CompressedImage
from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import draccus
from dataclasses import asdict, dataclass
import logging
from pprint import pformat
import os
import importlib.util
import yaml


from gr00t.policy.server_client import PolicyClient
from lerobot.utils.utils import init_logging, log_say

try:
    from cv_bridge import CvBridge
except ImportError as e:
    raise ImportError(
        "cv_bridge is required. Install (Ubuntu/ROS2): sudo apt install ros-${ROS_DISTRO}-cv-bridge"
    ) from e


# -----------------------------------------------------------------------------
# Checkpoint config parser (shared by headless + GUI)
# -----------------------------------------------------------------------------

class _SafeIgnoreLoader(yaml.SafeLoader):
    pass


def _ignore_unknown(loader, tag_suffix, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    elif isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


_SafeIgnoreLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/", _ignore_unknown
)


def parse_checkpoint_config(checkpoint_path):
    """
    Parse experiment_cfg/config.yaml from a checkpoint directory.
    Returns dict with video_keys, state_keys, action_keys, language_key,
    action_rep, dataset_path, arm_only, use_wrist_view.
    """
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(os.getcwd(), checkpoint_path)

    config_path = os.path.join(checkpoint_path, "experiment_cfg", "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.load(f, Loader=_SafeIgnoreLoader)

    modality = cfg["data"]["modality_configs"]["new_embodiment"]
    video_keys = modality["video"]["modality_keys"]
    state_keys = modality["state"]["modality_keys"]
    action_keys = modality["action"]["modality_keys"]
    language_keys = modality["language"]["modality_keys"]
    language_key = (
        language_keys[0]
        if language_keys
        else "annotation.human.action.task_description"
    )

    action_configs = modality["action"].get("action_configs", [])
    action_rep = "REL"
    if action_configs:
        first_rep = action_configs[0].get("rep", "relative")
        if isinstance(first_rep, list):
            first_rep = first_rep[0]
        action_rep = "REL" if "relative" in str(first_rep).lower() else "ABS"

    dataset_path = cfg["data"]["datasets"][0]["dataset_paths"][0]
    arm_only = "head" not in state_keys
    use_wrist_view = "left_wrist_view" in video_keys

    delta_indices = modality["action"].get("delta_indices", [])
    action_horizon = len(delta_indices) if delta_indices else 16

    return {
        "video_keys": video_keys,
        "state_keys": state_keys,
        "action_keys": action_keys,
        "language_key": language_key,
        "action_rep": action_rep,
        "dataset_path": dataset_path,
        "arm_only": arm_only,
        "use_wrist_view": use_wrist_view,
        "action_horizon": action_horizon,
    }


# -----------------------------------------------------------------------------
# Helper: add (B=1, T=1) dims as expected by GR00T policy server
# -----------------------------------------------------------------------------
def recursive_add_extra_dim(obs: Dict) -> Dict:
    """
    Recursively add an extra dim to arrays or scalars.
    Calling twice makes (B=1, T=1, ...).
    """
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            obs[key] = val[np.newaxis, ...]
        elif isinstance(val, dict):
            obs[key] = recursive_add_extra_dim(val)
        else:
            obs[key] = [val]
    return obs

def concat_obs_rgb(
    rgb,
    scale: float = 1.0,
    key_order=("left_wrist_view", "ego_view", "right_wrist_view"),
):
    """
    Concatenate RGB images into a single ndarray.

    Accepts:
      - dict: {camera_key: np.uint8(H,W,3) RGB}
      - list/tuple: [img0, img1, ...]

    Returns concatenated RGB ndarray, or None if no valid images.
    """
    if rgb is None:
        return None

    if isinstance(rgb, dict):
        if len(rgb) == 0:
            return None
        ordered_keys = [k for k in key_order if k in rgb]
        for k in rgb.keys():
            if k not in ordered_keys:
                ordered_keys.append(k)
        rgb_list = [rgb.get(k, None) for k in ordered_keys]

    elif isinstance(rgb, (list, tuple)):
        if len(rgb) == 0:
            return None
        rgb_list = list(rgb)
    else:
        return None

    imgs = []
    for img in rgb_list:
        if img is None:
            continue
        if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
            continue
        if scale != 1.0:
            h, w = img.shape[:2]
            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
        imgs.append(img)

    if len(imgs) == 0:
        return None
    if len(imgs) == 1:
        return imgs[0]

    target_h = min(im.shape[0] for im in imgs)
    resized = []
    for im in imgs:
        h, w = im.shape[:2]
        if h != target_h:
            new_w = max(1, int(w * (target_h / h)))
            im = cv2.resize(im, (new_w, target_h))
        resized.append(im)

    return cv2.hconcat(resized)



# -----------------------------------------------------------------------------
# Embodiment config loader (FILE PATH version)
# -----------------------------------------------------------------------------
def load_embodiment_cfg_from_path(file_path: str, var_name: str = "ai_worker") -> dict:
    """
    Load a python file by path and extract `ai_worker` dict from it.

    Example:
      file_path = "data/jkim50104/ai_worker_config.py"
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Embodiment config file not found: {file_path}")

    module_name = "_ai_worker_embodiment_config"

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from file: {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, var_name):
        raise AttributeError(
            f"Config file '{file_path}' does not define '{var_name}'"
        )

    cfg = getattr(module, var_name)

    if not isinstance(cfg, dict):
        raise TypeError(
            f"'{var_name}' in {file_path} must be a dict, got {type(cfg)}"
        )

    return cfg


def get_modality_keys(emb_cfg: dict, group: str) -> List[str]:
    """
    emb_cfg[group] is a ModalityConfig object (from GR00T).
    We read `.modality_keys`.
    """
    if group not in emb_cfg:
        raise KeyError(f"embodiment config missing group '{group}'")
    mc = emb_cfg[group]
    if not hasattr(mc, "modality_keys"):
        raise TypeError(f"emb_cfg['{group}'] must have .modality_keys")
    keys = list(mc.modality_keys)
    return keys


# Fixed per-key joint name mapping (kept in main code, not config)
JOINTS_BY_KEY: Dict[str, List[str]] = {
    "left_arm": [
        "arm_l_joint1", "arm_l_joint2", "arm_l_joint3",
        "arm_l_joint4", "arm_l_joint5", "arm_l_joint6",
        "arm_l_joint7",
    ],
    "left_gripper": ["gripper_l_joint1"],
    "right_arm": [
        "arm_r_joint1", "arm_r_joint2", "arm_r_joint3",
        "arm_r_joint4", "arm_r_joint5", "arm_r_joint6",
        "arm_r_joint7",
    ],
    "right_gripper": ["gripper_r_joint1"],
    "head": ["head_joint1", "head_joint2"],
    "lift": ["lift_joint"],
    # base is special: comes from cmd_vel (3 dims)
}

# Base dims are always [lin.x, lin.y, ang.z]
BASE_DIM = 3

# Dims needed for slicing state_vec into blocks
DIMS_BY_KEY = {
    "left_arm": 7,
    "left_gripper": 1,
    "right_arm": 7,
    "right_gripper": 1,
    "head": 2,
    "lift": 1,
    # base handled as BASE_DIM internally
}


# -----------------------------------------------------------------------------
# Adapter: ai_worker obs dict -> GR00T VLA input, and decode action chunk
# -----------------------------------------------------------------------------
class AiWorkerAdapter:
    """
    Config-driven adapter.

    Expects obs:
      obs["rgb"]       : np.uint8 (H,W,3) RGB
      obs["state_vec"] : np.float32 (N,)  where N depends on enabled state keys
      obs["lang"]      : str

    Produces GR00T input:
      model_obs["video"] = {<camera_keys>: rgb}  (first key in config by default)
      model_obs["state"] = dict of enabled state blocks
      model_obs["language"] = {<lang_key>: lang}
    with (B=1,T=1) dims.
    """

    def __init__(
        self,
        policy_client,
        state_keys: List[str],
        action_keys: List[str],
        video_keys: List[str],
        language_key: str,
        dims_by_key: Dict[str, int],
    ):
        self.policy = policy_client

        self.state_keys = list(state_keys)
        self.action_keys = list(action_keys)

        # Use first configured camera key
        self.camera_keys = video_keys

        # Use first configured language key (string like "annotation.human.action.task_description")
        self.language_key = language_key

        # How to slice state_vec into blocks
        self.dims_by_key = dict(dims_by_key)

        # Precompute slices for state_vec based on state_keys ordering
        self.state_slices: Dict[str, Tuple[int, int]] = {}
        idx = 0
        for k in self.state_keys:
            if k == "base":
                dim = BASE_DIM
            else:
                if k not in self.dims_by_key:
                    raise KeyError(f"Missing dims for state key '{k}'")
                dim = int(self.dims_by_key[k])
            self.state_slices[k] = (idx, idx + dim)
            idx += dim

        self.state_vec_dim = idx

    def obs_to_policy_inputs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        assert isinstance(obs["rgb"], dict), \
            f"obs['rgb'] must be dict keyed by modality key, got {type(obs['rgb'])}"
        assert "state_vec" in obs, "obs must include 'state_vec' (N,)"
        assert "lang" in obs, "obs must include 'lang' (string)"

        # Verify required camera keys exist
        missing = [k for k in self.camera_keys if k not in obs["rgb"]]
        assert not missing, \
            f"Missing required camera keys: {missing}. Got keys={list(obs['rgb'].keys())}"
        
        state_vec = obs["state_vec"]
        if not isinstance(state_vec, np.ndarray):
            raise TypeError(f"state_vec must be np.ndarray, got {type(state_vec)}")
        if state_vec.shape != (self.state_vec_dim,):
            raise ValueError(
                f"state_vec must have shape ({self.state_vec_dim},), got {state_vec.shape}. "
                f"Enabled state keys: {self.state_keys}"
            )

        state_vec = state_vec.astype(np.float32, copy=False)

        model_obs: Dict[str, Any] = {}

        # (1) Video
        model_obs["video"] = {k: obs["rgb"][k] for k in self.camera_keys}

        # (2) State blocks (only enabled)
        model_obs["state"] = {}
        for k in self.state_keys:
            s, e = self.state_slices[k]
            model_obs["state"][k] = state_vec[s:e]

        # (3) Language
        model_obs["language"] = {self.language_key: obs["lang"]}

        # (4) Add dims: (B=1,T=1)
        model_obs = recursive_add_extra_dim(model_obs)
        model_obs = recursive_add_extra_dim(model_obs)
        return model_obs

    def decode_action_chunk(self, chunk: Dict[str, np.ndarray], t: int) -> Dict[str, np.ndarray]:
        """
        Expects chunk keys shaped (B,T,D). Returns per-block arrays for timestep t.

        We ONLY decode keys listed in action_keys (from config).
        """
        out: Dict[str, np.ndarray] = {}
        for k in self.action_keys:
            if k not in chunk:
                raise KeyError(f"action_chunk missing key '{k}' (expected from config)")
            out[k] = chunk[k][0][t]  # (D,)
        return out

    def get_action(self, obs: Dict[str, Any]) -> List[Dict[str, np.ndarray]]:
        """
        Calls policy. Returns list of per-step actions (dict per step).
        """
        model_input = self.obs_to_policy_inputs(obs)
        action_chunk, info = self.policy.get_action(model_input)

        # Determine horizon T from any returned key
        any_key = next(iter(action_chunk.keys()))
        horizon = action_chunk[any_key].shape[1]  # (B,T,D) -> T

        return [self.decode_action_chunk(action_chunk, t) for t in range(horizon)]


# -----------------------------------------------------------------------------
# ROS2 collector: subscribes + builds ai_worker obs dict
# -----------------------------------------------------------------------------
class AiWorkerObsCollector(Node):
    def __init__(self, enabled_state_keys: List[str], video_keys: List[str], use_compressed_rgb: bool = False):
        super().__init__("ai_worker_obs_collector")
        self.enabled_state_keys = list(enabled_state_keys)
        self.video_keys = list(video_keys)
        self.use_compressed_rgb = bool(use_compressed_rgb)

        if len(self.video_keys) == 0:
            raise ValueError("video_keys is empty; need at least one camera modality key.")
        
        CAMERA_KEY_TO_TOPIC = {
            "ego_view": "/zed/zed_node/left/image_rect_color",
            "left_wrist_view": "/camera_left/camera_left/color/image_rect_raw",
            "right_wrist_view": "/camera_right/camera_right/color/image_rect_raw",
        }
        
        # Resolve which topics to subscribe to based on modality keys
        self.camera_key_to_topic: Dict[str, str] = {}
        for k in self.video_keys:
            if k not in CAMERA_KEY_TO_TOPIC:
                raise KeyError(f"Unknown video modality key '{k}'. Add it to CAMERA_KEY_TO_TOPIC.")
            topic = CAMERA_KEY_TO_TOPIC[k]
            if self.use_compressed_rgb:
                topic = topic + "/compressed"
            self.camera_key_to_topic[k] = topic

        # For logging / loops
        self.topic_rgb_list = list(self.camera_key_to_topic.values())
        self.topic_to_camera_key = {t: k for k, t in self.camera_key_to_topic.items()}

        self.topic_joint_states = "/joint_states"
        self.topic_cmd_vel = "/cmd_vel"

        # Build the required joint order based on enabled_state_keys.
        # base is special: from cmd_vel, no joints.
        self.state_joint_order: List[str] = []
        for k in self.enabled_state_keys:
            if k == "base":
                continue
            if k not in JOINTS_BY_KEY:
                raise KeyError(
                    f"Enabled state key '{k}' has no fixed joint mapping in JOINTS_BY_KEY. "
                    f"Add it to JOINTS_BY_KEY or remove it from config."
                )
            self.state_joint_order.extend(JOINTS_BY_KEY[k])

        # Total state_vec dim = sum(joint dims) + (base? 3)
        self.state_vec_dim = len(self.state_joint_order) + (BASE_DIM if "base" in self.enabled_state_keys else 0)

        # Internal state
        self.bridge = CvBridge()

        # Latest RGB per camera KEY
        self.latest_rgb_by_key: Dict[str, np.ndarray] = {}
        self.latest_rgb_time_by_key: Dict[str, float] = {}

        self.joint_map: Dict[str, float] = {}
        self.latest_joint_time: Optional[float] = None

        self.latest_cmd_vel = np.zeros(3, dtype=np.float32)
        self.latest_cmd_vel_time: Optional[float] = None

        # Subscribers
        for topic in self.topic_rgb_list:
            if self.use_compressed_rgb:
                self.create_subscription(
                    CompressedImage, topic,
                    lambda msg, t=topic: self._on_rgb_compressed(msg, t),
                    10
                )
            else:
                self.create_subscription(
                    Image, topic,
                    lambda msg, t=topic: self._on_rgb(msg, t),
                    10
                )
        self.create_subscription(JointState, self.topic_joint_states, self._on_joint_states, 10)
        self.create_subscription(Twist, self.topic_cmd_vel, self._on_cmd_vel, 10)

        self.get_logger().info("AiWorkerObsCollector started (config-driven modalities)")
        self.get_logger().info(f"  Enabled state keys: {self.enabled_state_keys}")
        self.get_logger().info(f"  state_vec_dim: {self.state_vec_dim}")
        self.get_logger().info(f"  use_compressed_rgb: {self.use_compressed_rgb}")
        self.get_logger().info(f"  RGB topics: {self.topic_rgb_list}")
        self.get_logger().info(f"  JOINTS: {self.topic_joint_states}")
        self.get_logger().info(f"  CMD:    {self.topic_cmd_vel}")

    def _on_rgb(self, msg: Image, topic: str):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            cam_key = self.topic_to_camera_key[topic]
            self.latest_rgb_by_key[cam_key] = img
            self.latest_rgb_time_by_key[cam_key] = time.time()
        except Exception as e:
            self.get_logger().warn(f"RGB convert failed for {topic}: {e}")

    def _on_rgb_compressed(self, msg: CompressedImage, topic: str):
        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError("cv2.imdecode returned None")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            cam_key = self.topic_to_camera_key[topic]
            self.latest_rgb_by_key[cam_key] = rgb
            self.latest_rgb_time_by_key[cam_key] = time.time()
        except Exception as e:
            self.get_logger().warn(f"Compressed RGB decode failed for {topic}: {e}")

    def _on_joint_states(self, msg: JointState):
        n = min(len(msg.name), len(msg.position))
        for i in range(n):
            self.joint_map[msg.name[i]] = float(msg.position[i])
        self.latest_joint_time = time.time()

    def _on_cmd_vel(self, msg: Twist):
        self.latest_cmd_vel[:] = [msg.linear.x, msg.linear.y, msg.angular.z]
        self.latest_cmd_vel_time = time.time()

    def wait_until_fresh_after(self, after_time: float, timeout_sec: float = 1.0) -> bool:
        t0 = time.time()
        while rclpy.ok() and (time.time() - t0) < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.01)

            ok = True

            if self.latest_joint_time is not None and self.latest_joint_time < after_time:
                ok = False

            # Only require requested camera keys
            for k in self.video_keys:
                t = self.latest_rgb_time_by_key.get(k, None)
                if t is None or t < after_time:
                    ok = False
                    break

            if ok:
                return True
        return False

    def build_obs(self, lang: str, require_cmd_vel: bool = False) -> Optional[Dict[str, Any]]:
        # Require all requested camera keys
        for k in self.video_keys:
            if k not in self.latest_rgb_by_key:
                return None

        # Require joints
        for jname in self.state_joint_order:
            if jname not in self.joint_map:
                return None

        # Require cmd_vel only if base enabled & requested
        if "base" in self.enabled_state_keys and require_cmd_vel and self.latest_cmd_vel_time is None:
            return None

        state_vec = np.zeros(self.state_vec_dim, dtype=np.float32)
        for i, jname in enumerate(self.state_joint_order):
            state_vec[i] = self.joint_map[jname]
        if "base" in self.enabled_state_keys:
            state_vec[len(self.state_joint_order): len(self.state_joint_order) + BASE_DIM] = self.latest_cmd_vel

        # rgb is a dict keyed by modality key (subset)
        rgb_dict = {k: self.latest_rgb_by_key[k] for k in self.video_keys}

        return {
            "rgb": rgb_dict,
            "state_vec": state_vec,
            "lang": lang,
            "timestamps": {
                "rgb_time_by_key": dict(self.latest_rgb_time_by_key),
                "joint_time": self.latest_joint_time,
                "cmd_vel_time": self.latest_cmd_vel_time,
            },
        }


class ObsExecutionTimeChecker:
    """
    Human-readable checker:
    verifies obs timestamps are AFTER the last action execution window.
    Prints relative deltas instead of raw times.
    """
    def __init__(self, exec_sec: float, logger=None):
        self.exec_sec = float(exec_sec)
        self.logger = logger
        self.last_exec_start = None
        self.call_idx = 0

    def mark_execution_start(self):
        self.last_exec_start = time.time()

    def _fmt_delta_ms(self, t_obs: float, t_req: float) -> str:
        # positive = too early (bad), negative = after execution (good)
        delta_ms = (t_req - t_obs) * 1000.0
        return f"{delta_ms:+.1f} ms"

    def check(self, obs: dict):
        if self.last_exec_start is None:
            return

        self.call_idx += 1
        exec_end = self.last_exec_start + self.exec_sec

        ts = obs.get("timestamps", {})
        joint_time = ts.get("joint_time", None)
        rgb_time_by_topic = ts.get("rgb_time_by_topic", {})

        problems = []

        if joint_time is not None:
            delta = exec_end - joint_time
            if delta > 0:
                problems.append(
                    f"joint_state: obs is {self._fmt_delta_ms(joint_time, exec_end)} EARLY"
                )

        for topic, t in rgb_time_by_topic.items():
            if t is None:
                continue
            delta = exec_end - t
            if delta > 0:
                problems.append(
                    f"rgb[{topic}]: obs is {self._fmt_delta_ms(t, exec_end)} EARLY"
                )

        if problems:
            header = (
                f"[ObsTiming] MID-EXECUTION OBS DETECTED\n"
                f"  policy_call={self.call_idx}\n"
                f"  exec_sec={self.exec_sec:.2f}s\n"
                f"  (positive Δ means obs came BEFORE execution finished)\n"
            )
            body = "\n".join(f"  - {p}" for p in problems)
            msg = header + body

            if self.logger:
                self.logger.warn(msg)
            else:
                print(msg)

# -----------------------------------------------------------------------------
# Command Sender: publishes JointTrajectory + Twist (reference-based)
# -----------------------------------------------------------------------------
class AiWorkerCommandSender(Node):
    """
    Publishes robot commands using the same topics / joint layouts as your
    KeyboardController + BaseKeyboardDriver reference scripts.

    IMPORTANT:
      - left_arm / right_arm are treated as RELATIVE deltas (7 each)
      - left_gripper / right_gripper / head / lift are treated as ABSOLUTE targets
      - base is Twist velocities: [lin.x, lin.y, ang.z]
    """

    def __init__(self, enabled_action_keys: List[str]):
        super().__init__("ai_worker_command_sender")
        
        self.enabled_action_keys = list(enabled_action_keys)

        # Publishers (same topics as your reference code)
        self.pub_left = self.create_publisher(
            JointTrajectory,
            "/leader/joint_trajectory_command_broadcaster_left/joint_trajectory",
            10,
        )
        self.pub_right = self.create_publisher(
            JointTrajectory,
            "/leader/joint_trajectory_command_broadcaster_right/joint_trajectory",
            10,
        )
        self.pub_head = self.create_publisher(
            JointTrajectory,
            "/leader/joystick_controller_left/joint_trajectory",
            10,
        )
        self.pub_lift = self.create_publisher(
            JointTrajectory,
            "/leader/joystick_controller_right/joint_trajectory",
            10,
        )
        self.pub_base = self.create_publisher(Twist, "/cmd_vel", 10)

        # Joint order (must match controllers)
        self.left_joint_names = [
            "arm_l_joint1", "arm_l_joint2", "arm_l_joint3",
            "arm_l_joint4", "arm_l_joint5", "arm_l_joint6",
            "arm_l_joint7", "gripper_l_joint1",
        ]
        self.right_joint_names = [
            "arm_r_joint1", "arm_r_joint2", "arm_r_joint3",
            "arm_r_joint4", "arm_r_joint5", "arm_r_joint6",
            "arm_r_joint7", "gripper_r_joint1",
        ]
        self.head_joint_names = ["head_joint1", "head_joint2"]
        self.lift_joint_names = ["lift_joint"]

    def _duration_msg(self, t_sec: float) -> Duration:
        d = Duration()
        d.sec = int(t_sec)
        d.nanosec = int((t_sec - int(t_sec)) * 1e9)
        return d

    def _traj_msg_multi(self, joint_names, positions_seq, dt_per_point: float) -> JointTrajectory:
        """
        positions_seq: (K, D) array/list: K points, D joints
        """
        msg = JointTrajectory()
        msg.joint_names = list(joint_names)

        t = 0.0
        for k in range(len(positions_seq)):
            t += dt_per_point  # start at dt, 2*dt, ...
            pt = JointTrajectoryPoint()
            pt.positions = [float(x) for x in np.asarray(positions_seq[k]).reshape(-1)]
            pt.time_from_start = self._duration_msg(t)
            msg.points.append(pt)
        return msg

    def send_action_sequence_1s(self, action_seq, exec_sec: float = 1.0) -> None:
        """
        Execute one policy action chunk over exec_sec seconds.

        This is config-driven:
        - Only publishes modalities present in enabled_action_keys.
        - Policy outputs are treated as ABSOLUTE targets (per your note).
        """
        if len(action_seq) == 0:
            return

        K = len(action_seq)
        dt_per_point = exec_sec / K

        # ---- Left / Right arms+grippers (publish only if present) ----
        has_left_arm = "left_arm" in self.enabled_action_keys
        has_left_grip = "left_gripper" in self.enabled_action_keys
        has_right_arm = "right_arm" in self.enabled_action_keys
        has_right_grip = "right_gripper" in self.enabled_action_keys

        if has_left_arm and has_left_grip:
            left_arm = np.stack([a["left_arm"] for a in action_seq], axis=0)       # (K,7)
            left_grip = np.stack([a["left_gripper"] for a in action_seq], axis=0)  # (K,1)
            left_pos_seq = np.concatenate([left_arm, left_grip], axis=1)           # (K,8)
            self.pub_left.publish(self._traj_msg_multi(self.left_joint_names, left_pos_seq, dt_per_point))
        elif has_left_arm or has_left_grip:
            self.get_logger().warn("Config includes only one of left_arm/left_gripper; skipping left trajectory publish.")

        if has_right_arm and has_right_grip:
            right_arm = np.stack([a["right_arm"] for a in action_seq], axis=0)       # (K,7)
            right_grip = np.stack([a["right_gripper"] for a in action_seq], axis=0)  # (K,1)
            right_pos_seq = np.concatenate([right_arm, right_grip], axis=1)          # (K,8)
            self.pub_right.publish(self._traj_msg_multi(self.right_joint_names, right_pos_seq, dt_per_point))
        elif has_right_arm or has_right_grip:
            self.get_logger().warn("Config includes only one of right_arm/right_gripper; skipping right trajectory publish.")

        # ---- Head (optional) ----
        if "head" in self.enabled_action_keys:
            head = np.stack([a["head"] for a in action_seq], axis=0)  # (K,2)
            self.pub_head.publish(self._traj_msg_multi(self.head_joint_names, head, dt_per_point))

        # ---- Lift (optional) ----
        if "lift" in self.enabled_action_keys:
            lift = np.stack([a["lift"] for a in action_seq], axis=0)  # (K,1)
            self.pub_lift.publish(self._traj_msg_multi(self.lift_joint_names, lift, dt_per_point))

        # ---- Base (optional) ----
        if "base" in self.enabled_action_keys:
            base = np.asarray(action_seq[-1]["base"]).reshape(3)
            tw = Twist()
            tw.linear.x = float(base[0])
            tw.linear.y = float(base[1])
            tw.angular.z = float(base[2])
            self.pub_base.publish(tw)


# =============================================================================
# Evaluation Config
# =============================================================================

@dataclass
class EvalConfig:
    """
    Command-line configuration for ai_worker real-robot policy evaluation.
    """
    policy_host: str = "localhost"
    policy_port: int = 5555
    action_horizon: int = 16
    lang_instruction: str = ""
    play_sounds: bool = False
    timeout: int = 60  # (kept to match SO100 style; not enforced here)
    
    # If True: wait until cmd_vel has been received at least once before producing obs
    require_cmd_vel: bool = False
    
    use_compressed_rgb: bool = False  # subscribe to /compressed topics and decode JPEG/PNG
    
    # Checkpoint path — used to derive modality config + dataset path
    checkpoint_path: str = ""

    # Fallback: legacy embodiment config file (used only if checkpoint_path is empty)
    embodiment_config_path: str = ""
    embodiment_config_var: str = "ai_worker"




# =============================================================================
# Reusable Eval Loop (used by both headless and GUI)
# =============================================================================

def run_eval_loop(
    collector,
    sender,
    adapter,
    lang_instruction,
    action_horizon=16,
    exec_sec=1.0,
    require_cmd_vel=False,
    should_stop=None,
    on_obs=None,
    on_status=None,
    get_action_horizon=None,
):
    """
    Core eval loop shared by headless and GUI modes.

    Args:
        collector: AiWorkerObsCollector
        sender: AiWorkerCommandSender
        adapter: AiWorkerAdapter
        lang_instruction: language command string
        action_horizon: default number of action steps
        exec_sec: seconds per action chunk execution
        require_cmd_vel: wait for cmd_vel before building obs
        should_stop: () -> bool, checked each iteration
        on_obs: (obs_dict) -> None, called when obs is built
        on_status: (str) -> None, called with status messages
        get_action_horizon: () -> int, for dynamic horizon (e.g. GUI slider)
    """
    if should_stop is None:
        should_stop = lambda: False

    time_checker = ObsExecutionTimeChecker(exec_sec, logger=collector.get_logger())
    _last_wait_log = 0.0

    while rclpy.ok() and not should_stop():
        rclpy.spin_once(collector, timeout_sec=0.01)
        rclpy.spin_once(sender, timeout_sec=0.0)

        obs = collector.build_obs(lang_instruction, require_cmd_vel=require_cmd_vel)
        if obs is None:
            now = time.time()
            if on_status:
                on_status("Waiting for obs...")
            elif now - _last_wait_log >= 1.0:
                print(f"\r[INFO] Waiting for obs... ({time.strftime('%H:%M:%S')})", end="", flush=True)
                _last_wait_log = now
            continue

        time_checker.check(obs)

        if on_obs:
            on_obs(obs)

        if should_stop():
            break

        if on_status:
            on_status("Running inference...")

        horizon = get_action_horizon() if get_action_horizon else action_horizon
        actions = adapter.get_action(obs)
        action_seq = actions[:horizon]

        if should_stop():
            break

        sender.send_action_sequence_1s(action_seq, exec_sec=exec_sec)
        time_checker.mark_execution_start()

        if on_status:
            on_status("Executing actions...")

        t_end = time.time() + exec_sec
        while time.time() < t_end and rclpy.ok() and not should_stop():
            rclpy.spin_once(collector, timeout_sec=0.01)
            rclpy.spin_once(sender, timeout_sec=0.0)

        if not should_stop():
            ok = collector.wait_until_fresh_after(after_time=t_end, timeout_sec=0.01)
            if not ok:
                collector.get_logger().warn(
                    "Freshness barrier timed out (RGB likely lagging). Proceeding anyway."
                )


# =============================================================================
# Headless CLI Entry Point
# =============================================================================
@draccus.wrap()
def eval(cfg: EvalConfig):
    """Headless ai_worker policy evaluation (ROS2)."""
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Load modality config from checkpoint or legacy config file
    if cfg.checkpoint_path:
        ckpt_config = parse_checkpoint_config(cfg.checkpoint_path)
        video_keys = ckpt_config["video_keys"]
        state_keys = ckpt_config["state_keys"]
        action_keys = ckpt_config["action_keys"]
        language_key = ckpt_config["language_key"]
        dataset_path = ckpt_config["dataset_path"]
        if dataset_path and not os.path.isabs(dataset_path):
            dataset_path = os.path.normpath(os.path.join(os.getcwd(), dataset_path))
        logging.info(f"Config from checkpoint: video={video_keys}, state={state_keys}, action={action_keys}")
    elif cfg.embodiment_config_path:
        emb_cfg = load_embodiment_cfg_from_path(cfg.embodiment_config_path, cfg.embodiment_config_var)
        video_keys = get_modality_keys(emb_cfg, "video")
        state_keys = get_modality_keys(emb_cfg, "state")
        action_keys = get_modality_keys(emb_cfg, "action")
        lang_keys = get_modality_keys(emb_cfg, "language")
        language_key = lang_keys[0] if len(lang_keys) > 0 else "annotation.human.action.task_description"
        dataset_path = ""
    else:
        raise ValueError("Either --checkpoint_path or --embodiment_config_path must be provided")

    # Load init_pose module
    _init_mod = None
    if dataset_path:

        _init_mod_path = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "..", "..",
            "help_scripts", "deployment_process", "init_pose_from_data.py",
        ))
        _init_spec = importlib.util.spec_from_file_location("init_pose_from_data", _init_mod_path)
        _init_mod = importlib.util.module_from_spec(_init_spec)
        _init_spec.loader.exec_module(_init_mod)

    # If no lang instruction given, read from dataset
    lang_instruction = cfg.lang_instruction
    if not lang_instruction and _init_mod and dataset_path:
        lang_instruction = _init_mod.get_lang_instruction(dataset_path) or ""
        if lang_instruction:
            logging.info(f"Using language instruction from dataset: \"{lang_instruction}\"")
    if not lang_instruction:
        lang_instruction = "pick the blue bowl"
        logging.warning(f"No language instruction provided, using default: \"{lang_instruction}\"")

    rclpy.init()
    collector = AiWorkerObsCollector(
        enabled_state_keys=state_keys,
        video_keys=video_keys,
        use_compressed_rgb=cfg.use_compressed_rgb,
    )
    sender = AiWorkerCommandSender(enabled_action_keys=action_keys)

    log_say("Initializing ROS2 collector/sender", cfg.play_sounds, blocking=True)

    adapter = AiWorkerAdapter(
        policy_client=PolicyClient(host=cfg.policy_host, port=cfg.policy_port),
        state_keys=state_keys,
        action_keys=action_keys,
        video_keys=video_keys,
        language_key=language_key,
        dims_by_key=DIMS_BY_KEY,
    )

    log_say(f'Policy ready: "{lang_instruction}"', cfg.play_sounds, blocking=True)

    # Init pose reset (once at startup)
    if _init_mod and dataset_path:
        logging.info(f"Running init pose from dataset: {dataset_path}")

        # Spin until we have joint states
        for _ in range(500):  # ~5s max
            rclpy.spin_once(collector, timeout_sec=0.01)
            if collector.joint_map:
                break

        if not collector.joint_map:
            logging.warning("No joint states received — skipping init pose")
        else:
            positions, ref_ep, ep_lang = _init_mod.compute_init_pose(dataset_path)
            if ep_lang and not lang_instruction:
                lang_instruction = ep_lang
                logging.info(f"Using language from episode {ref_ep}: \"{lang_instruction}\"")

            max_attempts = 5
            for attempt in range(1, max_attempts + 1):
                logging.info(f"Sending robot to init pose (ep {ref_ep}, attempt {attempt})...")
                _init_mod.send_init_pose(sender, collector.joint_map, positions)

                t_wait_end = time.time() + 3.0
                while time.time() < t_wait_end and rclpy.ok():
                    rclpy.spin_once(collector, timeout_sec=0.01)
                    rclpy.spin_once(sender, timeout_sec=0.0)

                reached, max_err, worst = _init_mod.check_init_pose_reached(
                    collector.joint_map, positions
                )
                if reached:
                    logging.info(f"Init pose reached (attempt {attempt}, err={max_err:.4f})")
                    break
                logging.warning(f"Not at init pose (err={max_err:.4f} @ {worst}), retrying...")
            else:
                logging.warning(f"Init pose not reached after {max_attempts} attempts (err={max_err:.4f} @ {worst})")

    input("\nPress Enter to start eval loop...")

    try:
        run_eval_loop(
            collector=collector,
            sender=sender,
            adapter=adapter,
            lang_instruction=lang_instruction,
            action_horizon=cfg.action_horizon,
            require_cmd_vel=cfg.require_cmd_vel,
        )
    except KeyboardInterrupt:
        pass
    finally:
        collector.destroy_node()
        sender.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    eval()

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
   obs -> policy -> multi-step action chunk -> stream actions at ~control_hz

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
import cv2
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import draccus
from dataclasses import asdict, dataclass
import logging
from pprint import pformat

from gr00t.policy.server_client import PolicyClient
from lerobot.utils.utils import init_logging, log_say

try:
    from cv_bridge import CvBridge
except ImportError as e:
    raise ImportError(
        "cv_bridge is required. Install (Ubuntu/ROS2): sudo apt install ros-${ROS_DISTRO}-cv-bridge"
    ) from e
    


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

def format_state_vec(state_vec: np.ndarray) -> str:
    sv = np.asarray(state_vec).reshape(-1)
    assert sv.shape[0] == 22

    left_arm = sv[0:7]
    left_grip = sv[7:8]
    right_arm = sv[8:15]
    right_grip = sv[15:16]
    head = sv[16:18]
    lift = sv[18:19]
    base = sv[19:22]

    def arr(a):
        return np.array2string(a, precision=4, floatmode="fixed", suppress_small=False)

    return "\n".join([
        "STATE_VEC (22):",
        f"  left_arm(7)      {arr(left_arm)}",
        f"  left_gripper(1)  {arr(left_grip)}",
        f"  right_arm(7)     {arr(right_arm)}",
        f"  right_gripper(1) {arr(right_grip)}",
        f"  head(2)          {arr(head)}",
        f"  lift(1)          {arr(lift)}",
        f"  base(3)          {arr(base)}   # [vx, vy, wz]",
    ])


# -----------------------------------------------------------------------------
# Adapter: ai_worker obs dict -> GR00T VLA input, and decode action chunk
# -----------------------------------------------------------------------------
class AiWorkerAdapter:
    """
    Expects obs:
      obs["rgb"]       : np.uint8 (H,W,3) RGB
      obs["state_vec"] : np.float32 (22,)
      obs["lang"]      : str

    Produces GR00T input:
      model_obs["video"] = {"ego_view": rgb}
      model_obs["state"] = {
          "left_arm": (7,), "left_gripper": (1,),
          "right_arm": (7,), "right_gripper": (1,),
          "head": (2,), "lift": (1,), "base": (3,)
      }
      model_obs["language"] = {"annotation.human.action.task_description": lang}
    with (B=1,T=1) dims.
    """

    def __init__(self, policy_client=None):
        self.policy = policy_client

        self.state_slices: Dict[str, Tuple[int, int]] = {
            "left_arm": (0, 7),
            "left_gripper": (7, 8),
            "right_arm": (8, 15),
            "right_gripper": (15, 16),
            "head": (16, 18),
            "lift": (18, 19),
            "base": (19, 22),
        }

        # camera key name used inside model_obs["video"]
        self.camera_key = "ego_view"

    def obs_to_policy_inputs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        assert "rgb" in obs, "obs must include 'rgb' (H,W,3 uint8)"
        assert "state_vec" in obs, "obs must include 'state_vec' (22,)"
        assert "lang" in obs, "obs must include 'lang' (string)"

        state_vec = obs["state_vec"]
        if not isinstance(state_vec, np.ndarray):
            raise TypeError(f"state_vec must be np.ndarray, got {type(state_vec)}")
        if state_vec.shape != (22,):
            raise ValueError(f"state_vec must have shape (22,), got {state_vec.shape}")

        state_vec = state_vec.astype(np.float32, copy=False)

        model_obs: Dict[str, Any] = {}

        # (1) Video
        model_obs["video"] = {self.camera_key: obs["rgb"]}

        # (2) State blocks
        model_obs["state"] = {}
        for k, (s, e) in self.state_slices.items():
            model_obs["state"][k] = state_vec[s:e]

        # (3) Language
        model_obs["language"] = {"annotation.human.action.task_description": obs["lang"]}

        # (4) Add dims: (B=1,T=1)
        model_obs = recursive_add_extra_dim(model_obs)
        model_obs = recursive_add_extra_dim(model_obs)
        return model_obs

    def decode_action_chunk(self, chunk: Dict[str, np.ndarray], t: int) -> Dict[str, np.ndarray]:
        """
        Expects chunk keys matching self.state_slices, each shaped (B,T,D).
        Returns dict of per-block arrays for timestep t.
        """
        out: Dict[str, np.ndarray] = {}
        for k, _ in self.state_slices.items():
            if k not in chunk:
                raise KeyError(f"action_chunk missing key '{k}'")
            out[k] = chunk[k][0][t]  # (D,)
        return out

    def get_action(self, obs: Dict[str, Any]) -> List[Dict[str, np.ndarray]]:
        """
        Calls policy if self.policy is set. Returns list of per-step actions.
        """
        if self.policy is None:
            raise RuntimeError("AiWorkerAdapter.policy is None. Provide a PolicyClient to call get_action().")

        model_input = self.obs_to_policy_inputs(obs)
        action_chunk, info = self.policy.get_action(model_input)

        any_key = next(iter(action_chunk.keys()))
        horizon = action_chunk[any_key].shape[1]  # (B,T,D) -> T
        return [self.decode_action_chunk(action_chunk, t) for t in range(horizon)]


# -----------------------------------------------------------------------------
# ROS2 collector: subscribes + builds ai_worker obs dict
# -----------------------------------------------------------------------------
class AiWorkerObsCollector(Node):
    """
    Collects ai_worker observation from ROS2 topics:
      - RGB image topic
      - JointState
      - cmd_vel
    """

    def __init__(self):
        super().__init__("ai_worker_obs_collector")

        # Topics from your current ROS2 graph
        self.topic_rgb = "/zed/zed_node/left/image_rect_color"
        self.topic_joint_states = "/joint_states"
        self.topic_cmd_vel = "/cmd_vel"

        # Joint ordering for dims 0..18 (19 values total)
        self.left_arm_joints = [
            "arm_l_joint1", "arm_l_joint2", "arm_l_joint3",
            "arm_l_joint4", "arm_l_joint5", "arm_l_joint6",
            "arm_l_joint7",
        ]
        self.left_gripper_joint = ["gripper_l_joint1"]

        self.right_arm_joints = [
            "arm_r_joint1", "arm_r_joint2", "arm_r_joint3",
            "arm_r_joint4", "arm_r_joint5", "arm_r_joint6",
            "arm_r_joint7",
        ]
        self.right_gripper_joint = ["gripper_r_joint1"]

        self.head_joints = ["head_joint1", "head_joint2"]
        self.lift_joint = ["lift_joint"]

        self.state19_joint_order: List[str] = (
            self.left_arm_joints
            + self.left_gripper_joint
            + self.right_arm_joints
            + self.right_gripper_joint
            + self.head_joints
            + self.lift_joint
        )
        if len(self.state19_joint_order) != 19:
            raise ValueError(f"Expected 19 joints for state[0:19], got {len(self.state19_joint_order)}")

        # Internal state
        self.bridge = CvBridge()

        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_rgb_time: Optional[float] = None

        self.joint_map: Dict[str, float] = {}
        self.latest_joint_time: Optional[float] = None

        # base dims: [lin.x, lin.y, ang.z]
        self.latest_cmd_vel = np.zeros(3, dtype=np.float32)
        self.latest_cmd_vel_time: Optional[float] = None

        # Subscribers
        self.create_subscription(Image, self.topic_rgb, self._on_rgb, 10)
        self.create_subscription(JointState, self.topic_joint_states, self._on_joint_states, 10)
        self.create_subscription(Twist, self.topic_cmd_vel, self._on_cmd_vel, 10)

        self.get_logger().info("AiWorkerObsCollector started")
        self.get_logger().info(f"  RGB:    {self.topic_rgb}")
        self.get_logger().info(f"  JOINTS: {self.topic_joint_states}")
        self.get_logger().info(f"  CMD:    {self.topic_cmd_vel}")

    def _on_rgb(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            self.latest_rgb = img
            self.latest_rgb_time = time.time()
        except Exception as e:
            self.get_logger().warn(f"RGB convert failed: {e}")

    def _on_joint_states(self, msg: JointState):
        n = min(len(msg.name), len(msg.position))
        for i in range(n):
            self.joint_map[msg.name[i]] = float(msg.position[i])
        self.latest_joint_time = time.time()

    def _on_cmd_vel(self, msg: Twist):
        self.latest_cmd_vel[:] = [msg.linear.x, msg.linear.y, msg.angular.z]
        self.latest_cmd_vel_time = time.time()

    def build_obs(self, lang: str, require_cmd_vel: bool = False) -> Optional[Dict[str, Any]]:
        """
        Build and return ai_worker obs dict, or None if not ready.

        require_cmd_vel:
          - False: base defaults to zeros until cmd_vel arrives
          - True: return None until cmd_vel is received at least once
        """
        if self.latest_rgb is None:
            return None

        for jname in self.state19_joint_order:
            if jname not in self.joint_map:
                return None

        if require_cmd_vel and self.latest_cmd_vel_time is None:
            return None

        state_vec = np.zeros(22, dtype=np.float32)

        # 0..18 from joints
        for i, jname in enumerate(self.state19_joint_order):
            state_vec[i] = self.joint_map[jname]

        # 19..21 base from cmd_vel
        state_vec[19:22] = self.latest_cmd_vel

        return {
            "rgb": self.latest_rgb,
            "state_vec": state_vec,
            "lang": lang,
            "timestamps": {
                "rgb_time": self.latest_rgb_time,
                "joint_time": self.latest_joint_time,
                "cmd_vel_time": self.latest_cmd_vel_time,
            },
        }


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

    def __init__(self):
        super().__init__("ai_worker_command_sender")

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

        # Single-point horizon for controllers (simple; can be replaced with smoothing later)
        self.point_dt_sec = 0.10  # seconds

    def _traj_msg(self, joint_names: List[str], positions: np.ndarray) -> JointTrajectory:
        msg = JointTrajectory()
        msg.joint_names = list(joint_names)

        pt = JointTrajectoryPoint()
        pt.positions = [float(x) for x in positions.reshape(-1)]
        pt.time_from_start.sec = int(self.point_dt_sec)
        pt.time_from_start.nanosec = int((self.point_dt_sec % 1.0) * 1e9)

        msg.points = [pt]
        return msg

    def send_action_blocks(self, action_blocks: Dict[str, np.ndarray]) -> None:
        # ---------- Validate shapes ----------
        left_arm_abs = np.asarray(action_blocks["left_arm"]).reshape(-1)
        right_arm_abs = np.asarray(action_blocks["right_arm"]).reshape(-1)
        if left_arm_abs.shape[0] != 7:
            raise ValueError(f"left_arm must be 7 values, got {left_arm_abs.shape[0]}")
        if right_arm_abs.shape[0] != 7:
            raise ValueError(f"right_arm must be 7 values, got {right_arm_abs.shape[0]}")

        left_grip_abs = np.asarray(action_blocks["left_gripper"]).reshape(-1)
        right_grip_abs = np.asarray(action_blocks["right_gripper"]).reshape(-1)
        if left_grip_abs.shape[0] != 1:
            raise ValueError(f"left_gripper must be 1 value, got {left_grip_abs.shape[0]}")
        if right_grip_abs.shape[0] != 1:
            raise ValueError(f"right_gripper must be 1 value, got {right_grip_abs.shape[0]}")

        head_abs = np.asarray(action_blocks["head"]).reshape(-1)
        lift_abs = np.asarray(action_blocks["lift"]).reshape(-1)
        if head_abs.shape[0] != 2:
            raise ValueError(f"head must be 2 values, got {head_abs.shape[0]}")
        if lift_abs.shape[0] != 1:
            raise ValueError(f"lift must be 1 value, got {lift_abs.shape[0]}")

        base = np.asarray(action_blocks["base"]).reshape(-1)
        if base.shape[0] != 3:
            raise ValueError(f"base must be 3 values [lin.x, lin.y, ang.z], got {base.shape[0]}")

        # ---------- Arms are ABSOLUTE targets now ----------
        left_pos = np.concatenate([left_arm_abs, left_grip_abs], axis=0)     # 8
        right_pos = np.concatenate([right_arm_abs, right_grip_abs], axis=0)  # 8

        # ---------- Publish JointTrajectory ----------
        self.pub_left.publish(self._traj_msg(self.left_joint_names, left_pos))
        self.pub_right.publish(self._traj_msg(self.right_joint_names, right_pos))
        # self.pub_head.publish(self._traj_msg(self.head_joint_names, head_abs))
        # self.pub_lift.publish(self._traj_msg(self.lift_joint_names, lift_abs))

        # ---------- Publish Twist ----------
        tw = Twist()
        tw.linear.x = float(base[0])
        tw.linear.y = float(base[1])
        tw.angular.z = float(base[2])
        # self.pub_base.publish(tw)
        
    def compute_targets_for_preview(
        self,
        action_blocks: Dict[str, np.ndarray],
        obs_state_vec: np.ndarray,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Preview what will be sent, alongside current joint states (from obs_state_vec).

        Actions are ABSOLUTE targets:
        - left_arm/right_arm/head/lift/grippers: absolute joint targets
        - base: velocity [vx, vy, wz]

        Returns None only if shapes are wrong.
        """
        try:
            sv = np.asarray(obs_state_vec).reshape(-1)
            if sv.shape[0] != 22:
                return None

            # ---- current from obs state_vec (fixed layout) ----
            cur_left_arm = sv[0:7]
            cur_left_grip = sv[7:8]
            cur_right_arm = sv[8:15]
            cur_right_grip = sv[15:16]
            cur_head = sv[16:18]
            cur_lift = sv[18:19]
            cur_base = sv[19:22]  # last commanded cmd_vel (not true base state)

            # ---- targets from action (absolute) ----
            tgt_left_arm = np.asarray(action_blocks["left_arm"]).reshape(-1)
            tgt_right_arm = np.asarray(action_blocks["right_arm"]).reshape(-1)
            if tgt_left_arm.shape[0] != 7 or tgt_right_arm.shape[0] != 7:
                return None

            tgt_left_grip = np.asarray(action_blocks["left_gripper"]).reshape(-1)
            tgt_right_grip = np.asarray(action_blocks["right_gripper"]).reshape(-1)
            if tgt_left_grip.shape[0] != 1 or tgt_right_grip.shape[0] != 1:
                return None

            tgt_head = np.asarray(action_blocks["head"]).reshape(-1)
            tgt_lift = np.asarray(action_blocks["lift"]).reshape(-1)
            if tgt_head.shape[0] != 2 or tgt_lift.shape[0] != 1:
                return None

            tgt_base = np.asarray(action_blocks["base"]).reshape(-1)
            if tgt_base.shape[0] != 3:
                return None

            return {
                # current
                "left_arm_current": cur_left_arm,
                "right_arm_current": cur_right_arm,
                "left_gripper_current": cur_left_grip,
                "right_gripper_current": cur_right_grip,
                "head_current": cur_head,
                "lift_current": cur_lift,
                "base_current": cur_base,

                # targets (what WILL be sent)
                "left_arm_target": tgt_left_arm,
                "right_arm_target": tgt_right_arm,
                "left_gripper_target": tgt_left_grip,
                "right_gripper_target": tgt_right_grip,
                "head_target": tgt_head,
                "lift_target": tgt_lift,
                "base_target": tgt_base,

                # optional deltas for sanity (not used for control)
                "left_arm_delta": (tgt_left_arm - cur_left_arm),
                "right_arm_delta": (tgt_right_arm - cur_right_arm),
                "left_gripper_delta": (tgt_left_grip - cur_left_grip),
                "right_gripper_delta": (tgt_right_grip - cur_right_grip),
                "head_delta": (tgt_head - cur_head),
                "lift_delta": (tgt_lift - cur_lift),
            }
        except Exception:
            return None


class DebugUI:
    """
    OpenCV-based visualization + key control.
      - shows RGB
      - prints/overlays state vector + staleness
      - lets you press SPACE to send the currently previewed action
    """
    def __init__(self, window="ai_worker_debug", scale=1.0):
        self.window = window
        self.scale = float(scale)
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)

    def _put_lines(self, img_bgr, lines, x=10, y=20, dy=18):
        for i, line in enumerate(lines):
            cv2.putText(
                img_bgr, line, (x, y + i * dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
            )

    def show_obs(self, obs: Dict[str, Any], now: float):
        rgb = obs["rgb"]  # RGB uint8
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if self.scale != 1.0:
            h, w = bgr.shape[:2]
            bgr = cv2.resize(bgr, (int(w * self.scale), int(h * self.scale)))

        ts = obs.get("timestamps", {})
        rgb_age = (now - ts.get("rgb_time", now)) if ts.get("rgb_time") else -1
        joint_age = (now - ts.get("joint_time", now)) if ts.get("joint_time") else -1
        cmd_age = (now - ts.get("cmd_vel_time", now)) if ts.get("cmd_vel_time") else -1

        sv = obs["state_vec"]
        # groups from your fixed layout
        left_arm = sv[0:7]
        left_grip = sv[7]
        right_arm = sv[8:15]
        right_grip = sv[15]
        head = sv[16:18]
        lift = sv[18]
        base = sv[19:22]

        lines = [
            f"OBS ages: rgb={rgb_age:.3f}s joints={joint_age:.3f}s cmd={cmd_age:.3f}s",
            f"base(cmd_vel) = [vx={base[0]:+.3f}, vy={base[1]:+.3f}, wz={base[2]:+.3f}]",
            f"left_grip={left_grip:+.3f} right_grip={right_grip:+.3f} lift={lift:+.3f}",
            f"head = [{head[0]:+.3f}, {head[1]:+.3f}]",
            f"L arm: [{', '.join([f'{x:+.3f}' for x in left_arm])}]",
            f"R arm: [{', '.join([f'{x:+.3f}' for x in right_arm])}]",
            f"lang: {obs.get('lang','')[:80]}",
            "Keys: [SPACE]=send  [n]=next(skip)  [r]=recompute  [q]=quit",
        ]
        self._put_lines(bgr, lines)

        cv2.imshow(self.window, bgr)

    def show_action_preview(self, action_blocks: Dict[str, np.ndarray]):
        # Print a compact preview to terminal (you can expand as needed)
        def fmt(name):
            a = np.asarray(action_blocks[name]).reshape(-1)
            return f"{name}: " + np.array2string(a, precision=3, floatmode="fixed", suppress_small=True)

        print("---- ACTION PREVIEW ----")
        for k in ["left_arm","left_gripper","right_arm","right_gripper","head","lift","base"]:
            print(fmt(k))
        print("------------------------")

    def get_key(self, delay_ms=1) -> int:
        # returns ASCII code (lowercase handled by caller)
        return cv2.waitKey(int(delay_ms)) & 0xFF

    def close(self):
        cv2.destroyWindow(self.window)


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
    action_horizon: int = 8
    lang_instruction: str = "Clear the items on the shelf."
    play_sounds: bool = False
    timeout: int = 60  # (kept to match SO100 style; not enforced here)

    # Control loop rate (matches SO100 ~30Hz)
    control_hz: float = 30.0

    # If True: wait until cmd_vel has been received at least once before producing obs
    require_cmd_vel: bool = False

    # Single-point trajectory time_from_start (seconds)
    traj_point_dt: float = 0.10


# =============================================================================
# Main Eval Loop
# =============================================================================

@draccus.wrap()
def eval(cfg: EvalConfig):
    """
    Main entry point for ai_worker policy evaluation (ROS2).
    """
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # -------------------------------------------------------------------------
    # 1) Initialize ROS2 + Collector + Sender
    # -------------------------------------------------------------------------
    rclpy.init()
    collector = AiWorkerObsCollector()
    sender = AiWorkerCommandSender()
    sender.point_dt_sec = float(cfg.traj_point_dt)

    log_say("Initializing ROS2 collector/sender", cfg.play_sounds, blocking=True)

    # -------------------------------------------------------------------------
    # 2) Initialize Policy Client + Adapter
    # -------------------------------------------------------------------------
    policy_client = PolicyClient(host=cfg.policy_host, port=cfg.policy_port)
    policy = AiWorkerAdapter(policy_client)

    log_say(
        f'Policy ready with instruction: "{cfg.lang_instruction}"',
        cfg.play_sounds,
        blocking=True,
    )

    dt = 1.0 / max(cfg.control_hz, 1e-6)

    ui = DebugUI(scale=1.0)

    try:
        actions = None
        action_idx = 0

        while rclpy.ok():
            # keep ROS callbacks flowing
            rclpy.spin_once(collector, timeout_sec=0.01)
            rclpy.spin_once(sender, timeout_sec=0.0)

            now = time.time()
            obs = collector.build_obs(cfg.lang_instruction, require_cmd_vel=cfg.require_cmd_vel)
            if obs is None:
                # show nothing until first image arrives; avoid spamming logs
                continue

            ui.show_obs(obs, now)
            print(format_state_vec(obs["state_vec"]))

            # If we don't have a pending action plan, compute it once
            if actions is None or action_idx >= len(actions):
                actions = policy.get_action(obs)
                action_idx = 0
                print(f"\nComputed new action chunk: {len(actions)} steps")

            # Preview current action
            current = actions[action_idx]

            # Print raw action
            print(f"\nStep {action_idx}/{min(len(actions), cfg.action_horizon)-1}")
            ui.show_action_preview(current)

            # Print "actual targets" preview (especially for delta arms)
            targets = sender.compute_targets_for_preview(current, obs["state_vec"])
            if targets is None:
                print("NOTE: Preview unavailable (bad shapes / missing keys).")
            else:
                def s(x): return np.array2string(np.asarray(x), precision=3, floatmode="fixed")

                print("---- CURRENT (from obs state_vec) ----")
                print("left_arm   :", s(targets["left_arm_current"]))
                print("right_arm  :", s(targets["right_arm_current"]))
                print("left_grip  :", s(targets["left_gripper_current"]))
                print("right_grip :", s(targets["right_gripper_current"]))
                print("head       :", s(targets["head_current"]))
                print("lift       :", s(targets["lift_current"]))
                print("base(cmd)  :", s(targets["base_current"]))
                print("-------------------------------------")

                print("---- WILL SEND (absolute targets) ----")
                print("left_arm   :", s(targets["left_arm_target"]))
                print("right_arm  :", s(targets["right_arm_target"]))
                print("left_grip  :", s(targets["left_gripper_target"]))
                print("right_grip :", s(targets["right_gripper_target"]))
                print("head       :", s(targets["head_target"]))
                print("lift       :", s(targets["lift_target"]))
                print("base(cmd)  :", s(targets["base_target"]))
                print("-------------------------------------")

                print("---- DELTA (target - current) [sanity] ----")
                print("left_arm   :", s(targets["left_arm_delta"]))
                print("right_arm  :", s(targets["right_arm_delta"]))
                print("left_grip  :", s(targets["left_gripper_delta"]))
                print("right_grip :", s(targets["right_gripper_delta"]))
                print("head       :", s(targets["head_delta"]))
                print("lift       :", s(targets["lift_delta"]))
                print("------------------------------------------")

            # Wait for user key
            while rclpy.ok():
                rclpy.spin_once(collector, timeout_sec=0.01)
                rclpy.spin_once(sender, timeout_sec=0.0)

                now2 = time.time()
                obs2 = collector.build_obs(cfg.lang_instruction, require_cmd_vel=cfg.require_cmd_vel)
                if obs2 is not None:
                    ui.show_obs(obs2, now2)

                key = ui.get_key(delay_ms=10)
                if key == ord('q'):
                    raise KeyboardInterrupt
                if key == ord('r'):
                    # recompute policy from latest obs
                    actions = None
                    action_idx = 0
                    break
                if key == ord('n'):
                    # skip sending
                    action_idx += 1
                    break
                if key == 32:  # SPACE
                    # send
                    sender.send_action_blocks(current)
                    action_idx += 1
                    break

    except KeyboardInterrupt:
        pass
    finally:
        ui.close()
        collector.destroy_node()
        sender.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    eval()

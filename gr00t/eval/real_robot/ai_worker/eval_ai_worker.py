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
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration  # add this import

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

        - Arms / head / lift: multi-point JointTrajectory
        - Base: hold last Twist command for exec_sec
        """
        if len(action_seq) == 0:
            return

        K = len(action_seq)
        dt_per_point = exec_sec / K

        # ---------- Stack joint waypoints ----------
        left_arm = np.stack([a["left_arm"] for a in action_seq], axis=0)        # (K,7)
        left_grip = np.stack([a["left_gripper"] for a in action_seq], axis=0)   # (K,1)
        right_arm = np.stack([a["right_arm"] for a in action_seq], axis=0)
        right_grip = np.stack([a["right_gripper"] for a in action_seq], axis=0)

        head = np.stack([a["head"] for a in action_seq], axis=0)                # (K,2)
        lift = np.stack([a["lift"] for a in action_seq], axis=0)                # (K,1)

        left_pos_seq = np.concatenate([left_arm, left_grip], axis=1)            # (K,8)
        right_pos_seq = np.concatenate([right_arm, right_grip], axis=1)         # (K,8)

        # ---------- Publish joint trajectories ----------
        self.pub_left.publish(
            self._traj_msg_multi(self.left_joint_names, left_pos_seq, dt_per_point)
        )
        self.pub_right.publish(
            self._traj_msg_multi(self.right_joint_names, right_pos_seq, dt_per_point)
        )
        # self.pub_head.publish(
        #     self._traj_msg_multi(self.head_joint_names, head, dt_per_point)
        # )
        # self.pub_lift.publish(
        #     self._traj_msg_multi(self.lift_joint_names, lift, dt_per_point)
        # )

        # ---------- Base: hold last velocity for exec_sec ----------
        base = np.asarray(action_seq[-1]["base"]).reshape(3)
        tw = Twist()
        tw.linear.x = float(base[0])
        tw.linear.y = float(base[1])
        tw.angular.z = float(base[2])
        # self.pub_base.publish(tw)


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
    lang_instruction: str = "clear the items on the shelf"
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

    try:
        exec_sec = 1.0  # fixed execution per plan

        while rclpy.ok():
            rclpy.spin_once(collector, timeout_sec=0.01)
            rclpy.spin_once(sender, timeout_sec=0.0)

            obs = collector.build_obs(cfg.lang_instruction, require_cmd_vel=cfg.require_cmd_vel)
            if obs is None:
                collector.get_logger().info("Waiting for obs (rgb + joints + optional cmd_vel)...")
                continue

            actions = policy.get_action(obs)  # list length = model horizon

            # Use as many points as we have (or cap if you want)
            action_seq = actions[: cfg.action_horizon]
            print(action_seq)

            # Publish joint trajectories that execute over 1 second
            sender.send_action_sequence_1s(action_seq, exec_sec=exec_sec)

            # Wait 1 second (keep spinning so joint_states keeps updating)
            t_end = time.time() + exec_sec
            while time.time() < t_end and rclpy.ok():
                rclpy.spin_once(collector, timeout_sec=0.01)
                rclpy.spin_once(sender, timeout_sec=0.0)
                time.sleep(0.005)


    except KeyboardInterrupt:
        pass
    finally:
        collector.destroy_node()
        sender.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    eval()

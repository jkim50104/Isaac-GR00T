#!/usr/bin/env python3
"""
Move robot to initial pose from a LeRobot v2 dataset.

Standalone:
    python init_pose_from_data.py --dataset-path ./data/jkim50104/ffw_sg2_rev1_pick_item
    python init_pose_from_data.py --dataset-path ./data/jkim50104/ffw_sg2_rev1_pick_item --mode average
    python init_pose_from_data.py --dataset-path ./data/jkim50104/ffw_sg2_rev1_pick_item --episode 5

Importable (from GUI):
    from init_pose_from_data import compute_init_pose, send_init_pose
"""

import os
import sys
import json
import time
import random
import argparse

import numpy as np
import pyarrow.parquet as pq

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState, CompressedImage


# =========================================================
# Constants
# =========================================================

CONTROLLER_SLICES = {
    "arm_l": slice(0, 8),
    "arm_r": slice(8, 16),
    "head":  slice(16, 18),
    "lift":  slice(18, 19),
}

CONTROLLER_CONFIG = {
    "arm_l": {
        "joints": [
            "arm_l_joint1", "arm_l_joint2", "arm_l_joint3",
            "arm_l_joint4", "arm_l_joint5", "arm_l_joint6",
            "arm_l_joint7", "gripper_l_joint1",
        ],
        "vel": 1.0,
        "acc": 2.0,
    },
    "arm_r": {
        "joints": [
            "arm_r_joint1", "arm_r_joint2", "arm_r_joint3",
            "arm_r_joint4", "arm_r_joint5", "arm_r_joint6",
            "arm_r_joint7", "gripper_r_joint1",
        ],
        "vel": 1.0,
        "acc": 2.0,
    },
    "head": {
        "joints": ["head_joint1", "head_joint2"],
        "vel": 0.5,
        "acc": 1.0,
    },
    "lift": {
        "joints": ["lift_joint"],
        "vel": 0.1,
        "acc": 0.4,
    },
}

TOPIC_MAP = {
    "arm_l": "/leader/joint_trajectory_command_broadcaster_left/joint_trajectory",
    "arm_r": "/leader/joint_trajectory_command_broadcaster_right/joint_trajectory",
    "head":  "/leader/joystick_controller_left/joint_trajectory",
    "lift":  "/leader/joystick_controller_right/joint_trajectory",
}

# Maps controller key -> (publisher_attr, joint_names_attr) on AiWorkerCommandSender
CTRL_TO_SENDER_ATTRS = {
    "arm_l": ("pub_left",  "left_joint_names"),
    "arm_r": ("pub_right", "right_joint_names"),
    "head":  ("pub_head",  "head_joint_names"),
    "lift":  ("pub_lift",  "lift_joint_names"),
}

HEAD_CAM_TOPIC = "/zed/zed_node/left/image_rect_color/compressed"


# =========================================================
# Dataset loading (pure functions, no ROS2 needed)
# =========================================================

def load_dataset_info(dataset_path):
    meta_path = os.path.join(dataset_path, "meta", "info.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Not a valid LeRobot v2 dataset: {meta_path}")
    with open(meta_path) as f:
        return json.load(f)


def read_episodes_meta(dataset_path):
    episodes_path = os.path.join(dataset_path, "meta", "episodes.jsonl")
    episodes = []
    if os.path.exists(episodes_path):
        with open(episodes_path) as f:
            for line in f:
                episodes.append(json.loads(line))
    return episodes


def read_init_poses(dataset_path, info):
    total_episodes = info["total_episodes"]
    data_path_template = info["data_path"]
    chunks_size = info.get("chunks_size", 1000)

    init_poses = []
    for ep_idx in range(total_episodes):
        ep_chunk = ep_idx // chunks_size
        parquet_rel = data_path_template.format(
            episode_chunk=ep_chunk, episode_index=ep_idx
        )
        parquet_path = os.path.join(dataset_path, parquet_rel)
        if not os.path.exists(parquet_path):
            continue
        table = pq.read_table(parquet_path, columns=["frame_index", "observation.state"])
        states = table.column("observation.state")
        init_poses.append(np.array(states[0].as_py(), dtype=np.float32))

    return init_poses


def select_init_pose(init_poses, mode="random", episode_idx=None):
    if episode_idx is not None:
        if episode_idx < 0 or episode_idx >= len(init_poses):
            raise ValueError(
                f"Episode {episode_idx} out of range [0, {len(init_poses) - 1}]"
            )
        return init_poses[episode_idx], episode_idx

    if mode == "average":
        pose = np.mean(init_poses, axis=0)
        dists = [np.linalg.norm(p - pose) for p in init_poses]
        closest = int(np.argmin(dists))
        return pose, closest

    idx = random.randint(0, len(init_poses) - 1)
    return init_poses[idx], idx


def state_to_controller_positions(state_vector):
    positions = {}
    for ctrl_key, s in CONTROLLER_SLICES.items():
        if s.stop <= len(state_vector):
            positions[ctrl_key] = state_vector[s].tolist()
    return positions


def get_lang_instruction(dataset_path):
    """
    Read language instruction from dataset's episodes.jsonl.
    Returns the task string from the first episode, or None.
    """
    episodes = read_episodes_meta(dataset_path)
    if not episodes:
        return None
    tasks = episodes[0].get("tasks", [])
    if tasks:
        return tasks[0]
    return None


def compute_init_pose(dataset_path, mode="random", episode_idx=None):
    """
    Load dataset, pick init pose.
    Returns (positions_dict, ref_episode_idx, lang_instruction).
    """
    info = load_dataset_info(dataset_path)
    init_poses = read_init_poses(dataset_path, info)
    if not init_poses:
        raise RuntimeError(f"No episodes found in {dataset_path}")
    state_vector, ref_ep = select_init_pose(
        init_poses, mode=mode, episode_idx=episode_idx
    )
    positions = state_to_controller_positions(state_vector)

    # Get language from the selected episode
    episodes_meta = read_episodes_meta(dataset_path)
    lang = None
    if ref_ep < len(episodes_meta):
        tasks = episodes_meta[ref_ep].get("tasks", [])
        if tasks:
            lang = tasks[0]

    print(f"Init pose from episode {ref_ep}, state dim={len(state_vector)}, lang=\"{lang}\"")
    for k, v in positions.items():
        print(f"  {k}: {[round(x, 4) for x in v]}")
    return positions, ref_ep, lang


# =========================================================
# Trajectory generation
# =========================================================

def create_smooth_trajectory(
    joint_names, start_pos, end_pos, vel=1.0, acc=2.0, time_step=0.01
):
    """Quintic polynomial trajectory from start to end positions."""
    traj = JointTrajectory()
    traj.joint_names = list(joint_names)

    start = np.array(start_pos, dtype=np.float64)
    end = np.array(end_pos, dtype=np.float64)
    deltas = end - start
    max_delta = np.max(np.abs(deltas))

    if max_delta < 1e-6:
        pt = JointTrajectoryPoint()
        pt.positions = end.tolist()
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = int(0.1 * 1e9)
        traj.points.append(pt)
        return traj

    vel = max(vel, 1e-3)
    acc = max(acc, 1e-3)
    T_from_vel = max_delta * 1.875 / vel
    T_from_acc = (max_delta * 5.7735026 / acc) ** 0.5
    duration = max(T_from_vel, T_from_acc, 0.1)

    num_points = max(int(duration / time_step) + 1, 2)
    times = np.linspace(0.0, duration, num_points)

    t_norm = times / duration
    t3 = t_norm ** 3
    t4 = t3 * t_norm
    t5 = t4 * t_norm
    pos_coeffs = 10 * t3 - 15 * t4 + 6 * t5
    all_positions = start + np.outer(pos_coeffs, deltas)

    for k, t in enumerate(times):
        pt = JointTrajectoryPoint()
        pt.positions = all_positions[k].tolist()
        pt.time_from_start.sec = int(t)
        pt.time_from_start.nanosec = int((t % 1.0) * 1e9)
        traj.points.append(pt)

    return traj


# =========================================================
# Send init pose via AiWorkerCommandSender (for GUI import)
# =========================================================

def check_init_pose_reached(joint_map, target_positions, tolerance=0.05):
    """
    Check if current joint positions are within tolerance of targets.
    Returns (reached: bool, max_error: float, worst_joint: str).
    """
    max_error = 0.0
    worst_joint = ""
    for ctrl_key, target_pos in target_positions.items():
        cfg = CONTROLLER_CONFIG.get(ctrl_key)
        if cfg is None:
            continue
        joints = cfg["joints"]
        for i, jname in enumerate(joints):
            current = joint_map.get(jname, None)
            if current is None:
                return False, float("inf"), jname
            error = abs(current - target_pos[i])
            if error > max_error:
                max_error = error
                worst_joint = jname
    return max_error <= tolerance, max_error, worst_joint


def send_init_pose(sender, current_joint_map, target_positions):
    """
    Publish smooth trajectories to move robot to target positions.

    Args:
        sender: AiWorkerCommandSender node (has pub_left, pub_right, etc.)
        current_joint_map: {joint_name: float} from collector.joint_map
        target_positions: {ctrl_key: [float]} from compute_init_pose
    """
    for ctrl_key, target_pos in target_positions.items():
        if ctrl_key not in CTRL_TO_SENDER_ATTRS:
            continue
        pub_attr, joints_attr = CTRL_TO_SENDER_ATTRS[ctrl_key]
        publisher = getattr(sender, pub_attr)
        joint_names = getattr(sender, joints_attr)
        current_pos = [current_joint_map.get(j, 0.0) for j in joint_names]
        cfg = CONTROLLER_CONFIG.get(ctrl_key, {})
        traj = create_smooth_trajectory(
            joint_names, current_pos, target_pos,
            vel=cfg.get("vel", 1.0), acc=cfg.get("acc", 2.0),
        )
        publisher.publish(traj)


# =========================================================
# Standalone ROS2 Node
# =========================================================

class InitPosePlayer(Node):
    def __init__(self):
        super().__init__("init_pose_player")

        self.controllers = {}
        for ctrl_key, cfg in CONTROLLER_CONFIG.items():
            self.controllers[ctrl_key] = {
                "joints": cfg["joints"],
                "vel": cfg["vel"],
                "acc": cfg["acc"],
                "publisher": self.create_publisher(
                    JointTrajectory, TOPIC_MAP[ctrl_key], 10
                ),
                "last_positions": [0.0] * len(cfg["joints"]),
            }

        self.joint_received = False
        self.live_frame = None

        self._joint_map = {}
        for ctrl_key, ctrl in self.controllers.items():
            for i, joint in enumerate(ctrl["joints"]):
                self._joint_map[joint] = (ctrl_key, i)

        self.create_subscription(JointState, "/joint_states", self._on_joints, 10)
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            CompressedImage, HEAD_CAM_TOPIC, self._on_image, image_qos
        )

    def _on_joints(self, msg):
        for name, pos in zip(msg.name, msg.position):
            if name in self._joint_map:
                ctrl_key, i = self._joint_map[name]
                self.controllers[ctrl_key]["last_positions"][i] = float(pos)
        self.joint_received = True

    def _on_image(self, msg):
        try:
            import cv2
            np_arr = np.frombuffer(msg.data, np.uint8)
            self.live_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception:
            pass

    def apply_positions(self, target_positions):
        for ctrl_key, target_pos in target_positions.items():
            if ctrl_key not in self.controllers:
                continue
            ctrl = self.controllers[ctrl_key]
            traj = create_smooth_trajectory(
                ctrl["joints"], ctrl["last_positions"], target_pos,
                vel=ctrl["vel"], acc=ctrl["acc"],
            )
            ctrl["publisher"].publish(traj)
            ctrl["last_positions"] = list(target_pos)


# =========================================================
# Visual comparison (standalone only)
# =========================================================

def extract_reference_frame(dataset_path, info, episode_idx,
                            video_key="observation.images.ego_view"):
    import cv2
    video_path_template = info.get("video_path")
    if not video_path_template:
        return None
    chunks_size = info.get("chunks_size", 1000)
    ep_chunk = episode_idx // chunks_size

    # Try requested key, then fallback to cam_head
    for vk in [video_key, "observation.images.cam_head"]:
        video_rel = video_path_template.format(
            episode_chunk=ep_chunk, episode_index=episode_idx, video_key=vk,
        )
        video_path = os.path.join(dataset_path, video_rel)
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                return frame

    print("WARNING: No reference video found")
    return None


def compare_images(reference, live):
    import cv2
    from skimage.metrics import structural_similarity as ssim

    if reference.shape[:2] != live.shape[:2]:
        live = cv2.resize(live, (reference.shape[1], reference.shape[0]))

    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    live_gray = cv2.cvtColor(live, cv2.COLOR_BGR2GRAY)
    score, diff = ssim(ref_gray, live_gray, full=True)
    diff_uint8 = (255 - (diff * 255)).astype(np.uint8)
    heatmap = cv2.applyColorMap(diff_uint8, cv2.COLORMAP_JET)

    threshold = 50
    mask = diff_uint8 > threshold
    diff_overlay = live.copy()
    diff_overlay[mask] = cv2.addWeighted(live, 0.4, heatmap, 0.6, 0)[mask]
    blend = cv2.addWeighted(reference, 0.5, live, 0.5, 0)

    h, w = reference.shape[:2]
    pad, label_h = 4, 32
    cell_h = label_h + h
    grid = np.zeros((cell_h * 2 + pad, w * 2 + pad, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    ssim_color = (
        (0, 255, 0) if score > 0.8
        else (0, 165, 255) if score > 0.6
        else (0, 0, 255)
    )

    cv2.putText(grid, "Reference", (4, label_h - 6), font, 0.6, (255, 255, 255), 2)
    grid[label_h:label_h + h, 0:w] = reference

    x1 = w + pad
    cv2.putText(grid, "Live", (x1 + 4, label_h - 6), font, 0.6, (255, 255, 255), 2)
    grid[label_h:label_h + h, x1:x1 + w] = live

    y1 = cell_h + pad
    cv2.putText(
        grid, f"Diff SSIM={score:.3f}", (4, y1 + label_h - 6),
        font, 0.6, ssim_color, 2,
    )
    grid[y1 + label_h:y1 + label_h + h, 0:w] = diff_overlay

    cv2.putText(
        grid, "Ref+Live Blend", (x1 + 4, y1 + label_h - 6),
        font, 0.6, (255, 255, 255), 2,
    )
    grid[y1 + label_h:y1 + label_h + h, x1:x1 + w] = blend

    return score, grid


# =========================================================
# Main (standalone)
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Move robot to init pose from LeRobot v2 dataset"
    )
    parser.add_argument("--dataset-path", required=True, help="Path to dataset")
    parser.add_argument(
        "--mode", choices=["random", "average"], default="random",
        help="Init pose selection mode (default: random)",
    )
    parser.add_argument("--episode", type=int, default=None, help="Specific episode")
    parser.add_argument("--no-visual", action="store_true", help="Skip visual comparison")
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_path):
        print(f"ERROR: Dataset not found: {args.dataset_path}")
        sys.exit(1)

    # Load dataset
    info = load_dataset_info(args.dataset_path)
    episodes_meta = read_episodes_meta(args.dataset_path)
    init_poses = read_init_poses(args.dataset_path, info)
    if not init_poses:
        print("ERROR: No episodes found")
        sys.exit(1)

    state_vector, ref_episode = select_init_pose(
        init_poses, mode=args.mode, episode_idx=args.episode,
    )
    positions = state_to_controller_positions(state_vector)

    task_name = "unknown"
    if ref_episode < len(episodes_meta) and "tasks" in episodes_meta[ref_episode]:
        task_name = ", ".join(episodes_meta[ref_episode]["tasks"])

    print(f"\nTask: {task_name}")
    print(f"Target positions (ref episode {ref_episode}):")
    for ctrl_key, pos in positions.items():
        print(f"  {ctrl_key}: {[round(v, 4) for v in pos]}")

    ref_frame = None
    if not args.no_visual:
        ref_frame = extract_reference_frame(args.dataset_path, info, ref_episode)

    # Send to robot
    rclpy.init()
    node = InitPosePlayer()

    print("\nWaiting for /joint_states...")
    t0 = time.time()
    while rclpy.ok() and not node.joint_received:
        rclpy.spin_once(node, timeout_sec=0.5)
        if time.time() - t0 > 5:
            print("WARNING: /joint_states not received, using zeros.")
            break

    print("Sending init pose...")
    node.apply_positions(positions)

    t_end = time.time() + 3.0
    while rclpy.ok() and time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.1)

    print("Robot should be at init pose.")

    # Visual comparison
    if ref_frame is not None:
        import cv2

        print(f"\nVisual comparison (press 'q' to quit, 's' to save)")
        window_title = f"Init Pose - {task_name} (ep {ref_episode})"
        cv2.imshow(window_title, ref_frame)
        cv2.waitKey(1)

        print("Waiting for live camera...")
        t0 = time.time()
        while rclpy.ok() and node.live_frame is None:
            rclpy.spin_once(node, timeout_sec=0.1)
            cv2.waitKey(1)
            if time.time() - t0 > 30:
                print("WARNING: No camera frame after 30s.")
                break

        if node.live_frame is not None:
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.03)
                if node.live_frame is None:
                    continue
                score, grid = compare_images(ref_frame, node.live_frame)
                cv2.imshow(window_title, grid)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('s'):
                    path = f"init_pose_comparison_ep{ref_episode}.png"
                    cv2.imwrite(path, grid)
                    print(f"Saved: {path} (SSIM={score:.3f})")

        cv2.destroyAllWindows()

    print("Done.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

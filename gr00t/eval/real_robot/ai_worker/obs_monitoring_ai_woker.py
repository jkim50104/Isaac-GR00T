#!/usr/bin/env python3
import time
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CompressedImage, JointState
from geometry_msgs.msg import Twist

import cv2
import numpy as np
from cv_bridge import CvBridge


class Ros2ObservationMonitor(Node):
    """
    ROS2 Observation Monitor (debugger)

    Shows:
      - Side-by-side RGB: RAW (left) vs COMPRESSED (right)
      - Joint states from /joint_states
      - cmd_vel (and limited_cmd_vel if available)

    Controls:
      - Press 'q' in the OpenCV window to quit.
    """

    def __init__(self):
        super().__init__("ros2_observation_monitor")

        # -----------------------------
        # Topics
        # -----------------------------
        # NOTE: your current graph uses realsense-style topics, so update these
        # if you are not using ZED anymore.
        
        view_point = "stereo"
        self.topic_raw = f"/zed/zed_node/{view_point}/image_rect_color"
        self.topic_comp = f"/zed/zed_node/{view_point}/image_rect_color/compressed"
        
        # self.topic_raw = "/camera_left/camera_left/color/image_rect_raw"
        # self.topic_comp = "/camera_left/camera_left/color/image_rect_raw/compressed"

        self.topic_joint_states = "/joint_states"

        self.topic_cmd_vel = "/cmd_vel"
        self.topic_limited_cmd_vel = "/limited_cmd_vel"

        # -----------------------------
        # Joint groups (policy-relevant)
        # -----------------------------
        self.controllers: Dict[str, List[str]] = {
            "arm_l": [
                "arm_l_joint1", "arm_l_joint2", "arm_l_joint3",
                "arm_l_joint4", "arm_l_joint5", "arm_l_joint6",
                "arm_l_joint7", "gripper_l_joint1"
            ],
            "arm_r": [
                "arm_r_joint1", "arm_r_joint2", "arm_r_joint3",
                "arm_r_joint4", "arm_r_joint5", "arm_r_joint6",
                "arm_r_joint7", "gripper_r_joint1"
            ],
            "head": ["head_joint1", "head_joint2"],
            "lift": ["lift_joint"],
        }

        # Optional: limit overlay clutter (None = all listed joints)
        self.max_joints_per_group: Optional[int] = None

        # -----------------------------
        # OpenCV window
        # -----------------------------
        self.window = "ROS2 OBS | RAW (L) | COMP (R) | JOINTS + CMD_VEL  [q to quit]"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)

        # -----------------------------
        # State
        # -----------------------------
        self.bridge = CvBridge()

        self.raw_bgr: Optional[np.ndarray] = None
        self.comp_bgr: Optional[np.ndarray] = None
        self.raw_time: Optional[float] = None
        self.comp_time: Optional[float] = None

        self.joint_map: Dict[str, float] = {}
        self.joint_time: Optional[float] = None

        # cmd_vel tracking
        self.cmd_vel: Optional[Tuple[float, float, float, float, float, float]] = None
        self.cmd_vel_time: Optional[float] = None

        self.limited_cmd_vel: Optional[Tuple[float, float, float, float, float, float]] = None
        self.limited_cmd_vel_time: Optional[float] = None

        # Logging throttle
        self.log_hz = 2.0
        self.log_period = 1.0 / self.log_hz
        self._last_log = 0.0

        # -----------------------------
        # Subscribers
        # -----------------------------
        self.create_subscription(Image, self.topic_raw, self.on_raw, 10)
        self.create_subscription(CompressedImage, self.topic_comp, self.on_comp, 10)
        self.create_subscription(JointState, self.topic_joint_states, self.on_joint_states, 10)

        self.create_subscription(Twist, self.topic_cmd_vel, self.on_cmd_vel, 10)
        # limited_cmd_vel may or may not publish; subscribing is harmless
        self.create_subscription(Twist, self.topic_limited_cmd_vel, self.on_limited_cmd_vel, 10)

        # Render loop
        self.create_timer(0.01, self.tick)

        self.get_logger().info("ROS2 Observation Monitor started")
        self.get_logger().info(f"  RAW:              {self.topic_raw}")
        self.get_logger().info(f"  COMPRESSED:       {self.topic_comp}")
        self.get_logger().info(f"  JOINTS:           {self.topic_joint_states}")
        self.get_logger().info(f"  CMD_VEL:          {self.topic_cmd_vel}")
        self.get_logger().info(f"  LIMITED_CMD_VEL:  {self.topic_limited_cmd_vel}")

    # --------------------------------------------------
    # Callbacks
    # --------------------------------------------------
    def on_raw(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.raw_bgr = self._to_bgr(cv_img, msg.encoding)
            self.raw_time = time.time()
        except Exception as e:
            self.get_logger().warn(f"Raw image convert failed: {e}")

    def on_comp(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            decoded = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            if decoded is None:
                return

            if decoded.ndim == 2:
                decoded = cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR)
            elif decoded.shape[2] == 4:
                decoded = cv2.cvtColor(decoded, cv2.COLOR_BGRA2BGR)

            self.comp_bgr = decoded
            self.comp_time = time.time()
        except Exception as e:
            self.get_logger().warn(f"Compressed decode failed: {e}")

    def on_joint_states(self, msg: JointState):
        n = min(len(msg.name), len(msg.position))
        for i in range(n):
            self.joint_map[msg.name[i]] = msg.position[i]
        self.joint_time = time.time()

    def on_cmd_vel(self, msg: Twist):
        self.cmd_vel = (msg.linear.x, msg.linear.y, msg.linear.z,
                        msg.angular.x, msg.angular.y, msg.angular.z)
        self.cmd_vel_time = time.time()

    def on_limited_cmd_vel(self, msg: Twist):
        self.limited_cmd_vel = (msg.linear.x, msg.linear.y, msg.linear.z,
                                msg.angular.x, msg.angular.y, msg.angular.z)
        self.limited_cmd_vel_time = time.time()

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def _to_bgr(self, img: np.ndarray, encoding: str) -> np.ndarray:
        enc = (encoding or "").lower()

        if enc in ("bgra8", "bgra"):
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if enc in ("rgba8", "rgba"):
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        if enc in ("rgb8", "rgb"):
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if enc in ("bgr8", "bgr"):
            return img
        if enc in ("mono8", "8uc1"):
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def _format_joint_lines(self) -> List[str]:
        lines: List[str] = []
        for group, joints in self.controllers.items():
            js = joints[: self.max_joints_per_group] if self.max_joints_per_group else joints

            parts = []
            for j in js:
                if j in self.joint_map:
                    parts.append(f"{j.split('_')[-1]}={self.joint_map[j]:+.2f}")
                else:
                    parts.append(f"{j.split('_')[-1]}=?")

            lines.append(f"{group}: " + " ".join(parts))
        return lines

    def _format_twist_line(self, label: str,
                           twist: Optional[Tuple[float, float, float, float, float, float]],
                           age: Optional[float],
                           x_offset: int,
                           y: int,
                           img: np.ndarray) -> int:
        if twist is None or age is None:
            text = f"{label}: (no msgs)"
        else:
            lx, ly, lz, ax, ay, az = twist
            text = (f"{label} age={age:.2f}s  "
                    f"lin=({lx:+.2f},{ly:+.2f},{lz:+.2f}) "
                    f"ang=({ax:+.2f},{ay:+.2f},{az:+.2f})")

        cv2.putText(img, text, (x_offset, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        return y + 24

    def _draw_overlay(self, img: np.ndarray, now: float) -> np.ndarray:
        out = img

        raw_age = now - self.raw_time if self.raw_time else None
        comp_age = now - self.comp_time if self.comp_time else None
        joint_age = now - self.joint_time if self.joint_time else None
        cmd_age = now - self.cmd_vel_time if self.cmd_vel_time else None
        lcmd_age = now - self.limited_cmd_vel_time if self.limited_cmd_vel_time else None

        header = (
            f"raw={raw_age:.3f}s  " if raw_age is not None else "raw=?  "
        ) + (
            f"comp={comp_age:.3f}s  " if comp_age is not None else "comp=?  "
        ) + (
            f"joint={joint_age:.3f}s" if joint_age is not None else "joint=?"
        )

        y = 28
        cv2.putText(out, header, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Twist info (cmd_vel + limited_cmd_vel)
        y += 28
        y = self._format_twist_line("cmd_vel", self.cmd_vel, cmd_age, 10, y, out)
        y = self._format_twist_line("limited", self.limited_cmd_vel, lcmd_age, 10, y, out)

        # Joint lines
        y += 4
        for line in self._format_joint_lines():
            cv2.putText(out, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            y += 22

        return out

    # --------------------------------------------------
    # Render loop
    # --------------------------------------------------
    def tick(self):
        if self.raw_bgr is None or self.comp_bgr is None:
            return

        # Make heights match
        h = min(self.raw_bgr.shape[0], self.comp_bgr.shape[0])
        raw = self.raw_bgr[:h, :]
        comp = self.comp_bgr[:h, :]

        # Match widths if needed
        if raw.shape[1] != comp.shape[1]:
            comp = cv2.resize(comp, (raw.shape[1], h), interpolation=cv2.INTER_AREA)

        combined = cv2.hconcat([raw, comp])
        now = time.time()
        combined = self._draw_overlay(combined, now)

        cv2.imshow(self.window, combined)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.get_logger().info("Quit requested")
            rclpy.shutdown()
            return

        # Throttled terminal log
        if now - self._last_log >= self.log_period:
            self._last_log = now
            self._log_status(now)

    def _log_status(self, now: float):
        raw_age = (now - self.raw_time) if self.raw_time else None
        comp_age = (now - self.comp_time) if self.comp_time else None
        joint_age = (now - self.joint_time) if self.joint_time else None
        cmd_age = (now - self.cmd_vel_time) if self.cmd_vel_time else None

        msg = "[STATUS] "
        msg += f"raw_age={raw_age:.3f}s " if raw_age is not None else "raw_age=? "
        msg += f"comp_age={comp_age:.3f}s " if comp_age is not None else "comp_age=? "
        msg += f"joint_age={joint_age:.3f}s " if joint_age is not None else "joint_age=? "
        msg += f"cmd_vel_age={cmd_age:.3f}s" if cmd_age is not None else "cmd_vel_age=?"

        self.get_logger().info(msg)

    def destroy_node(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = Ros2ObservationMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()

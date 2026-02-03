#!/usr/bin/env python3
import time
import subprocess
import importlib
from collections import deque
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


def get_topic_type(topic: str) -> str:
    """
    Returns ROS 2 type string like 'sensor_msgs/msg/Image'
    """
    out = subprocess.check_output(["ros2", "topic", "type", topic], text=True).strip()
    # Some setups return multiple types; first token is usually the actual type
    return out.splitlines()[0].split()[0]


def import_msg_class(type_str: str):
    """
    Convert 'sensor_msgs/msg/Image' -> sensor_msgs.msg.Image
    """
    pkg, _, name = type_str.partition("/msg/")
    if not name:
        raise RuntimeError(f"Unexpected type format: {type_str}")
    module = importlib.import_module(f"{pkg}.msg")
    return getattr(module, name)


class TopicHzMonitor(Node):
    def __init__(self):
        super().__init__("topic_hz_monitor")

        self.topics = [
            "/zed/zed_node/left/image_rect_color/compressed",
            "/camera_left/camera_left/color/image_rect_raw/compressed",
            "/camera_right/camera_right/color/image_rect_raw/compressed",
            "/joint_states",
            "/cmd_vel",
        ]

        self.buffers = {t: deque(maxlen=200) for t in self.topics}

        # Good default for camera/image streams and many sensors
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        for topic in self.topics:
            try:
                t = get_topic_type(topic)
                msg_cls = import_msg_class(t)
                self.get_logger().info(f"Subscribing {topic} [{t}]")
                self.create_subscription(
                    msg_cls,
                    topic,
                    lambda _msg, tt=topic: self.cb(tt),
                    qos,
                )
            except Exception as e:
                self.get_logger().error(f"Failed to subscribe {topic}: {e}")

        self.create_timer(1.0, self.report)

    def cb(self, topic: str):
        self.buffers[topic].append(time.time())

    def report(self):
        os.system("clear")

        print(f"{'Topic':<60} Hz")
        print("-" * 72)

        for topic, stamps in self.buffers.items():
            if len(stamps) < 2:
                print(f"{topic:<60} {'no data'}")
                continue

            periods = [stamps[i] - stamps[i - 1] for i in range(1, len(stamps))]
            hz = 1.0 / (sum(periods) / len(periods))
            print(f"{topic:<60} {hz:6.2f}")



def main():
    rclpy.init()
    node = TopicHzMonitor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

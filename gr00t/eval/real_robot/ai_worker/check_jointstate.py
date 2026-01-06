#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointStatesInspector(Node):
    """
    Minimal node to inspect exactly what is published on /joint_states.
    """

    def __init__(self):
        super().__init__("joint_states_inspector")

        self.create_subscription(
            JointState,
            "/joint_states",
            self.callback,
            10,
        )

        self.get_logger().info("JointStatesInspector started")
        self.get_logger().info("Listening to /joint_states ...")

    def callback(self, msg: JointState):
        self.get_logger().info("---- /joint_states ----")

        n = len(msg.name)
        self.get_logger().info(f"joint count: {n}")

        for i in range(n):
            name = msg.name[i]

            pos = msg.position[i] if i < len(msg.position) else None
            vel = msg.velocity[i] if i < len(msg.velocity) else None
            eff = msg.effort[i] if i < len(msg.effort) else None

            self.get_logger().info(
                f"[{i:02d}] {name:25s} "
                f"pos={pos!s:>8} "
                f"vel={vel!s:>8} "
                f"eff={eff!s:>8}"
            )

        self.get_logger().info("------------------------\n")


def main():
    rclpy.init()
    node = JointStatesInspector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

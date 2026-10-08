#!/usr/bin/python3
"""Isaac-only endpoint adaptation for the upstream OpenArmX VR node."""

from __future__ import annotations

import rclpy
from openarmx_teleop_vr.openarmx_teleop_vr_node import OpenArmXTeleopVRNode


HARDWARE_LEFT_EE_FRAME = "openarmx_left_link7_pico"
HARDWARE_RIGHT_EE_FRAME = "openarmx_right_link7_pico"


class OpenFlexIsaacVRArmNode(OpenArmXTeleopVRNode):
    def __init__(self) -> None:
        super().__init__()
        self.declare_parameter("isaac_left_ee_frame", HARDWARE_LEFT_EE_FRAME)
        self.declare_parameter("isaac_right_ee_frame", HARDWARE_RIGHT_EE_FRAME)
        left_frame = str(self.get_parameter("isaac_left_ee_frame").value)
        right_frame = str(self.get_parameter("isaac_right_ee_frame").value)
        for core in self.core_by_mode.values():
            solver = core.ik_solver
            for frame_name in (left_frame, right_frame):
                if not solver.model.existFrame(frame_name):
                    raise ValueError(
                        f"Isaac IK end-effector frame not found: {frame_name}"
                    )
            # Keep the same tool frames as the known-good hardware IK path in
            # every input mode. The generated Isaac URDF carries these frames.
            solver.left_ee_frame = left_frame
            solver.right_ee_frame = right_frame
            solver.left_ee_id = solver.model.getFrameId(left_frame)
            solver.right_ee_id = solver.model.getFrameId(right_frame)
            solver.reset()
        self.get_logger().info(
            f"Isaac IK endpoints (hardware-compatible): {left_frame}, {right_frame}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OpenFlexIsaacVRArmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

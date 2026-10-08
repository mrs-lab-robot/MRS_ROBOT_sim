from unittest.mock import patch

from mrs_robot_arena_bridge import relay_node


class FakeNode:
    def __init__(self):
        self.destroyed = False

    def destroy_node(self):
        self.destroyed = True


def test_main_tolerates_rclpy_context_shutdown_by_signal_handler():
    node = FakeNode()
    with (
        patch.object(relay_node, "ArenaRelayNode", return_value=node),
        patch.object(relay_node.rclpy, "spin", side_effect=lambda _node: relay_node.rclpy.shutdown()),
    ):
        relay_node.main()

    assert node.destroyed
    assert not relay_node.rclpy.ok()

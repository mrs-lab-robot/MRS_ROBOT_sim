"""Small Kit window for VR teleop and manual episode capture controls."""

from __future__ import annotations

from collections import deque


class TeleopPanel:
    """UI callbacks enqueue actions; the simulation loop consumes them safely."""

    def __init__(self) -> None:
        import omni.ui as ui

        self._ui = ui
        self._events: deque[str] = deque()
        self.window = ui.Window("OpenFlex VR Teleoperation", width=420, height=260)
        with self.window.frame:
            with ui.VStack(spacing=8, height=0):
                self.status_label = ui.Label("等待 ROS 2 VR 桥接…", word_wrap=True, height=42)
                with ui.HStack(spacing=8, height=34):
                    ui.Button("开始采集", clicked_fn=lambda: self._events.append("record_start"))
                    ui.Button("保存 Episode", clicked_fn=lambda: self._events.append("record_save"))
                    ui.Button("丢弃", clicked_fn=lambda: self._events.append("record_discard"))
                with ui.HStack(spacing=8, height=34):
                    ui.Button("机器人复位", clicked_fn=lambda: self._events.append("reset"))
                    ui.Button("急停 / 解除", clicked_fn=lambda: self._events.append("estop_toggle"))
                ui.Label("VR 遥操作需先启动 ROS relay；失联时底盘自动归零，关节保持。", word_wrap=True)

    def pop_events(self) -> list[str]:
        events = list(self._events)
        self._events.clear()
        return events

    def set_status(self, message: str) -> None:
        self.status_label.text = message

    def close(self) -> None:
        self.window.visible = False


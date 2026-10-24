from __future__ import annotations

from contextlib import redirect_stderr
import io
import importlib.util
from pathlib import Path
import struct
import sys
import time
import unittest
from types import SimpleNamespace


_SCRIPT = Path(__file__).resolve().parents[3] / "arena_benchmark" / "scripts" / "livox_ros2_relay.py"
_SPEC = importlib.util.spec_from_file_location("livox_ros2_relay", _SCRIPT)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load Livox relay module: {_SCRIPT}")
_RELAY = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _RELAY
_SPEC.loader.exec_module(_RELAY)


class FakeCustomPoint:
    def __init__(self) -> None:
        self.offset_time = 0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.reflectivity = 0
        self.tag = 0
        self.line = 0


class FakeCustomMsg:
    def __init__(self) -> None:
        self.header = None
        self.timebase = 0
        self.point_num = 0
        self.lidar_id = 0
        self.rsvd = [0, 0, 0]
        self.points = []


class LivoxRelayConversionTest(unittest.TestCase):
    def test_pointcloud_fields_become_livox_points_with_frame_relative_times(self) -> None:
        fields = [
            SimpleNamespace(name="x", offset=0, datatype=7, count=1),
            SimpleNamespace(name="y", offset=4, datatype=7, count=1),
            SimpleNamespace(name="z", offset=8, datatype=7, count=1),
            SimpleNamespace(name="intensity", offset=12, datatype=7, count=1),
            SimpleNamespace(name="timestamp", offset=16, datatype=7, count=1),
            SimpleNamespace(name="channelId", offset=20, datatype=2, count=1),
        ]
        data = b"".join(
            struct.pack("<fffffB3x", *point)
            for point in (
                (1.25, -2.5, 0.75, 19.0, 0.01, 4),
                (-0.5, 0.25, 1.5, 300.0, 0.02, 7),
            )
        )
        header = SimpleNamespace(
            stamp=SimpleNamespace(sec=12, nanosec=345), frame_id="source_frame"
        )
        cloud = SimpleNamespace(
            header=header,
            fields=fields,
            data=data,
            height=1,
            width=2,
            point_step=24,
            row_step=48,
            is_bigendian=False,
        )

        message = _RELAY.pointcloud_to_custom_msg(
            cloud,
            custom_msg_type=FakeCustomMsg,
            custom_point_type=FakeCustomPoint,
            frame_id="livox_frame",
        )

        self.assertEqual(message.header.frame_id, "livox_frame")
        self.assertEqual(message.timebase, 12_000_000_345)
        self.assertEqual(message.point_num, 2)
        self.assertEqual(message.rsvd, [0, 0, 0])
        self.assertEqual(
            [
                (
                    point.offset_time,
                    point.x,
                    point.y,
                    point.z,
                    point.reflectivity,
                    point.line,
                )
                for point in message.points
            ],
            [
                (10_000_000, 1.25, -2.5, 0.75, 19, 4),
                (20_000_000, -0.5, 0.25, 1.5, 255, 7),
            ],
        )

    def test_pointcloud_without_optional_fields_uses_safe_livox_defaults(self) -> None:
        cloud = SimpleNamespace(
            header=SimpleNamespace(
                stamp=SimpleNamespace(sec=3, nanosec=4), frame_id="source_frame"
            ),
            fields=[
                SimpleNamespace(name="x", offset=0, datatype=7, count=1),
                SimpleNamespace(name="y", offset=4, datatype=7, count=1),
                SimpleNamespace(name="z", offset=8, datatype=7, count=1),
            ],
            data=struct.pack("<fff", 0.1, 0.2, 0.3),
            height=1,
            width=1,
            point_step=12,
            row_step=12,
            is_bigendian=False,
        )

        message = _RELAY.pointcloud_to_custom_msg(
            cloud,
            custom_msg_type=FakeCustomMsg,
            custom_point_type=FakeCustomPoint,
        )

        self.assertEqual(message.point_num, 1)
        self.assertEqual(message.timebase, 3_000_000_004)
        self.assertEqual(message.points[0].offset_time, 0)
        self.assertEqual(message.points[0].reflectivity, 0)
        self.assertEqual(message.points[0].line, 0)

    def test_absolute_float_timestamp_is_rebased_to_first_point(self) -> None:
        fields = [
            SimpleNamespace(name="x", offset=0, datatype=7, count=1),
            SimpleNamespace(name="y", offset=4, datatype=7, count=1),
            SimpleNamespace(name="z", offset=8, datatype=7, count=1),
            SimpleNamespace(name="timestamp", offset=12, datatype=8, count=1),
        ]
        data = b"".join(
            struct.pack("<fff d", x, 0.0, 0.0, timestamp)
            for x, timestamp in ((1.0, 12.001), (2.0, 12.002))
        )
        cloud = SimpleNamespace(
            header=SimpleNamespace(
                stamp=SimpleNamespace(sec=12, nanosec=0), frame_id="source_frame"
            ),
            fields=fields,
            data=data,
            height=1,
            width=2,
            point_step=20,
            row_step=40,
            is_bigendian=False,
        )

        message = _RELAY.pointcloud_to_custom_msg(
            cloud,
            custom_msg_type=FakeCustomMsg,
            custom_point_type=FakeCustomPoint,
        )

        self.assertEqual(message.timebase, 12_001_000_000)
        self.assertEqual([point.offset_time for point in message.points], [0, 1_000_000])

    def test_pointcloud_without_xyz_is_rejected(self) -> None:
        cloud = SimpleNamespace(fields=[], data=b"", height=1, width=0, point_step=0, row_step=0)

        with self.assertRaisesRegex(ValueError, "x, y, and z"):
            _RELAY.pointcloud_to_custom_msg(
                cloud,
                custom_msg_type=FakeCustomMsg,
                custom_point_type=FakeCustomPoint,
            )

    def test_managed_relay_waits_for_ready_signal_and_stops_its_child(self) -> None:
        relay = _RELAY.ManagedLivoxRelay.start(
            [
                sys.executable,
                "-u",
                "-c",
                "import time; print('ARENA_LIVOX_RELAY_READY test', flush=True); time.sleep(30)",
            ]
        )
        child_pid = relay.pid
        try:
            self.assertTrue(relay.wait_ready(timeout_s=2.0))
            self.assertEqual(relay.pid, child_pid)
        finally:
            relay.stop()

        self.assertIsNotNone(relay.returncode)
        self.assertNotEqual(relay.returncode, 0)

    def test_relay_data_readiness_requires_a_published_pointcloud(self) -> None:
        relay = _RELAY.ManagedLivoxRelay.start(
            [
                sys.executable,
                "-u",
                "-c",
                "import time; print('ARENA_LIVOX_RELAY_READY test', flush=True); "
                "time.sleep(0.05); print('ARENA_LIVOX_RELAY_DATA_READY test', flush=True); "
                "time.sleep(30)",
            ]
        )
        try:
            self.assertTrue(relay.wait_ready(timeout_s=2.0))
            self.assertTrue(relay.wait_data_ready(timeout_s=2.0))
        finally:
            relay.stop()

    def test_managed_relay_forwards_every_line_of_a_child_traceback(self) -> None:
        child_code = (
            "import sys\n"
            "sys.stderr.write('Traceback (most recent call last):\\n')\n"
            "sys.stderr.write('  File \\\"relay.py\\\", line 7, in main\\n')\n"
            "sys.stderr.write('RuntimeError: relay boom\\n')\n"
            "sys.stderr.flush()\n"
            "sys.exit(1)\n"
        )
        relay = _RELAY.ManagedLivoxRelay.start(
            [sys.executable, "-u", "-c", child_code]
        )
        captured = io.StringIO()
        with redirect_stderr(captured):
            try:
                deadline = time.monotonic() + 2.0
                while relay.returncode is None and time.monotonic() < deadline:
                    time.sleep(0.005)
            finally:
                relay.stop()

        self.assertEqual(relay.returncode, 1)
        self.assertIn("Traceback (most recent call last):", captured.getvalue())
        self.assertIn('File "relay.py", line 7, in main', captured.getvalue())
        self.assertIn("RuntimeError: relay boom", captured.getvalue())

    def test_managed_relay_stops_child_with_sigint_for_graceful_ros_shutdown(self) -> None:
        child_code = (
            "import signal, sys, time\n"
            "signal.signal(signal.SIGINT, lambda *_: sys.exit(0))\n"
            "print('ARENA_LIVOX_RELAY_READY', flush=True)\n"
            "while True: time.sleep(0.05)\n"
        )
        relay = _RELAY.ManagedLivoxRelay.start(
            [sys.executable, "-u", "-c", child_code]
        )
        self.assertTrue(relay.wait_ready(timeout_s=2.0))

        relay.stop()

        self.assertEqual(relay.returncode, 0)


if __name__ == "__main__":
    unittest.main()

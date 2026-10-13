#!/usr/bin/env python3
"""静态检查性能脚本是否遵守固定报告和批次证据规范。"""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]


class ReportingLayoutTest(unittest.TestCase):
    def test_reporting_spec_contains_required_template_and_boundaries(self) -> None:
        text = (ROOT / "docs" / "REPORTING_CN.md").read_text(encoding="utf-8")
        for required in (
            "统一批次模板",
            "功能结论",
            "实时结论",
            "YYYYMMDD-HHMMSS-类别-执行者",
            "阶段 4 未授权时始终保持“停止”",
        ):
            self.assertIn(required, text)

    def test_empty_scene_script_uses_its_own_fixed_report_and_evidence_directory(self) -> None:
        text = (ROOT / "test" / "performance" / "empty_scene_performance.py").read_text(encoding="utf-8")
        self.assertIn('REPORT_PATH = REPORT_DIR / "empty_scene_performance.md"', text)
        self.assertIn('RAW_ROOT = REPORT_DIR / "raw" / "empty_scene"', text)
        self.assertIn('batch_dir / "result.json"', text)
        self.assertNotIn('robot-control performance report', text)

    def test_fixed_reports_use_chinese_conclusion_states(self) -> None:
        for relative_path in (
            "reports/performance/robot_control_performance.md",
            "reports/performance/empty_scene_performance.md",
            "reports/sensors/sensor_validation.md",
        ):
            text = (ROOT / relative_path).read_text(encoding="utf-8")
            self.assertNotIn("**PASS**", text)
            self.assertNotIn("**FAIL", text)
            self.assertNotIn("**NOT_", text)

    def test_control_report_has_separate_automatic_append_sections(self) -> None:
        text = (ROOT / "reports/performance/robot_control_performance.md").read_text(encoding="utf-8")
        self.assertIn("### 功能验证（自动追加）", text)
        self.assertIn("<!-- ROBOT_CONTROL_FULL_BATCHES -->", text)
        self.assertIn("### 实时性能与底盘验证（自动追加）", text)
        self.assertIn("<!-- ROBOT_CONTROL_BASELINE_BATCHES -->", text)


if __name__ == "__main__":
    unittest.main()

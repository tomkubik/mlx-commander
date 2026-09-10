"""
Unit tests for MCP server tool endpoints.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlx_commander.mcp_server import (
    convert_dataset_headless_tool,
    inspect_dataset_tool,
    launch_conversion_tui_tool,
    run_mcp_server,
)
from tests.conftest import make_sample_qa_records


class TestMcpServer(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.src_file = Path(self.temp_dir) / "source.jsonl"
        self.out_dir = Path(self.temp_dir) / "mlx_output"

        records = make_sample_qa_records(20)
        with open(self.src_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_inspect_dataset_tool(self):
        result = inspect_dataset_tool(str(self.src_file))
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["total_rows"], 20)
        self.assertIn("instruction", result["columns"])
        self.assertIn("output", result["columns"])
        self.assertIn("suggested_mappings", result)
        self.assertIn("prompt_completion", result["suggested_mappings"])

    def test_convert_dataset_headless_tool(self):
        result = convert_dataset_headless_tool(
            dataset_path=str(self.src_file),
            format="prompt_completion",
            prompt_col="instruction",
            completion_col="output",
            train_pct=80.0,
            valid_pct=20.0,
            test_pct=0.0,
            output_dir=str(self.out_dir),
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["total_records"], 20)
        self.assertEqual(result["splits"]["train"], 16)
        self.assertEqual(result["splits"]["valid"], 4)
        self.assertTrue(Path(result["output_dir"]).exists())

    @patch("mlx_commander.mcp_server.is_macos", return_value=True)
    @patch("mlx_commander.mcp_server.spawn_terminal_tui")
    def test_launch_conversion_tui_tool(self, mock_spawn, mock_is_mac):
        mock_spawn.return_value = 0
        manifest_path = self.out_dir / "mlx_manifest.json"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"status": "success", "total_records": 20}, f)

        res = launch_conversion_tui_tool(
            dataset_path=str(self.src_file),
            format="prompt_completion",
            output_dir=str(self.out_dir),
        )

        mock_spawn.assert_called_once()
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["total_records"], 20)

    @patch("mlx_commander.mcp_server.is_macos", return_value=False)
    def test_launch_conversion_tui_tool_non_macos(self, mock_is_mac):
        res = launch_conversion_tui_tool(
            dataset_path=str(self.src_file),
            format="prompt_completion",
            output_dir=str(self.out_dir),
        )
        self.assertEqual(res["status"], "error")
        self.assertIn("supported on macOS", res["message"])

    def test_run_mcp_server_missing_dep(self):
        # In environment without mcp installed, run_mcp_server exits with 1
        with patch.dict("sys.modules", {"mcp": None, "mcp.server.fastmcp": None}):
            ec = run_mcp_server()
            self.assertEqual(ec, 1)


if __name__ == "__main__":
    unittest.main()

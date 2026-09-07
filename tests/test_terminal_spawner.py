"""
Unit tests for terminal_spawner helper functions.
"""

import json
import os
import tempfile
import unittest

from mlx_commander.terminal_spawner import build_terminal_script, is_macos


class TestTerminalSpawner(unittest.TestCase):

    def test_is_macos_boolean(self):
        self.assertIsInstance(is_macos(), bool)

    def test_build_terminal_script(self):
        script = build_terminal_script(
            python_exe="/usr/bin/python3",
            args=["--dataset", "data.parquet", "--format", "chat"],
            status_file="/tmp/test_status.json",
            working_dir="/Users/test",
        )

        self.assertIn("cd /Users/test", script)
        self.assertIn("/usr/bin/python3 -m mlx_commander", script)
        self.assertIn("--tui", script)
        self.assertIn("--dataset data.parquet", script)
        self.assertIn("--format chat", script)
        self.assertIn("/tmp/test_status.json", script)
        self.assertIn("finished", script)


if __name__ == "__main__":
    unittest.main()

"""
Unit tests for GUI picker module and path selection defaults.
"""

import os
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

from hf2mlx.gui_picker import (
    is_macos,
    pick_dataset_gui,
    pick_file_gui,
    pick_folder_gui,
    run_runtime_compiled_picker,
)


class TestGUIPicker(unittest.TestCase):

    def test_is_macos_detection(self):
        self.assertIsInstance(is_macos(), bool)

    @patch("subprocess.run")
    @patch("hf2mlx.gui_picker._compile_picker_at_runtime")
    def test_runtime_picker_selection(self, mock_bin, mock_run):
        mock_bin.return_value = "/tmp/hf2mlx_picker_test"
        mock_run.return_value = MagicMock(returncode=0, stdout="/Users/tomkubik/my_dataset\n", stderr="")
        res = run_runtime_compiled_picker(mode="both", default_dir="/Users/tomkubik")
        self.assertEqual(res, "/Users/tomkubik/my_dataset")

    @patch("subprocess.run")
    @patch("hf2mlx.gui_picker._compile_picker_at_runtime")
    def test_runtime_picker_cancelled(self, mock_bin, mock_run):
        mock_bin.return_value = "/tmp/hf2mlx_picker_test"
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        res = run_runtime_compiled_picker(mode="both", default_dir="/Users/tomkubik")
        self.assertIsNone(res)

    @patch("hf2mlx.gui_picker.run_runtime_compiled_picker")
    def test_pick_dataset_gui_success(self, mock_picker):
        mock_picker.return_value = "/Users/tomkubik/dataset"
        with patch("hf2mlx.gui_picker.is_macos", return_value=True):
            res = pick_dataset_gui()
            self.assertEqual(res, "/Users/tomkubik/dataset")

    @patch("hf2mlx.gui_picker.run_runtime_compiled_picker")
    def test_pick_dataset_gui_cancelled_no_second_window(self, mock_picker):
        mock_picker.return_value = None
        with patch("hf2mlx.gui_picker.is_macos", return_value=True):
            res = pick_dataset_gui()
            self.assertIsNone(res)


if __name__ == "__main__":
    unittest.main()

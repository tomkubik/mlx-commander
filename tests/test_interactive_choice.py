import unittest
from unittest.mock import MagicMock, patch

from hf2mlx.tui.wizard_fallback import (
    _ask_choice_fallback,
    _ask_choice_interactive,
    ask_choice,
)


class TestInteractiveChoice(unittest.TestCase):
    def setUp(self):
        self.options = ["Option A", "Option B", "Option C", "Option D"]

    @patch("hf2mlx.tui.wizard_fallback.sys.stdout.write")
    @patch("hf2mlx.tui.wizard_fallback.sys.stdout.flush")
    @patch("hf2mlx.tui.wizard_fallback.tty.setcbreak")
    @patch("hf2mlx.tui.wizard_fallback.termios.tcgetattr")
    @patch("hf2mlx.tui.wizard_fallback.termios.tcsetattr")
    @patch("hf2mlx.tui.wizard_fallback.sys.stdin.fileno", return_value=0)
    def test_arrow_down_and_enter(self, mock_fn, mock_tcset, mock_tcget, mock_cbreak, mock_flush, mock_write):
        with patch("hf2mlx.tui.wizard_fallback._read_char", side_effect=["DOWN", "ENTER"]):
            res = _ask_choice_interactive("Pick One", self.options, default_idx=0)
            self.assertEqual(res, 1)

    @patch("hf2mlx.tui.wizard_fallback.sys.stdout.write")
    @patch("hf2mlx.tui.wizard_fallback.sys.stdout.flush")
    @patch("hf2mlx.tui.wizard_fallback.tty.setcbreak")
    @patch("hf2mlx.tui.wizard_fallback.termios.tcgetattr")
    @patch("hf2mlx.tui.wizard_fallback.termios.tcsetattr")
    @patch("hf2mlx.tui.wizard_fallback.sys.stdin.fileno", return_value=0)
    def test_arrow_up_wrap_and_enter(self, mock_fn, mock_tcset, mock_tcget, mock_cbreak, mock_flush, mock_write):
        with patch("hf2mlx.tui.wizard_fallback._read_char", side_effect=["UP", "ENTER"]):
            # From 0, UP wraps to index 3 (Option D)
            res = _ask_choice_interactive("Pick One", self.options, default_idx=0)
            self.assertEqual(res, 3)

    @patch("hf2mlx.tui.wizard_fallback.sys.stdout.write")
    @patch("hf2mlx.tui.wizard_fallback.sys.stdout.flush")
    @patch("hf2mlx.tui.wizard_fallback.tty.setcbreak")
    @patch("hf2mlx.tui.wizard_fallback.termios.tcgetattr")
    @patch("hf2mlx.tui.wizard_fallback.termios.tcsetattr")
    @patch("hf2mlx.tui.wizard_fallback.sys.stdin.fileno", return_value=0)
    def test_number_jump_and_enter(self, mock_fn, mock_tcset, mock_tcget, mock_cbreak, mock_flush, mock_write):
        with patch("hf2mlx.tui.wizard_fallback._read_char", side_effect=["3", "ENTER"]):
            # '3' jumps to Option C (index 2)
            res = _ask_choice_interactive("Pick One", self.options, default_idx=0)
            self.assertEqual(res, 2)

    @patch("hf2mlx.tui.wizard_fallback.sys.stdout.write")
    @patch("hf2mlx.tui.wizard_fallback.sys.stdout.flush")
    @patch("hf2mlx.tui.wizard_fallback.tty.setcbreak")
    @patch("hf2mlx.tui.wizard_fallback.termios.tcgetattr")
    @patch("hf2mlx.tui.wizard_fallback.termios.tcsetattr")
    @patch("hf2mlx.tui.wizard_fallback.sys.stdin.fileno", return_value=0)
    def test_multi_digit_number(self, mock_fn, mock_tcset, mock_tcget, mock_cbreak, mock_flush, mock_write):
        long_opts = [f"Col {i}" for i in range(1, 16)]
        with patch("hf2mlx.tui.wizard_fallback._read_char", side_effect=["1", "2", "ENTER"]):
            # Typing '1' then '2' selects Col 12 (index 11)
            res = _ask_choice_interactive("Pick Col", long_opts, default_idx=0)
            self.assertEqual(res, 11)

    @patch("hf2mlx.tui.wizard_fallback.sys.stdout.write")
    @patch("hf2mlx.tui.wizard_fallback.sys.stdout.flush")
    @patch("hf2mlx.tui.wizard_fallback.tty.setcbreak")
    @patch("hf2mlx.tui.wizard_fallback.termios.tcgetattr")
    @patch("hf2mlx.tui.wizard_fallback.termios.tcsetattr")
    @patch("hf2mlx.tui.wizard_fallback.sys.stdin.fileno", return_value=0)
    def test_number_then_arrow_mix(self, mock_fn, mock_tcset, mock_tcget, mock_cbreak, mock_flush, mock_write):
        with patch("hf2mlx.tui.wizard_fallback._read_char", side_effect=["2", "DOWN", "ENTER"]):
            # '2' jumps to index 1, DOWN moves to index 2
            res = _ask_choice_interactive("Pick One", self.options, default_idx=0)
            self.assertEqual(res, 2)

    @patch("hf2mlx.tui.wizard_fallback.sys.stdout.write")
    @patch("hf2mlx.tui.wizard_fallback.sys.stdout.flush")
    @patch("hf2mlx.tui.wizard_fallback.tty.setcbreak")
    @patch("hf2mlx.tui.wizard_fallback.termios.tcgetattr")
    @patch("hf2mlx.tui.wizard_fallback.termios.tcsetattr")
    @patch("hf2mlx.tui.wizard_fallback.sys.stdin.fileno", return_value=0)
    def test_ctrl_c_raises_keyboard_interrupt(self, mock_fn, mock_tcset, mock_tcget, mock_cbreak, mock_flush, mock_write):
        with patch("hf2mlx.tui.wizard_fallback._read_char", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                _ask_choice_interactive("Pick One", self.options, default_idx=0)
            # Guarantee tcsetattr was called to restore terminal
            self.assertTrue(mock_tcset.called)

    def test_non_tty_fallback(self):
        with patch("hf2mlx.tui.wizard_fallback.sys.stdin.isatty", return_value=False):
            with patch("hf2mlx.tui.wizard_fallback.ask_input", return_value="2"):
                res = ask_choice("Pick One", self.options, default_idx=0)
                self.assertEqual(res, 1)


if __name__ == "__main__":
    unittest.main()

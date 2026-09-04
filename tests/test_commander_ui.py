import curses
import unittest
from unittest.mock import MagicMock, patch

from hf2mlx.formats import MLXFormat
from hf2mlx.tui.app import run_commander_tui
from hf2mlx.tui.state import ActivePanel, CommanderState
from hf2mlx.tui.widgets import (
    show_column_picker_dialog,
    show_help_dialog,
    show_results_dialog,
    show_text_edit_dialog,
)


class TestCommanderUI(unittest.TestCase):
    def setUp(self):
        self.mock_win = MagicMock()
        self.mock_win.getmaxyx.return_value = (30, 100)

    @patch("hf2mlx.tui.app.curses.has_colors", return_value=False)
    @patch("hf2mlx.tui.app.init_colors")
    @patch("hf2mlx.tui.app.curses.curs_set")
    def test_run_commander_tui_quit_q(self, mock_curs, mock_colors, mock_has_colors):
        # Simulate pressing 'q' immediately to exit
        self.mock_win.getch.side_effect = [ord("q")]
        res = run_commander_tui(self.mock_win)
        self.assertIsNone(res)
        self.assertTrue(self.mock_win.erase.called)
        self.assertTrue(self.mock_win.refresh.called)

    @patch("hf2mlx.tui.app.curses.has_colors", return_value=False)
    @patch("hf2mlx.tui.app.init_colors")
    @patch("hf2mlx.tui.app.curses.curs_set")
    def test_run_commander_tui_tab_and_arrow(self, mock_curs, mock_colors, mock_has_colors):
        # Press Tab (switches panel), Right arrow (cycles format), then 'q'
        self.mock_win.getch.side_effect = [
            9,                  # Tab
            curses.KEY_RIGHT,   # Right arrow
            ord("q"),           # Quit
        ]
        res = run_commander_tui(self.mock_win)
        self.assertIsNone(res)

    @patch("hf2mlx.tui.app.curses.has_colors", return_value=False)
    @patch("hf2mlx.tui.app.init_colors")
    @patch("hf2mlx.tui.app.curses.curs_set")
    def test_run_commander_tui_too_small_window(self, mock_curs, mock_colors, mock_has_colors):
        self.mock_win.getmaxyx.return_value = (10, 40)
        self.mock_win.getch.side_effect = [ord("q")]
        res = run_commander_tui(self.mock_win)
        self.assertIsNone(res)

    @patch("hf2mlx.tui.app.curses.has_colors", return_value=False)
    @patch("hf2mlx.tui.app.init_colors")
    @patch("hf2mlx.tui.app.curses.curs_set")
    def test_run_commander_tui_help_dialog(self, mock_curs, mock_colors, mock_has_colors):
        # Press '?' (help), close help with Enter, then 'q' to quit
        self.mock_win.getch.side_effect = [
            ord("?"),           # Help
            10,                 # Close help
            ord("q"),           # Quit
        ]
        res = run_commander_tui(self.mock_win)
        self.assertIsNone(res)

    @patch("hf2mlx.tui.app.curses.has_colors", return_value=False)
    @patch("hf2mlx.tui.app.init_colors")
    @patch("hf2mlx.tui.app.curses.curs_set")
    def test_run_commander_tui_randomize_seed(self, mock_curs, mock_colors, mock_has_colors):
        # Press 'r' (randomize seed), then 'q'
        self.mock_win.getch.side_effect = [
            ord("r"),           # Randomize
            ord("q"),           # Quit
        ]
        res = run_commander_tui(self.mock_win)
        self.assertIsNone(res)

    def test_show_column_picker_dialog(self):
        # Test selecting option 1 with Down then Enter (fast single-select without space)
        self.mock_win.getch.side_effect = [curses.KEY_DOWN, 10]
        chosen = show_column_picker_dialog(self.mock_win, "Pick Column", ["col_a", "col_b", "col_c"])
        self.assertEqual(chosen, "col_b")

    def test_show_column_picker_dialog_multi_select(self):
        # Select col_a with Space, navigate down to col_b, select with Space, then Enter
        self.mock_win.getch.side_effect = [
            32,                 # Space on col_a
            curses.KEY_DOWN,    # Move to col_b
            32,                 # Space on col_b
            10,                 # Enter to confirm
        ]
        chosen = show_column_picker_dialog(self.mock_win, "Pick Column", ["col_a", "col_b", "col_c"])
        self.assertEqual(chosen, "col_a + col_b")

    def test_show_column_picker_dialog_ordering(self):
        # Select col_b first, then col_a second -> order should be col_b + col_a
        self.mock_win.getch.side_effect = [
            curses.KEY_DOWN,    # Move to col_b
            32,                 # Space on col_b
            curses.KEY_UP,      # Move up to col_a
            32,                 # Space on col_a
            10,                 # Enter to confirm
        ]
        chosen = show_column_picker_dialog(self.mock_win, "Pick Column", ["col_a", "col_b", "col_c"])
        self.assertEqual(chosen, "col_b + col_a")

    def test_show_column_picker_dialog_deselect(self):
        # Select col_a, select col_b, deselect col_b -> only col_a remains
        self.mock_win.getch.side_effect = [
            32,                 # Space on col_a
            curses.KEY_DOWN,    # Move to col_b
            32,                 # Space on col_b
            32,                 # Space on col_b again (deselect)
            10,                 # Enter to confirm
        ]
        chosen = show_column_picker_dialog(self.mock_win, "Pick Column", ["col_a", "col_b", "col_c"])
        self.assertEqual(chosen, "col_a")

    def test_show_column_picker_dialog_esc_cancel(self):
        # Press ESC (27) to cancel
        self.mock_win.getch.side_effect = [27]
        chosen = show_column_picker_dialog(self.mock_win, "Pick Column", ["col_a", "col_b"], current_val="col_a")
        self.assertEqual(chosen, "col_a")

    def test_show_text_edit_dialog_esc_cancel(self):
        # Press ESC (27) to cancel
        self.mock_win.getch.side_effect = [27]
        val = show_text_edit_dialog(self.mock_win, "Edit", "Prompt", default_val="original")
        self.assertIsNone(val)

    def test_show_help_dialog_esc(self):
        # Press ESC (27) to dismiss help
        self.mock_win.getch.side_effect = [27]
        show_help_dialog(self.mock_win)
        self.assertTrue(self.mock_win.refresh.called)

    def test_show_results_dialog_esc(self):
        from pathlib import Path
        from hf2mlx.converter import ConversionResult
        res = ConversionResult(
            output_dir=Path("/tmp/out"),
            format_type=MLXFormat.PROMPT_COMPLETION,
            output_files={"train": Path("/tmp/out/train.jsonl")},
            record_counts={"train": 10},
            file_sizes={"train": 500},
            sample_records={"train": [{"prompt": "hi", "completion": "hello"}]},
            seed_used=42,
        )
        self.mock_win.getch.side_effect = [27]
        show_results_dialog(self.mock_win, res)
        self.assertTrue(self.mock_win.refresh.called)

    def test_configure_escdelay(self):
        import os
        from hf2mlx.tui.widgets import configure_escdelay

        configure_escdelay(25)
        self.assertEqual(os.environ.get("ESCDELAY"), "25")
        if hasattr(curses, "get_escdelay"):
            # If curses supports get_escdelay
            try:
                self.assertEqual(curses.get_escdelay(), 25)
            except curses.error:
                pass


if __name__ == "__main__":
    unittest.main()

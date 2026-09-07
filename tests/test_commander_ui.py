import curses
import unittest
from unittest.mock import MagicMock, patch

from mlx_commander.formats import MLXFormat
from mlx_commander.tui.app import run_commander_tui
from mlx_commander.tui.state import ActivePanel, CommanderState
from mlx_commander.tui.widgets import (
    show_column_picker_dialog,
    show_error_dialog,
    show_help_dialog,
    show_message_dialog,
    show_results_dialog,
    show_text_edit_dialog,
)


class TestCommanderUI(unittest.TestCase):
    def setUp(self):
        self.mock_win = MagicMock()
        self.mock_win.getmaxyx.return_value = (30, 100)

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=False)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_run_commander_tui_quit_q(self, mock_curs, mock_colors, mock_has_colors):
        # Simulate pressing 'q' immediately to exit
        self.mock_win.getch.side_effect = [ord("q")]
        res = run_commander_tui(self.mock_win)
        self.assertIsNone(res)
        self.assertTrue(self.mock_win.erase.called)
        self.assertTrue(self.mock_win.refresh.called)

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=False)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_run_commander_tui_tab_and_arrow(self, mock_curs, mock_colors, mock_has_colors):
        # Press Tab (switches panel), Right arrow (cycles format), then 'q'
        self.mock_win.getch.side_effect = [
            9,                  # Tab
            curses.KEY_RIGHT,   # Right arrow
            ord("q"),           # Quit
        ]
        res = run_commander_tui(self.mock_win)
        self.assertIsNone(res)

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=False)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_run_commander_tui_too_small_window(self, mock_curs, mock_colors, mock_has_colors):
        self.mock_win.getmaxyx.return_value = (10, 40)
        self.mock_win.getch.side_effect = [ord("q")]
        res = run_commander_tui(self.mock_win)
        self.assertIsNone(res)

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=False)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_run_commander_tui_help_dialog(self, mock_curs, mock_colors, mock_has_colors):
        # Press '?' (help), close help with Enter, then 'q' to quit
        self.mock_win.getch.side_effect = [
            ord("?"),           # Help
            10,                 # Close help
            ord("q"),           # Quit
        ]
        res = run_commander_tui(self.mock_win)
        self.assertIsNone(res)

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=False)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
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
        from mlx_commander.converter import ConversionResult
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

    def test_show_error_dialog_dismiss_enter(self):
        self.mock_win.getch.side_effect = [10]  # Enter key
        show_error_dialog(self.mock_win, "Test Error", "An unexpected failure occurred.")
        self.assertTrue(self.mock_win.refresh.called)
        # Verify double line borders were drawn
        has_double_border = any("╔" in str(call) for call in self.mock_win.addstr.call_args_list)
        self.assertTrue(has_double_border)

    def test_show_error_dialog_dismiss_esc(self):
        self.mock_win.getch.side_effect = [27]  # Esc key
        show_error_dialog(self.mock_win, "Test Error", ["Line 1", "Line 2"])
        self.assertTrue(self.mock_win.refresh.called)

    def test_show_results_dialog_resilience_to_legacy_call(self):
        # Verify calling with (stdscr, False, "error message") doesn't crash with TypeError
        self.mock_win.getch.side_effect = [10]
        show_results_dialog(self.mock_win, False, "Dataset Loading Failed")
        self.assertTrue(self.mock_win.refresh.called)

    def test_show_message_dialog_overlay(self):
        self.mock_win.getch.side_effect = [10]
        show_message_dialog(self.mock_win, "Notice", ["Everything is okay."])
        self.assertTrue(self.mock_win.refresh.called)

    def test_configure_escdelay(self):
        import os
        from mlx_commander.tui.widgets import configure_escdelay

        configure_escdelay(25)
        self.assertEqual(os.environ.get("ESCDELAY"), "25")
        if hasattr(curses, "get_escdelay"):
            # If curses supports get_escdelay
            try:
                self.assertEqual(curses.get_escdelay(), 25)
            except curses.error:
                pass


    @patch("mlx_commander.tui.app.curses.has_colors", return_value=False)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_run_commander_tui_wrapped_preview(self, mock_curs, mock_colors, mock_has_colors):
        state = CommanderState()
        long_text = "Word " * 50
        state.preview_cache = [
            f'{{"prompt": "{long_text}", "completion": "Answer 1"}}',
            f'{{"prompt": "Example 2", "completion": "{long_text}"}}',
            f'{{"prompt": "Example 3", "completion": "Answer 3"}}',
        ]
        self.mock_win.getmaxyx.return_value = (35, 90)
        self.mock_win.getch.side_effect = [ord("q")]
        res = run_commander_tui(self.mock_win, initial_state=state)
        self.assertIsNone(res)
        self.assertTrue(self.mock_win.refresh.called)

    def test_get_schema_mapping_targets_prompt_completion(self):
        from mlx_commander.formats import ColumnMapping, MLXFormat
        from mlx_commander.tui.widgets import get_schema_mapping_targets

        mapping = ColumnMapping(prompt_col="instruction + input", completion_col="output")
        columns = ["id", "instruction", "input", "output", "system_prompt"]
        targets, unmapped = get_schema_mapping_targets(MLXFormat.PROMPT_COMPLETION, mapping, columns)

        self.assertEqual(len(targets), 2)
        self.assertEqual(targets[0]["key"], "prompt")
        self.assertEqual(targets[0]["cols"], ["instruction", "input"])
        self.assertTrue(targets[0]["is_concat"])

        self.assertEqual(targets[1]["key"], "completion")
        self.assertEqual(targets[1]["cols"], ["output"])
        self.assertFalse(targets[1]["is_concat"])

        self.assertEqual(unmapped, ["id", "system_prompt"])

    def test_get_schema_mapping_targets_formats(self):
        from mlx_commander.formats import ColumnMapping, MLXFormat
        from mlx_commander.tui.widgets import get_schema_mapping_targets

        # Chat format
        chat_map = ColumnMapping(user_col="query", assistant_col="response", system_col="sys")
        cols = ["sys", "query", "response", "extra"]
        targets, unmapped = get_schema_mapping_targets(MLXFormat.CHAT, chat_map, cols)
        self.assertEqual([t["key"] for t in targets], ["user", "assistant", "system"])
        self.assertEqual(unmapped, ["extra"])

        # DPO format
        dpo_map = ColumnMapping(dpo_prompt_col="prompt_text", chosen_col="chosen_text", rejected_col="rejected_text")
        dpo_cols = ["prompt_text", "chosen_text", "rejected_text", "meta"]
        targets, unmapped = get_schema_mapping_targets(MLXFormat.DPO, dpo_map, dpo_cols)
        self.assertEqual([t["key"] for t in targets], ["prompt", "chosen", "rejected"])
        self.assertEqual(unmapped, ["meta"])

        # Text format
        text_map = ColumnMapping(text_col="raw_body")
        targets, unmapped = get_schema_mapping_targets(MLXFormat.TEXT, text_map, ["raw_body"])
        self.assertEqual([t["key"] for t in targets], ["text"])
        self.assertEqual(unmapped, [])

    def test_plan_mapping_panel_rows(self):
        from mlx_commander.tui.widgets import plan_mapping_panel_rows

        targets = [
            {"key": "prompt", "cols": ["instruction", "input"]},
            {"key": "completion", "cols": ["output"]},
        ]
        unmapped = ["id", "meta"]

        # Compact space: 5 rows (fits borders and 1 row per target)
        heights, show_unmapped = plan_mapping_panel_rows(targets, unmapped, avail_rows=5)
        self.assertEqual(heights, [1, 1])
        self.assertFalse(show_unmapped)

        # Space with room for unmapped: 7 rows
        heights, show_unmapped = plan_mapping_panel_rows(targets, unmapped, avail_rows=7)
        self.assertEqual(heights, [1, 1])
        self.assertTrue(show_unmapped)

        # Generous space: 9 rows (gives extra row to multi-column target)
        heights, show_unmapped = plan_mapping_panel_rows(targets, unmapped, avail_rows=9)
        self.assertEqual(heights, [2, 1])
        self.assertTrue(show_unmapped)

    def test_draw_mapping_pipeline_panel_no_emojis(self):
        from mlx_commander.formats import ColumnMapping, MLXFormat
        from mlx_commander.loader import LoadedDataset
        from mlx_commander.tui.widgets import draw_mapping_pipeline_panel

        state = CommanderState()
        state.target_format = MLXFormat.PROMPT_COMPLETION
        state.mapping = ColumnMapping(prompt_col="instruction + input", completion_col="output")
        state.loaded_dataset = LoadedDataset(
            source_path="/tmp/test.jsonl",
            is_split=False,
            split_names=["train"],
            split_counts={"train": 10},
            columns=["instruction", "input", "output", "id"],
            total_rows=10,
            sample_records=[{"instruction": "say hi", "input": "now", "output": "hello", "id": 1}],
        )

        printed_strings = []

        def fake_addstr(y, x, s, attr=0):
            printed_strings.append(s)

        self.mock_win.addstr.side_effect = fake_addstr
        self.mock_win.getmaxyx.return_value = (30, 90)

        # Test panel rendering across multiple heights
        for h in [6, 8, 12]:
            printed_strings.clear()
            draw_mapping_pipeline_panel(self.mock_win, y=10, x=0, h=h, w=90, state=state)
            self.assertTrue(len(printed_strings) > 0)

            # Assert ZERO emojis exist in any rendered string
            all_text = "".join(printed_strings)
            for ch in all_text:
                code = ord(ch)
                # Ensure no high Unicode emoji block characters
                is_emoji = (
                    0x1F300 <= code <= 0x1F9FF
                    or 0x2600 <= code <= 0x27BF
                    or 0x1F600 <= code <= 0x1F64F
                )
                self.assertFalse(
                    is_emoji,
                    f"Found forbidden emoji character '{ch}' (code {code:X}) in mapping panel output!",
                )

        # Verify that Target tags exist, but Source tags are removed from the left side
        self.assertFalse(any("Source: prompt" in s for s in printed_strings))
        self.assertFalse(any("Source: completion" in s for s in printed_strings))
        self.assertTrue(any("Target: prompt" in s for s in printed_strings))
        self.assertTrue(any("Target: completion" in s for s in printed_strings))
        self.assertTrue(any("instruction" in s for s in printed_strings))
        self.assertTrue(any("output" in s for s in printed_strings))
        self.assertTrue(any("Concat" in s for s in printed_strings))

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=False)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_run_commander_tui_multi_file_tip(self, mock_curs, mock_colors, mock_has_colors):
        printed_strings = []

        def fake_addstr(y, x, s, attr=0):
            printed_strings.append(s)

        self.mock_win.addstr.side_effect = fake_addstr
        self.mock_win.getmaxyx.return_value = (30, 100)
        self.mock_win.getch.side_effect = [ord("q")]

        run_commander_tui(self.mock_win)
        # Verify multi-file tip is drawn
        has_tip = any("Tip: You can load multiple files" in s for s in printed_strings)
        self.assertTrue(has_tip, "Multi-file loading tip was not found in TUI rendered output!")

    def test_commander_state_output_dir_default_and_custom(self):
        import json
        import tempfile
        from pathlib import Path
        from mlx_commander.tui.state import CommanderState

        temp_d = tempfile.mkdtemp()
        sub_d = Path(temp_d) / "my_hf_data"
        sub_d.mkdir(parents=True)
        sample_file = sub_d / "data.jsonl"
        with open(sample_file, "w") as f:
            f.write(json.dumps({"prompt": "p", "completion": "c"}) + "\n")

        state = CommanderState()
        # Loading dataset should set output_dir to <dataset_dir>/mlx_dataset
        state.load_dataset(str(sample_file))
        expected_default = str((sub_d / "mlx_dataset").resolve())
        self.assertEqual(state.output_dir, expected_default)
        self.assertFalse(state.has_custom_output_dir)

        # Customizing output_dir should be preserved on subsequent loads
        custom_out = str(Path(temp_d) / "my_custom_mlx")
        state.output_dir = custom_out
        state.has_custom_output_dir = True

        state.load_dataset(str(sample_file))
        self.assertEqual(state.output_dir, custom_out)

    def test_show_output_destination_dialog_esc(self):
        from mlx_commander.tui.widgets import show_output_destination_dialog
        self.mock_win.getch.side_effect = [27]  # Esc
        res = show_output_destination_dialog(self.mock_win, "/tmp/curr", "/tmp/default")
        self.assertIsNone(res)

    @patch("mlx_commander.gui_picker.is_macos", return_value=False)
    def test_show_output_destination_dialog_reset(self, mock_is_mac):
        from mlx_commander.tui.widgets import show_output_destination_dialog
        # On non-mac: option 0 = Manual, option 1 = Reset to default
        self.mock_win.getch.side_effect = [curses.KEY_DOWN, 10]
        res = show_output_destination_dialog(self.mock_win, "/tmp/curr", "/tmp/default")
        self.assertEqual(res, "/tmp/default")


if __name__ == "__main__":
    unittest.main()


import curses
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from mlx_commander.lora import LoraRunConfig, QueueManager
from mlx_commander.tui.app import run_commander_tui
from mlx_commander.tui.state import CommanderState
from mlx_commander.tui.widgets import (
    draw_queue_table,
    show_choice_dialog,
    show_model_picker_dialog,
)


class TestLoraUI(unittest.TestCase):
    def setUp(self):
        self.mock_win = MagicMock()
        self.mock_win.getmaxyx.return_value = (35, 120)
        self.temp_dir = tempfile.mkdtemp()
        self.queue_dir = Path(self.temp_dir) / "test_runs"

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=False)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_switch_to_lora_mode_f4(self, mock_curs, mock_colors, mock_has_colors):
        state = CommanderState()
        state.queue_manager = QueueManager(self.queue_dir)

        # Start in mode 0, press F4 (switches to mode 1), then 'q' to quit
        self.mock_win.getch.side_effect = [
            curses.KEY_F4,
            ord("q"),
        ]
        res = run_commander_tui(self.mock_win, initial_state=state)
        self.assertIsNone(res)
        self.assertEqual(state.active_tab, 1)
        self.assertIn("LoRA", state.status_message)

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=False)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_add_run_to_queue_f6(self, mock_curs, mock_colors, mock_has_colors):
        state = CommanderState()
        state.active_tab = 1
        state.queue_manager = QueueManager(self.queue_dir)
        state.lora_config.model = "mlx-community/Llama-3.2-3B-Instruct-4bit"

        # Press F6 to add to queue, then 'q'
        self.mock_win.getch.side_effect = [
            curses.KEY_F6,
            ord("q"),
        ]
        run_commander_tui(self.mock_win, initial_state=state)
        self.assertEqual(len(state.queue_manager.runs), 1)
        self.assertEqual(state.queue_manager.runs[0].model, "mlx-community/Llama-3.2-3B-Instruct-4bit")

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=False)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_queue_actions_clone_and_delete(self, mock_curs, mock_colors, mock_has_colors):
        state = CommanderState()
        state.active_tab = 1
        state.lora_active_panel = "queue"
        state.queue_manager = QueueManager(self.queue_dir)
        run1 = LoraRunConfig(id="r1", name="Original Run")
        state.queue_manager.add_run(run1)

        # Press 'c' to clone, then 'd' to delete, then 'q'
        self.mock_win.getch.side_effect = [
            ord("c"),  # Clones run1 -> now 2 runs
            ord("d"),  # Deletes selected run -> back to 1 run
            ord("q"),
        ]
        run_commander_tui(self.mock_win, initial_state=state)
        self.assertEqual(len(state.queue_manager.runs), 1)

    @patch("mlx_commander.tui.app.curses.has_colors", return_value=False)
    @patch("mlx_commander.tui.app.init_colors")
    @patch("mlx_commander.tui.app.curses.curs_set")
    def test_queue_clear_action_x(self, mock_curs, mock_colors, mock_has_colors):
        state = CommanderState()
        state.active_tab = 1
        state.lora_active_panel = "queue"
        state.queue_manager = QueueManager(self.queue_dir)
        state.queue_manager.add_run(LoraRunConfig(id="r1"))
        state.queue_manager.add_run(LoraRunConfig(id="r2"))
        self.assertEqual(len(state.queue_manager.runs), 2)

        # Press 'x' to clear queue, then 'q'
        self.mock_win.getch.side_effect = [
            ord("x"),
            ord("q"),
        ]
        run_commander_tui(self.mock_win, initial_state=state)
        self.assertEqual(len(state.queue_manager.runs), 0)

    def test_show_model_picker_dialog(self):
        # Press Down (select 2nd model) then Enter
        self.mock_win.getch.side_effect = [curses.KEY_DOWN, 10]
        chosen = show_model_picker_dialog(self.mock_win)
        self.assertIsNotNone(chosen)

    def test_show_choice_dialog(self):
        # Select option 1 with Down then Enter
        options = ["adamw", "adam", "sgd"]
        self.mock_win.getch.side_effect = [curses.KEY_DOWN, 10]
        chosen = show_choice_dialog(self.mock_win, "Optimizer", "Select optimizer:", options, current_val="adamw")
        self.assertEqual(chosen, "adam")

    def test_draw_queue_table(self):
        runs = [
            LoraRunConfig(id="r1", name="Run 1", model="mlx-community/Llama-3.2-1B-Instruct-4bit", iters=500, batch_size=4, status="queued"),
            LoraRunConfig(id="r2", name="Run 2", model="mlx-community/Qwen2.5-7B-Instruct-4bit", iters=1000, batch_size=2, status="running"),
        ]
        selected = draw_queue_table(
            self.mock_win,
            2,
            2,
            8,
            100,
            runs,
            selected_idx=0,
            is_focused=True,
        )
        self.assertEqual(selected, 0)

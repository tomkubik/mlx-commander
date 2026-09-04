import json
import tempfile
import unittest
from pathlib import Path

from hf2mlx.formats import MLXFormat
from hf2mlx.tui.state import ActivePanel, CommanderState


class TestCommanderState(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.src_file = Path(self.tmp_dir.name) / "test_data.jsonl"
        data = [
            {"id": i, "question": f"Question {i}", "answer": f"Answer {i}", "extra": "info"}
            for i in range(20)
        ]
        with open(self.src_file, "w") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_initial_defaults(self):
        state = CommanderState()
        self.assertEqual(state.target_format, MLXFormat.PROMPT_COMPLETION)
        self.assertEqual(state.train_pct, 80.0)
        self.assertEqual(state.valid_pct, 10.0)
        self.assertEqual(state.test_pct, 10.0)
        self.assertEqual(state.active_panel, ActivePanel.RIGHT)

    def test_load_dataset_and_auto_mapping(self):
        state = CommanderState()
        success = state.load_dataset(str(self.src_file))
        self.assertTrue(success)
        self.assertIsNotNone(state.loaded_dataset)
        self.assertEqual(state.loaded_dataset.total_rows, 20)
        self.assertIn("question", state.loaded_dataset.columns)
        self.assertIn("answer", state.loaded_dataset.columns)

        # Auto detection for prompt completion
        self.assertEqual(state.mapping.prompt_col, "question")
        self.assertEqual(state.mapping.completion_col, "answer")

        # Preview should have generated live
        self.assertTrue(len(state.preview_cache) > 0)
        rec0 = json.loads(state.preview_cache[0])
        self.assertEqual(rec0["prompt"], "Question 0")
        self.assertEqual(rec0["completion"], "Answer 0")

    def test_reactive_preview_on_format_change(self):
        state = CommanderState()
        state.load_dataset(str(self.src_file))

        # Change format to TEXT
        state.set_format(MLXFormat.TEXT)
        self.assertEqual(state.target_format, MLXFormat.TEXT)
        # Check mapping and preview
        state.set_mapping_field("text_col", "question")
        self.assertTrue(len(state.preview_cache) > 0)
        rec = json.loads(state.preview_cache[0])
        self.assertEqual(rec["text"], "Question 0")

    def test_split_counts_calculation(self):
        state = CommanderState()
        state.load_dataset(str(self.src_file))
        counts = state.get_split_counts()
        self.assertEqual(counts["train"] + counts["valid"] + counts["test"], 20)

    def test_randomize_seed(self):
        state = CommanderState()
        old_seed = state.seed
        # Calling randomize should set a valid 6-digit seed
        state.randomize_seed()
        self.assertGreaterEqual(state.seed, 100000)
        self.assertLessEqual(state.seed, 999999)


    def test_concatenated_preview_in_state(self):
        state = CommanderState()
        state.load_dataset(str(self.src_file))
        # Concatenate question and extra
        state.set_mapping_field("prompt_col", "question + extra")
        self.assertIsNone(state.preview_error)
        self.assertTrue(len(state.preview_cache) > 0)
        rec0 = json.loads(state.preview_cache[0])
        self.assertEqual(rec0["prompt"], "Question 0\n\ninfo")


if __name__ == "__main__":
    unittest.main()

"""
Unit tests for CommanderState prefill and custom mapping preservation.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from mlx_commander.formats import ColumnMapping, MLXFormat
from mlx_commander.tui.state import ActivePanel, CommanderState
from tests.conftest import make_sample_qa_records


class TestTuiPrefill(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.src_file = Path(self.temp_dir) / "source.jsonl"
        self.out_dir = Path(self.temp_dir) / "custom_out"

        records = make_sample_qa_records(30)
        with open(self.src_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_apply_prefill_dict(self):
        state = CommanderState()
        prefill = {
            "dataset": str(self.src_file),
            "format": "chat",
            "prompt_col": "instruction",
            "completion_col": "output",
            "train": 85.0,
            "valid": 15.0,
            "test": 0.0,
            "seed": 999,
            "output": str(self.out_dir),
        }

        state.apply_prefill(prefill)

        self.assertIsNotNone(state.loaded_dataset)
        self.assertEqual(state.target_format, MLXFormat.CHAT)
        self.assertEqual(state.train_pct, 85.0)
        self.assertEqual(state.valid_pct, 15.0)
        self.assertEqual(state.test_pct, 0.0)
        self.assertEqual(state.seed, 999)
        self.assertEqual(state.output_dir, str(self.out_dir))
        self.assertTrue(state.has_custom_output_dir)
        self.assertEqual(state.active_panel, ActivePanel.RIGHT)

    def test_custom_mapping_preserved_on_dataset_load(self):
        state = CommanderState()
        custom = ColumnMapping(prompt_col="id", completion_col="system_prompt")
        success = state.load_dataset(str(self.src_file), custom_mapping=custom)

        self.assertTrue(success)
        self.assertEqual(state.mapping.prompt_col, "id")
        self.assertEqual(state.mapping.completion_col, "system_prompt")


if __name__ == "__main__":
    unittest.main()

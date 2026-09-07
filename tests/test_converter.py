"""
Unit tests for conversion engine and JSONL output validation.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from mlx_commander.converter import convert_and_save
from mlx_commander.formats import ColumnMapping, MLXFormat
from mlx_commander.loader import load_local_dataset
from mlx_commander.splitter import SplitConfig
from tests.conftest import make_sample_qa_records


class TestConverter(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.src_file = Path(self.temp_dir) / "source.jsonl"
        self.out_dir = Path(self.temp_dir) / "mlx_output"

        # Prepare 100 sample records
        records = make_sample_qa_records(100)
        with open(self.src_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        self.dataset = load_local_dataset(str(self.src_file))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_convert_prompt_completion_with_test_split(self):
        mapping = ColumnMapping(prompt_col="instruction", completion_col="output")
        split_cfg = SplitConfig(train_pct=80, valid_pct=10, test_pct=10, seed=42)

        res = convert_and_save(
            dataset=self.dataset,
            format_type=MLXFormat.PROMPT_COMPLETION,
            mapping=mapping,
            output_dir_str=str(self.out_dir),
            split_config=split_cfg,
        )

        self.assertIn("train", res.output_files)
        self.assertIn("valid", res.output_files)
        self.assertIn("test", res.output_files)

        # Check file contents are valid JSONL
        for split_name, fpath in res.output_files.items():
            self.assertTrue(fpath.exists())
            with open(fpath, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f]
                self.assertGreater(len(lines), 0)
                for item in lines:
                    self.assertIn("prompt", item)
                    self.assertIn("completion", item)

        self.assertEqual(res.record_counts["train"], 80)
        self.assertEqual(res.record_counts["valid"], 10)
        self.assertEqual(res.record_counts["test"], 10)

        # Verify command string
        cmd = res.generate_mlx_lora_command("mlx-community/Mistral-7B-Instruct-v0.3-4bit")
        self.assertIn("mlx_lm.lora", cmd)
        self.assertIn(str(res.output_dir), cmd)
        self.assertIn("--mask-prompt", cmd)

    def test_convert_chat_format_without_test_split(self):
        mapping = ColumnMapping(
            system_col="system_prompt",
            user_col="instruction",
            assistant_col="output",
        )
        split_cfg = SplitConfig(train_pct=90, valid_pct=10, test_pct=0, seed=123)

        res = convert_and_save(
            dataset=self.dataset,
            format_type=MLXFormat.CHAT,
            mapping=mapping,
            output_dir_str=str(self.out_dir),
            split_config=split_cfg,
        )

        self.assertIn("train", res.output_files)
        self.assertIn("valid", res.output_files)
        self.assertNotIn("test", res.output_files)

        # Verify chat structure
        train_file = res.output_files["train"]
        with open(train_file, "r") as f:
            first_line = json.loads(f.readline())
            self.assertIn("messages", first_line)
            msgs = first_line["messages"]
            self.assertEqual(len(msgs), 3)
            self.assertEqual(msgs[0]["role"], "system")
            self.assertEqual(msgs[1]["role"], "user")
            self.assertEqual(msgs[2]["role"], "assistant")


    def test_convert_with_output_dir_keyword_arg(self):
        # Explicit test for the output_dir kwarg used by TUI
        mapping = ColumnMapping(prompt_col="instruction", completion_col="output")
        split_cfg = SplitConfig(train_pct=80, valid_pct=10, test_pct=10, seed=42)
        out_path = Path(self.temp_dir) / "output_dir_kwarg"

        res = convert_and_save(
            dataset=self.dataset,
            format_type=MLXFormat.PROMPT_COMPLETION,
            mapping=mapping,
            output_dir=out_path,
            split_config=split_cfg,
        )
        self.assertTrue(out_path.exists())
        self.assertIn("train", res.output_files)


if __name__ == "__main__":
    unittest.main()

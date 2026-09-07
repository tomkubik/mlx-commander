"""
Unit tests for ConversionResult manifest generation (mlx_manifest.json).
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


class TestManifest(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.src_file = Path(self.temp_dir) / "source.jsonl"
        self.out_dir = Path(self.temp_dir) / "mlx_output"

        records = make_sample_qa_records(50)
        with open(self.src_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        self.dataset = load_local_dataset(str(self.src_file))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_manifest_auto_created(self):
        mapping = ColumnMapping(prompt_col="instruction", completion_col="output")
        split_cfg = SplitConfig(train_pct=80, valid_pct=10, test_pct=10, seed=42)

        res = convert_and_save(
            dataset=self.dataset,
            format_type=MLXFormat.PROMPT_COMPLETION,
            mapping=mapping,
            output_dir=self.out_dir,
            split_config=split_cfg,
        )

        manifest_file = (self.out_dir / "mlx_manifest.json").resolve()
        self.assertTrue(manifest_file.exists())
        self.assertEqual(res.manifest_path, manifest_file)

        with open(manifest_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["status"], "success")
        self.assertEqual(data["format"], "prompt_completion")
        self.assertEqual(data["output_dir"], str(self.out_dir.resolve()))
        self.assertEqual(data["total_records"], 50)
        self.assertEqual(data["splits"]["train"], 40)
        self.assertEqual(data["splits"]["valid"], 5)
        self.assertEqual(data["splits"]["test"], 5)
        self.assertIn("train", data["files"])
        self.assertEqual(data["files"]["train"]["filename"], "train.jsonl")
        self.assertIn("mlx_lora_command", data)
        self.assertIn("mlx_lm.lora", data["mlx_lora_command"])

    def test_custom_manifest_path(self):
        custom_manifest = Path(self.temp_dir) / "custom_receipt.json"
        mapping = ColumnMapping(prompt_col="instruction", completion_col="output")

        res = convert_and_save(
            dataset=self.dataset,
            format_type=MLXFormat.PROMPT_COMPLETION,
            mapping=mapping,
            output_dir=self.out_dir,
            manifest_file=custom_manifest,
        )

        self.assertTrue(custom_manifest.exists())
        self.assertEqual(res.manifest_path, custom_manifest.resolve())


if __name__ == "__main__":
    unittest.main()

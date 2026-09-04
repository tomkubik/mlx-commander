"""
Unit tests for loading datasets from disk (JSONL, JSON, CSV, Mock HF dirs).
"""

import csv
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from hf2mlx.loader import inspect_dataset_path, load_local_dataset
from tests.conftest import make_sample_qa_records


class TestLoader(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_jsonl_file(self):
        file_path = Path(self.temp_dir) / "data.jsonl"
        records = make_sample_qa_records(25)
        with open(file_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        ds = load_local_dataset(str(file_path))
        self.assertEqual(ds.total_rows, 25)
        self.assertIn("instruction", ds.columns)
        self.assertIn("output", ds.columns)
        all_rows = ds.get_all_records()
        self.assertEqual(len(all_rows), 25)

    def test_load_json_array_file(self):
        file_path = Path(self.temp_dir) / "data.json"
        records = make_sample_qa_records(15)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(records, f)

        ds = load_local_dataset(str(file_path))
        self.assertEqual(ds.total_rows, 15)
        self.assertEqual(len(ds.get_all_records()), 15)

    def test_load_csv_file(self):
        file_path = Path(self.temp_dir) / "data.csv"
        records = make_sample_qa_records(10)
        with open(file_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            for r in records:
                writer.writerow(r)

        ds = load_local_dataset(str(file_path))
        self.assertEqual(ds.total_rows, 10)
        self.assertEqual(len(ds.get_all_records()), 10)

    def test_load_directory_with_split_files(self):
        dir_path = Path(self.temp_dir) / "my_dataset"
        dir_path.mkdir()
        train_records = make_sample_qa_records(20)
        valid_records = make_sample_qa_records(5)

        with open(dir_path / "train.jsonl", "w") as f:
            for r in train_records:
                f.write(json.dumps(r) + "\n")

        with open(dir_path / "valid.jsonl", "w") as f:
            for r in valid_records:
                f.write(json.dumps(r) + "\n")

        ds = load_local_dataset(str(dir_path))
        self.assertTrue(ds.is_split)
        self.assertEqual(ds.total_rows, 25)
        self.assertEqual(ds.split_counts.get("train"), 20)
        self.assertEqual(ds.split_counts.get("valid"), 5)

    def test_inspect_nonexistent_path(self):
        is_val, msg = inspect_dataset_path(str(Path(self.temp_dir) / "ghost"))
        self.assertFalse(is_val)


if __name__ == "__main__":
    unittest.main()

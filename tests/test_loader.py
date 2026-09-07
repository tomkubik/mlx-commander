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

from mlx_commander.loader import inspect_dataset_path, load_local_dataset
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

    def test_multi_file_dataset_loading_and_merging(self):
        file1 = Path(self.temp_dir) / "train.jsonl"
        file2 = Path(self.temp_dir) / "test.jsonl"
        rec1 = make_sample_qa_records(30)
        rec2 = make_sample_qa_records(20)

        with open(file1, "w", encoding="utf-8") as f:
            for r in rec1:
                f.write(json.dumps(r) + "\n")
        with open(file2, "w", encoding="utf-8") as f:
            for r in rec2:
                f.write(json.dumps(r) + "\n")

        # Test passing as list
        ds = load_local_dataset([str(file1), str(file2)])
        self.assertEqual(ds.total_rows, 50)
        self.assertFalse(ds.is_split)
        self.assertIn("Merged", ds.source_path)
        self.assertEqual(len(ds.get_all_records()), 50)
        self.assertEqual(ds.columns, list(rec1[0].keys()))

        # Test passing as newline-separated string
        ds_str = load_local_dataset(f"{file1}\n{file2}")
        self.assertEqual(ds_str.total_rows, 50)

        # Verify re-splitting from merged pool
        from mlx_commander.splitter import SplitConfig, split_records
        split_cfg = SplitConfig(train_pct=80, valid_pct=10, test_pct=10, seed=123)
        res = split_records(ds.get_all_records(), split_cfg)
        self.assertEqual(len(res["train"]), 40)
        self.assertEqual(len(res["valid"]), 5)
        self.assertEqual(len(res["test"]), 5)

    def test_schema_mismatch_detection(self):
        file1 = Path(self.temp_dir) / "part1.jsonl"
        file2 = Path(self.temp_dir) / "part2.jsonl"

        with open(file1, "w") as f:
            f.write(json.dumps({"prompt": "hi", "completion": "hello"}) + "\n")
        with open(file2, "w") as f:
            f.write(json.dumps({"question": "hi", "answer": "hello"}) + "\n")

        with self.assertRaises(ValueError) as ctx:
            load_local_dataset([str(file1), str(file2)])

        err_msg = str(ctx.exception)
        self.assertIn("Schema mismatch", err_msg)
        self.assertIn("part1.jsonl", err_msg)
        self.assertIn("part2.jsonl", err_msg)

    def test_inspect_multiple_paths(self):
        file1 = Path(self.temp_dir) / "file1.jsonl"
        file2 = Path(self.temp_dir) / "file2.jsonl"
        file1.write_text("{}\n")
        file2.write_text("{}\n")

        ok, msg = inspect_dataset_path(f"{file1}\n{file2}")
        self.assertTrue(ok)
        self.assertIn("Found 2 files to merge", msg)

        bad, bad_msg = inspect_dataset_path(f"{file1}\n/nonexistent/path.jsonl")
        self.assertFalse(bad)
        self.assertIn("Missing file", bad_msg)


if __name__ == "__main__":
    unittest.main()

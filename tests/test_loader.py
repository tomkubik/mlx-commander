"""
Unit tests for loading datasets from disk (JSONL, JSON, CSV, Mock HF dirs).
"""

import csv
import io
import json
import os
import shutil
import sqlite3
import tarfile
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

    # ----------------------------------------------------
    # TSV / Tab-Separated Tests
    # ----------------------------------------------------

    def test_load_tsv_file(self):
        file_path = Path(self.temp_dir) / "data.tsv"
        records = make_sample_qa_records(12)
        with open(file_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()), delimiter="\t")
            writer.writeheader()
            for r in records:
                writer.writerow(r)

        ds = load_local_dataset(str(file_path))
        self.assertEqual(ds.total_rows, 12)
        self.assertIn("instruction", ds.columns)
        self.assertIn("output", ds.columns)
        all_rows = ds.get_all_records()
        self.assertEqual(len(all_rows), 12)
        self.assertEqual(all_rows[0]["instruction"], records[0]["instruction"])

    def test_load_tab_extension(self):
        file_path = Path(self.temp_dir) / "data.tab"
        records = make_sample_qa_records(6)
        with open(file_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()), delimiter="\t")
            writer.writeheader()
            for r in records:
                writer.writerow(r)

        ds = load_local_dataset(str(file_path))
        self.assertEqual(ds.total_rows, 6)
        self.assertEqual(len(ds.get_all_records()), 6)

    def test_load_tsv_directory_splits(self):
        dir_path = Path(self.temp_dir) / "tsv_dataset"
        dir_path.mkdir()
        train_records = make_sample_qa_records(15)
        test_records = make_sample_qa_records(5)

        with open(dir_path / "train.tsv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(train_records[0].keys()), delimiter="\t")
            writer.writeheader()
            for r in train_records:
                writer.writerow(r)

        with open(dir_path / "test.tsv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(test_records[0].keys()), delimiter="\t")
            writer.writeheader()
            for r in test_records:
                writer.writerow(r)

        ds = load_local_dataset(str(dir_path))
        self.assertTrue(ds.is_split)
        self.assertEqual(ds.total_rows, 20)
        self.assertEqual(ds.split_counts.get("train"), 15)
        self.assertEqual(ds.split_counts.get("test"), 5)

    # ----------------------------------------------------
    # SQLite Database Tests
    # ----------------------------------------------------

    def test_load_sqlite_single_table(self):
        db_path = Path(self.temp_dir) / "test.sqlite"
        records = make_sample_qa_records(10)

        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("CREATE TABLE records (id INTEGER, instruction TEXT, input TEXT, output TEXT, system_prompt TEXT);")
        for r in records:
            cur.execute(
                "INSERT INTO records VALUES (?, ?, ?, ?, ?)",
                (r["id"], r["instruction"], r["input"], r["output"], r["system_prompt"]),
            )
        conn.commit()
        conn.close()

        ds = load_local_dataset(str(db_path))
        self.assertEqual(ds.total_rows, 10)
        self.assertFalse(ds.is_split)
        self.assertIn("instruction", ds.columns)
        self.assertEqual(len(ds.get_all_records()), 10)

    def test_load_sqlite_splits(self):
        db_path = Path(self.temp_dir) / "split_db.sqlite3"
        train_records = make_sample_qa_records(14)
        test_records = make_sample_qa_records(6)

        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("CREATE TABLE train (id INTEGER, instruction TEXT, input TEXT, output TEXT, system_prompt TEXT);")
        cur.execute("CREATE TABLE test (id INTEGER, instruction TEXT, input TEXT, output TEXT, system_prompt TEXT);")
        for r in train_records:
            cur.execute("INSERT INTO train VALUES (?, ?, ?, ?, ?)", tuple(r.values()))
        for r in test_records:
            cur.execute("INSERT INTO test VALUES (?, ?, ?, ?, ?)", tuple(r.values()))
        conn.commit()
        conn.close()

        ds = load_local_dataset(str(db_path))
        self.assertTrue(ds.is_split)
        self.assertEqual(ds.total_rows, 20)
        self.assertEqual(ds.split_counts.get("train"), 14)
        self.assertEqual(ds.split_counts.get("test"), 6)

    def test_load_sqlite_table_specifier(self):
        db_path = Path(self.temp_dir) / "multi_table.db"
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("CREATE TABLE users (id INTEGER, username TEXT);")
        cur.execute("CREATE TABLE prompts (id INTEGER, prompt TEXT, completion TEXT);")
        cur.execute("INSERT INTO users VALUES (1, 'alice');")
        cur.execute("INSERT INTO prompts VALUES (1, 'What is 2+2?', '4');")
        cur.execute("INSERT INTO prompts VALUES (2, 'What is 3+3?', '6');")
        conn.commit()
        conn.close()

        target_spec = f"{db_path}::prompts"
        ds = load_local_dataset(target_spec)
        self.assertEqual(ds.total_rows, 2)
        self.assertIn("prompt", ds.columns)
        self.assertIn("completion", ds.columns)
        self.assertEqual(ds.get_all_records()[0]["completion"], "4")

    def test_load_sqlite_invalid_table(self):
        db_path = Path(self.temp_dir) / "simple.db"
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("CREATE TABLE items (id INTEGER);")
        conn.commit()
        conn.close()

        with self.assertRaises(ValueError):
            load_local_dataset(f"{db_path}::nonexistent_table")

    # ----------------------------------------------------
    # WebDataset Tests
    # ----------------------------------------------------

    def test_load_webdataset_paired_samples(self):
        tar_path = Path(self.temp_dir) / "dataset.tar"
        with tarfile.open(tar_path, "w") as tar:
            for i in range(5):
                key = f"{i:05d}"
                json_data = json.dumps({"instruction": f"Instruction #{i}", "id": i}).encode("utf-8")
                txt_data = f"Response #{i}".encode("utf-8")

                ti_json = tarfile.TarInfo(f"{key}.json")
                ti_json.size = len(json_data)
                tar.addfile(ti_json, io.BytesIO(json_data))

                ti_txt = tarfile.TarInfo(f"{key}.txt")
                ti_txt.size = len(txt_data)
                tar.addfile(ti_txt, io.BytesIO(txt_data))

        ds = load_local_dataset(str(tar_path))
        self.assertEqual(ds.total_rows, 5)
        self.assertIn("instruction", ds.columns)
        self.assertIn("text", ds.columns)
        self.assertIn("__key__", ds.columns)
        rows = ds.get_all_records()
        self.assertEqual(rows[0]["text"], "Response #0")
        self.assertEqual(rows[0]["instruction"], "Instruction #0")

    def test_load_webdataset_prompt_completion(self):
        tar_path = Path(self.temp_dir) / "qa_webdataset.tar"
        with tarfile.open(tar_path, "w") as tar:
            for i in range(4):
                key = f"sample_{i}"
                prompt_bytes = f"Question {i}?".encode("utf-8")
                completion_bytes = f"Answer {i}.".encode("utf-8")

                ti_p = tarfile.TarInfo(f"{key}.prompt.txt")
                ti_p.size = len(prompt_bytes)
                tar.addfile(ti_p, io.BytesIO(prompt_bytes))

                ti_c = tarfile.TarInfo(f"{key}.completion.txt")
                ti_c.size = len(completion_bytes)
                tar.addfile(ti_c, io.BytesIO(completion_bytes))

        ds = load_local_dataset(str(tar_path))
        self.assertEqual(ds.total_rows, 4)
        self.assertIn("prompt", ds.columns)
        self.assertIn("completion", ds.columns)
        self.assertEqual(ds.get_all_records()[0]["prompt"], "Question 0?")
        self.assertEqual(ds.get_all_records()[0]["completion"], "Answer 0.")

    def test_load_webdataset_tar_gz(self):
        targz_path = Path(self.temp_dir) / "compressed.tar.gz"
        with tarfile.open(targz_path, "w:gz") as tar:
            for i in range(3):
                key = f"c_{i}"
                data = json.dumps({"prompt": f"P{i}", "completion": f"C{i}"}).encode("utf-8")
                ti = tarfile.TarInfo(f"{key}.json")
                ti.size = len(data)
                tar.addfile(ti, io.BytesIO(data))

        ds = load_local_dataset(str(targz_path))
        self.assertEqual(ds.total_rows, 3)
        self.assertEqual(ds.get_all_records()[1]["prompt"], "P1")

    def test_load_webdataset_embedded_jsonl(self):
        tar_path = Path(self.temp_dir) / "archive_with_jsonl.tar"
        records = make_sample_qa_records(8)
        jsonl_bytes = "\n".join(json.dumps(r) for r in records).encode("utf-8")

        with tarfile.open(tar_path, "w") as tar:
            ti = tarfile.TarInfo("train.jsonl")
            ti.size = len(jsonl_bytes)
            tar.addfile(ti, io.BytesIO(jsonl_bytes))

        ds = load_local_dataset(str(tar_path))
        self.assertEqual(ds.total_rows, 8)
        self.assertEqual(len(ds.get_all_records()), 8)

    def test_merge_tsv_and_jsonl(self):
        file_tsv = Path(self.temp_dir) / "part1.tsv"
        file_jsonl = Path(self.temp_dir) / "part2.jsonl"
        rec1 = [{"prompt": "q1", "completion": "a1"}, {"prompt": "q2", "completion": "a2"}]
        rec2 = [{"prompt": "q3", "completion": "a3"}]

        with open(file_tsv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["prompt", "completion"], delimiter="\t")
            writer.writeheader()
            for r in rec1:
                writer.writerow(r)

        with open(file_jsonl, "w", encoding="utf-8") as f:
            for r in rec2:
                f.write(json.dumps(r) + "\n")

        ds = load_local_dataset([str(file_tsv), str(file_jsonl)])
        self.assertEqual(ds.total_rows, 3)
        self.assertEqual(len(ds.get_all_records()), 3)

    # ----------------------------------------------------
    # Default Output Directory Tests
    # ----------------------------------------------------

    def test_default_output_dir_single_file(self):
        file_path = Path(self.temp_dir) / "sub" / "data.jsonl"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"prompt": "p", "completion": "c"}) + "\n")

        ds = load_local_dataset(str(file_path))
        expected = (file_path.parent / "mlx_dataset").resolve()
        self.assertEqual(ds.default_output_dir, expected)

    def test_default_output_dir_directory(self):
        dir_path = Path(self.temp_dir) / "hf_dir"
        dir_path.mkdir(parents=True, exist_ok=True)
        # Create a parquet/arrow or jsonl inside
        with open(dir_path / "train.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"prompt": "p", "completion": "c"}) + "\n")

        ds = load_local_dataset(str(dir_path))
        expected = (dir_path / "mlx_dataset").resolve()
        self.assertEqual(ds.default_output_dir, expected)

    def test_default_output_dir_sqlite_table_specifier(self):
        db_path = Path(self.temp_dir) / "sub_sql" / "test.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("CREATE TABLE records (prompt TEXT, completion TEXT);")
        cur.execute("INSERT INTO records VALUES ('p', 'c');")
        conn.commit()
        conn.close()

        ds = load_local_dataset(f"{db_path}::records")
        expected = (db_path.parent / "mlx_dataset").resolve()
        self.assertEqual(ds.default_output_dir, expected)

    def test_default_output_dir_multi_file(self):
        sub = Path(self.temp_dir) / "multi_sub"
        sub.mkdir(parents=True, exist_ok=True)
        f1 = sub / "part1.jsonl"
        f2 = sub / "part2.jsonl"
        with open(f1, "w") as f:
            f.write(json.dumps({"prompt": "p1", "completion": "c1"}) + "\n")
        with open(f2, "w") as f:
            f.write(json.dumps({"prompt": "p2", "completion": "c2"}) + "\n")

        ds = load_local_dataset([str(f1), str(f2)])
        expected = (sub / "mlx_dataset").resolve()
        self.assertEqual(ds.default_output_dir, expected)


if __name__ == "__main__":
    unittest.main()


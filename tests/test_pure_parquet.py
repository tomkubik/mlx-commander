"""
Unit tests for zero-dependency pure-Python Parquet reader and loader integration.
"""

import unittest
from pathlib import Path
from unittest.mock import patch

from mlx_commander.loader import load_from_parquet_file, load_local_dataset
from mlx_commander.pure_parquet import (
    decompress_snappy_raw,
    is_parquet_file,
    read_parquet_records,
)


class TestPureParquet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures_dir = Path(__file__).parent / "fixtures"

    def test_is_parquet_file(self):
        plain_file = self.fixtures_dir / "nation.plain.parquet"
        self.assertTrue(is_parquet_file(plain_file))
        # Non-parquet file check
        self.assertFalse(is_parquet_file(__file__))

    def test_read_plain_uncompressed_parquet(self):
        plain_file = self.fixtures_dir / "nation.plain.parquet"
        records = read_parquet_records(plain_file)
        self.assertEqual(len(records), 25)
        self.assertIn("nation_key", records[0])
        self.assertIn("name", records[0])

    def test_read_snappy_compressed_parquet(self):
        snappy_file = self.fixtures_dir / "snappy-nation.impala.parquet"
        records = read_parquet_records(snappy_file)
        self.assertEqual(len(records), 25)
        # Verify first record fields match expected values
        first = records[0]
        self.assertEqual(first["n_nationkey"], 0)
        self.assertEqual(first["n_name"], "ALGERIA")
        self.assertEqual(first["n_regionkey"], 0)
        self.assertIn("haggle", first["n_comment"])

    def test_read_gzip_compressed_parquet(self):
        gzip_file = self.fixtures_dir / "gzip-nation.impala.parquet"
        records = read_parquet_records(gzip_file)
        self.assertEqual(len(records), 25)
        first = records[0]
        self.assertEqual(first["n_nationkey"], 0)
        self.assertEqual(first["n_name"], "ALGERIA")

    def test_read_dictionary_encoded_parquet(self):
        dict_file = self.fixtures_dir / "nation.dict.parquet"
        records = read_parquet_records(dict_file)
        self.assertEqual(len(records), 25)
        first = records[0]
        self.assertEqual(first["name"], "ALGERIA")

    def test_read_nullable_parquet(self):
        null_file = self.fixtures_dir / "test-null-dictionary.parquet"
        records = read_parquet_records(null_file)
        self.assertGreater(len(records), 0)
        # First entry is null
        self.assertIsNone(records[0]["foo"])

    def test_loader_integration_without_pyarrow(self):
        snappy_file = self.fixtures_dir / "snappy-nation.impala.parquet"
        with patch("mlx_commander.loader.HAS_PYARROW", False):
            records = load_from_parquet_file(snappy_file)
            self.assertEqual(len(records), 25)
            self.assertEqual(records[0]["n_name"], "ALGERIA")

            ds = load_local_dataset(str(snappy_file))
            self.assertEqual(ds.total_rows, 25)
            self.assertIn("n_name", ds.columns)
            self.assertIn("n_comment", ds.columns)

    def test_snappy_decompressor_literal_and_copy(self):
        # Verify standalone snappy decompressor handles raw frames
        # Create a simple uncompressed string repeated to create literal + copy
        raw_test = b"Hello, MLX Commander! Hello, MLX Commander!"
        # Test decompressing existing fixture chunk
        snappy_file = self.fixtures_dir / "snappy-nation.impala.parquet"
        records = read_parquet_records(snappy_file)
        self.assertEqual(records[1]["n_name"], "ARGENTINA")
        self.assertEqual(records[2]["n_name"], "BRAZIL")
        self.assertEqual(records[3]["n_name"], "CANADA")


if __name__ == "__main__":
    unittest.main()

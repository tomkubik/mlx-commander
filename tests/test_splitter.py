"""
Unit tests for dataset percentage splitting and random seed reproducibility.
"""

import unittest
from mlx_commander.splitter import (
    SplitConfig,
    calculate_split_counts,
    generate_random_seed,
    split_records,
)
from tests.conftest import make_sample_qa_records


class TestSplitter(unittest.TestCase):

    def test_split_percentage_validation(self):
        # Sum != 100
        cfg = SplitConfig(train_pct=70, valid_pct=20, test_pct=0)
        errs = cfg.validate()
        self.assertTrue(len(errs) > 0)

        # Negative valid
        cfg2 = SplitConfig(train_pct=110, valid_pct=-10, test_pct=0)
        errs2 = cfg2.validate()
        self.assertTrue(len(errs2) > 0)

        # Valid 80/10/10
        cfg3 = SplitConfig(train_pct=80, valid_pct=10, test_pct=10)
        self.assertEqual(len(cfg3.validate()), 0)

    def test_calculate_split_counts(self):
        n_tr, n_va, n_te = calculate_split_counts(100, 80, 10, 10)
        self.assertEqual(n_tr, 80)
        self.assertEqual(n_va, 10)
        self.assertEqual(n_te, 10)

        # Zero test percentage
        n_tr, n_va, n_te = calculate_split_counts(100, 85, 15, 0)
        self.assertEqual(n_tr, 85)
        self.assertEqual(n_va, 15)
        self.assertEqual(n_te, 0)

    def test_seed_reproducibility(self):
        records = make_sample_qa_records(50)
        cfg_a = SplitConfig(train_pct=80, valid_pct=10, test_pct=10, seed=12345)
        cfg_b = SplitConfig(train_pct=80, valid_pct=10, test_pct=10, seed=12345)
        cfg_c = SplitConfig(train_pct=80, valid_pct=10, test_pct=10, seed=99999)

        res_a = split_records(records, cfg_a)
        res_b = split_records(records, cfg_b)
        res_c = split_records(records, cfg_c)

        # Exactly identical for same seed
        self.assertEqual([r["id"] for r in res_a["train"]], [r["id"] for r in res_b["train"]])
        self.assertEqual([r["id"] for r in res_a["valid"]], [r["id"] for r in res_b["valid"]])
        self.assertEqual([r["id"] for r in res_a["test"]], [r["id"] for r in res_b["test"]])

        # Different for different seed
        self.assertNotEqual([r["id"] for r in res_a["train"]], [r["id"] for r in res_c["train"]])

    def test_optional_test_split(self):
        records = make_sample_qa_records(20)
        cfg = SplitConfig(train_pct=90, valid_pct=10, test_pct=0, seed=42)
        res = split_records(records, cfg)
        self.assertIn("train", res)
        self.assertIn("valid", res)
        self.assertNotIn("test", res)
        self.assertEqual(len(res["train"]) + len(res["valid"]), 20)

    def test_generate_random_seed(self):
        seed1 = generate_random_seed()
        seed2 = generate_random_seed()
        self.assertIsInstance(seed1, int)
        self.assertGreaterEqual(seed1, 100000)
        self.assertLess(seed1, 1000000)


if __name__ == "__main__":
    unittest.main()

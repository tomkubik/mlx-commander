"""
Unit tests for CLI parsing, arguments, and direct execution.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from hf2mlx.cli import build_parser, main, parse_mapping_arg
from tests.conftest import make_sample_qa_records


class TestCLI(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.src_file = Path(self.temp_dir) / "source.jsonl"
        self.out_dir = Path(self.temp_dir) / "mlx_cli_out"

        records = make_sample_qa_records(30)
        with open(self.src_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_parse_mapping_arg_key_val(self):
        mapping = parse_mapping_arg("prompt=instruction,completion=output")
        self.assertEqual(mapping.prompt_col, "instruction")
        self.assertEqual(mapping.completion_col, "output")

    def test_parse_mapping_arg_json(self):
        mapping = parse_mapping_arg('{"text_col": "my_text"}')
        self.assertEqual(mapping.text_col, "my_text")

    def test_cli_direct_conversion(self):
        argv = [
            "-d", str(self.src_file),
            "-f", "prompt_completion",
            "-o", str(self.out_dir),
            "--prompt-col", "instruction",
            "--completion-col", "output",
            "--train", "70",
            "--valid", "20",
            "--test", "10",
            "--seed", "123456",
        ]
        ret = main(argv)
        self.assertEqual(ret, 0)

        train_file = self.out_dir / "train.jsonl"
        valid_file = self.out_dir / "valid.jsonl"
        test_file = self.out_dir / "test.jsonl"

        self.assertTrue(train_file.exists())
        self.assertTrue(valid_file.exists())
        self.assertTrue(test_file.exists())

        with open(train_file) as f:
            lines = [json.loads(l) for l in f]
            self.assertEqual(len(lines), 21)  # 70% of 30 is 21

    def test_interactive_wizard(self):
        import io
        import sys
        from hf2mlx.tui.wizard_fallback import run_interactive_wizard

        inputs = [
            "3",    # Format: prompt_completion
            "1",    # Prompt col: instruction
            "2",    # Completion col: output
            "80",   # Train %
            "10",   # Valid %
            "10",   # Test %
            "999",  # Seed
            "Y",    # Confirm
        ]
        orig_stdin = sys.stdin
        try:
            sys.stdin = io.StringIO("\n".join(inputs) + "\n")
            res = run_interactive_wizard(dataset_path=str(self.src_file), output_dir_arg=str(self.out_dir))
            self.assertEqual(res.record_counts["train"], 24)
            self.assertEqual(res.record_counts["valid"], 3)
            self.assertEqual(res.record_counts["test"], 3)
        finally:
            sys.stdin = orig_stdin

    def test_entry_points(self):
        import subprocess
        root_dir = Path(__file__).resolve().parent.parent
        python_bin = sys.executable

        cmds = [
            [python_bin, "run.py", "--version"],
            [python_bin, ".", "--version"],
            [python_bin, "-m", "hf2mlx", "--version"],
            [python_bin, "./hf2mlx_cli", "--version"],
            ["bash", "./hf2mlx_cli", "--version"],
            ["./hf2mlx_cli", "--version"],
        ]
        for cmd in cmds:
            p = subprocess.run(cmd, cwd=str(root_dir), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertEqual(p.returncode, 0, f"Command failed: {cmd}, stderr: {p.stderr}")
            self.assertIn("hf2mlx", p.stdout)


if __name__ == "__main__":
    unittest.main()


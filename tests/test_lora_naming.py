import unittest
from mlx_commander.lora.config import (
    LoraRunConfig,
    format_learning_rate,
    generate_deterministic_run_name,
    sanitize_model_slug,
)
from mlx_commander.lora.queue import QueueManager
import tempfile
import shutil
from pathlib import Path


class TestLoraNaming(unittest.TestCase):
    def test_sanitize_model_slug(self):
        self.assertEqual(sanitize_model_slug("mlx-community/Llama-3.2-3B-Instruct-4bit"), "Llama-3.2-3B-Instruct-4bit")
        self.assertEqual(sanitize_model_slug("Qwen/Qwen2.5-7B-Instruct"), "Qwen2.5-7B-Instruct")
        self.assertEqual(sanitize_model_slug("models/my_custom/model"), "model")
        self.assertEqual(sanitize_model_slug("special @#$% chars!"), "special-chars")
        self.assertEqual(sanitize_model_slug(""), "model")

    def test_format_learning_rate(self):
        self.assertEqual(format_learning_rate(1e-5), "1e-5")
        self.assertEqual(format_learning_rate(2e-4), "0p0002")
        self.assertEqual(format_learning_rate(0.0002), "0p0002")
        self.assertEqual(format_learning_rate(0.00001), "1e-5")
        self.assertEqual(format_learning_rate(0.001), "0p001")
        self.assertEqual(format_learning_rate(1.5e-4), "0p00015")

    def test_generate_deterministic_run_name(self):
        # Format: {index:02d}_{method}_r{rank}_a{alpha}_lr{lr}_b{batch}_i{iters}_{model_slug}
        name1 = generate_deterministic_run_name(
            index=1,
            fine_tune_type="lora",
            rank=16,
            alpha=32.0,
            learning_rate=1e-5,
            batch_size=4,
            iters=1000,
            model_name="mlx-community/Llama-3.2-3B-Instruct-4bit",
        )
        self.assertEqual(name1, "01_lora_r16_a32_lr1e-5_b4_i1000_Llama-3.2-3B-Instruct-4bit")

        name2 = generate_deterministic_run_name(
            index=12,
            fine_tune_type="dora",
            rank=8,
            alpha=16.0,
            learning_rate=3e-5,
            batch_size=8,
            iters=500,
            model_name="mlx-community/Qwen2.5-7B-Instruct-4bit",
        )
        self.assertEqual(name2, "12_dora_r8_a16_lr3e-5_b8_i500_Qwen2.5-7B-Instruct-4bit")

    def test_lora_config_auto_naming(self):
        cfg = LoraRunConfig(
            model="mlx-community/Llama-3.2-3B-Instruct-4bit",
            fine_tune_type="lora",
            lora_rank=16,
            lora_alpha=32.0,
            learning_rate=1e-5,
            batch_size=4,
            iters=1000,
        )
        self.assertEqual(cfg.name, "01_lora_r16_a32_lr1e-5_b4_i1000_Llama-3.2-3B-Instruct-4bit")
        self.assertEqual(cfg.adapter_path, "adapters/01_lora_r16_a32_lr1e-5_b4_i1000_Llama-3.2-3B-Instruct-4bit")
        self.assertFalse(cfg.is_custom_name)

    def test_lora_config_custom_name_preserved(self):
        cfg = LoraRunConfig(
            name="my_handcrafted_lora_experiment",
            adapter_path="my_adapters/exp1",
        )
        self.assertTrue(cfg.is_custom_name)
        self.assertEqual(cfg.name, "my_handcrafted_lora_experiment")
        self.assertEqual(cfg.adapter_path, "my_adapters/exp1")

    def test_queue_sequential_deterministic_naming(self):
        temp_dir = tempfile.mkdtemp()
        try:
            q_mgr = QueueManager(Path(temp_dir) / "runs")

            # Add first run
            r1 = LoraRunConfig(
                model="mlx-community/Llama-3.2-3B-Instruct-4bit",
                learning_rate=1e-5,
            )
            q_mgr.add_run(r1)
            self.assertEqual(q_mgr.runs[0].sequence_index, 1)
            self.assertTrue(q_mgr.runs[0].name.startswith("01_lora_"))
            self.assertTrue(q_mgr.runs[0].adapter_path.startswith("adapters/01_lora_"))

            # Add second run with different learning rate
            r2 = LoraRunConfig(
                model="mlx-community/Llama-3.2-3B-Instruct-4bit",
                learning_rate=3e-5,
            )
            q_mgr.add_run(r2)
            self.assertEqual(q_mgr.runs[1].sequence_index, 2)
            self.assertTrue(q_mgr.runs[1].name.startswith("02_lora_"))
            self.assertIn("lr3e-5", q_mgr.runs[1].name)

            # Clone r1 -> creates run index 3
            cloned = q_mgr.clone_run(r1.id)
            self.assertIsNotNone(cloned)
            self.assertEqual(cloned.sequence_index, 3)
            self.assertTrue(cloned.name.startswith("03_lora_"))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_queue_preserves_custom_names(self):
        temp_dir = tempfile.mkdtemp()
        try:
            q_mgr = QueueManager(Path(temp_dir) / "runs")
            custom_run = LoraRunConfig(
                name="custom_experiment_alpha",
                adapter_path="adapters/custom_experiment_alpha",
                is_custom_name=True,
            )
            q_mgr.add_run(custom_run)
            self.assertEqual(q_mgr.runs[0].name, "custom_experiment_alpha")
            self.assertEqual(q_mgr.runs[0].adapter_path, "adapters/custom_experiment_alpha")

            # Clone custom run
            cloned = q_mgr.clone_run(custom_run.id)
            self.assertEqual(cloned.name, "Copy of custom_experiment_alpha")
            self.assertTrue(cloned.is_custom_name)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

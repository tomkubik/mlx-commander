import json
import os
import tempfile
import unittest
from pathlib import Path

from mlx_commander.lora.model_info import (
    ModelMetadata,
    _calculate_dir_weights_size_gb,
    _read_safetensors_header,
    inspect_local_model,
    scan_local_models,
)


class TestModelInfo(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def test_inspect_local_model_empty(self):
        meta = inspect_local_model("")
        self.assertFalse(meta.is_valid)
        self.assertEqual(meta.name, "(No Model Selected)")

    def test_inspect_local_model_nonexistent(self):
        meta = inspect_local_model("/nonexistent/path/to/model-7b")
        self.assertFalse(meta.is_valid)
        self.assertEqual(meta.name, "model-7b")
        self.assertEqual(meta.architecture, "Not Found")

    def test_inspect_local_model_with_config_json(self):
        model_dir = Path(self.temp_dir) / "Llama-3.2-3B-Instruct-4bit"
        model_dir.mkdir(parents=True, exist_ok=True)

        config_data = {
            "_name_or_path": "meta-llama/Llama-3.2-3B-Instruct",
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "num_hidden_layers": 28,
            "hidden_size": 3072,
            "num_attention_heads": 24,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
            "max_position_embeddings": 131072,
            "quantization": {
                "bits": 4,
                "group_size": 64,
            },
        }
        with open(model_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f)

        # Create dummy weights file
        dummy_weights = model_dir / "model.safetensors"
        with open(dummy_weights, "wb") as f:
            f.write(b"\x00" * 1024 * 1024)  # 1 MB

        # Test inspecting via directory
        meta = inspect_local_model(str(model_dir))
        self.assertTrue(meta.is_valid)
        self.assertEqual(meta.name, "Llama-3.2-3B-Instruct")
        self.assertEqual(meta.architecture, "llama")
        self.assertEqual(meta.num_layers, 28)
        self.assertEqual(meta.hidden_size, 3072)
        self.assertEqual(meta.num_heads, 24)
        self.assertEqual(meta.num_kv_heads, 8)
        self.assertEqual(meta.vocab_size, 128256)
        self.assertEqual(meta.context_length, 131072)
        self.assertEqual(meta.quantization, "4-bit (group 64)")
        self.assertGreater(meta.file_size_gb, 0.0)

        # Verify formatted line
        formatted = meta.format_hyperparameters_line()
        self.assertIn("arch=llama", formatted)
        self.assertIn("28 layers", formatted)
        self.assertIn("3072 dim", formatted)
        self.assertIn("24 heads (KV: 8)", formatted)
        self.assertIn("128k vocab", formatted)
        self.assertIn("4-bit (group 64)", formatted)

        # Test inspecting via config.json file directly
        meta_file = inspect_local_model(str(model_dir / "config.json"))
        self.assertTrue(meta_file.is_valid)
        self.assertEqual(meta_file.num_layers, 28)

        # Test inspecting via weights file directly
        meta_weights = inspect_local_model(str(dummy_weights))
        self.assertTrue(meta_weights.is_valid)
        self.assertEqual(meta_weights.num_layers, 28)

    def test_inspect_local_model_hf_cache_snapshots(self):
        repo_dir = Path(self.temp_dir) / "models--mlx-community--Qwen2.5-7B-Instruct-4bit"
        snap_dir = repo_dir / "snapshots" / "1234567890abcdef1234567890abcdef12345678"
        snap_dir.mkdir(parents=True, exist_ok=True)

        config_data = {
            "model_type": "qwen2",
            "num_hidden_layers": 28,
            "hidden_size": 3584,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "vocab_size": 152064,
            "max_position_embeddings": 32768,
            "torch_dtype": "bfloat16",
        }
        with open(snap_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f)

        # Pointing to root repo dir should discover snapshot config
        meta = inspect_local_model(str(repo_dir))
        self.assertTrue(meta.is_valid)
        self.assertEqual(meta.architecture, "qwen2")
        self.assertEqual(meta.num_layers, 28)
        self.assertEqual(meta.hidden_size, 3584)
        self.assertEqual(meta.num_heads, 28)
        self.assertEqual(meta.num_kv_heads, 4)

    def test_inspect_local_model_safetensors_fallback(self):
        model_dir = Path(self.temp_dir) / "custom-safetensors-model"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Synthesize a safetensors file with header
        header = {
            "model.layers.15.self_attn.q_proj.weight": {"dtype": "F16", "shape": [4096, 4096], "data_offsets": [0, 100]},
            "model.embed_tokens.weight": {"dtype": "F16", "shape": [32000, 4096], "data_offsets": [100, 200]},
        }
        header_bytes = json.dumps(header).encode("utf-8")
        header_len = len(header_bytes)

        st_path = model_dir / "model.safetensors"
        with open(st_path, "wb") as f:
            f.write(header_len.to_bytes(8, "little"))
            f.write(header_bytes)
            f.write(b"\x00" * 200)

        meta = inspect_local_model(str(model_dir))
        self.assertTrue(meta.is_valid)
        self.assertEqual(meta.architecture, "MLX / Safetensors")
        self.assertEqual(meta.num_layers, 16)
        self.assertEqual(meta.hidden_size, 4096)
        self.assertEqual(meta.vocab_size, 32000)

    def test_scan_local_models(self):
        models_root = Path(self.temp_dir) / "models"
        m1 = models_root / "m1"
        m1.mkdir(parents=True, exist_ok=True)
        with open(m1 / "config.json", "w") as f:
            json.dump({"model_type": "llama", "num_hidden_layers": 16}, f)

        discovered = scan_local_models(search_dirs=[str(models_root)])
        self.assertEqual(len(discovered), 1)
        self.assertEqual(discovered[0].architecture, "llama")
        self.assertEqual(discovered[0].num_layers, 16)

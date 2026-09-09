import json
import shutil
import tempfile
import unittest
from pathlib import Path

from mlx_commander.lora.config import LoraRunConfig
from mlx_commander.lora.queue import QueueManager


class TestLoraQueue(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.queue_dir = Path(self.temp_dir) / "test_mlx_runs"
        self.manager = QueueManager(self.queue_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_queue_initialization(self):
        self.assertTrue(self.queue_dir.exists())
        self.assertTrue((self.queue_dir / "configs").exists())
        self.assertEqual(len(self.manager.runs), 0)

    def test_add_run(self):
        run = LoraRunConfig(
            id="run_test_001",
            name="Run Test 1",
            model="mlx-community/Llama-3.2-3B-Instruct-4bit",
            iters=500,
        )
        self.manager.add_run(run)

        self.assertEqual(len(self.manager.runs), 1)
        self.assertEqual(self.manager.runs[0].id, "run_test_001")

        # Verify YAML config was written
        yaml_file = self.queue_dir / "configs" / "run_test_001.yaml"
        self.assertTrue(yaml_file.exists())

        # Verify queue.json was written
        queue_file = self.queue_dir / "queue.json"
        self.assertTrue(queue_file.exists())
        with open(queue_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(len(data["runs"]), 1)
        self.assertEqual(data["runs"][0]["id"], "run_test_001")

    def test_clone_run(self):
        run = LoraRunConfig(
            id="run_orig",
            name="Original Run",
            model="mlx-community/Qwen2.5-7B-Instruct-4bit",
            learning_rate=3e-5,
        )
        self.manager.add_run(run)

        cloned = self.manager.clone_run("run_orig")
        self.assertIsNotNone(cloned)
        self.assertNotEqual(cloned.id, "run_orig")
        self.assertIn("Copy of", cloned.name)
        self.assertEqual(len(self.manager.runs), 2)
        self.assertEqual(cloned.status, "queued")

    def test_delete_run(self):
        run1 = LoraRunConfig(id="r1", name="Run 1")
        run2 = LoraRunConfig(id="r2", name="Run 2")
        self.manager.add_run(run1)
        self.manager.add_run(run2)

        self.assertEqual(len(self.manager.runs), 2)
        success = self.manager.delete_run("r1")
        self.assertTrue(success)
        self.assertEqual(len(self.manager.runs), 1)
        self.assertEqual(self.manager.runs[0].id, "r2")
        self.assertFalse((self.queue_dir / "configs" / "r1.yaml").exists())

    def test_clear_queue(self):
        self.manager.add_run(LoraRunConfig(id="r1"))
        self.manager.add_run(LoraRunConfig(id="r2"))
        self.assertEqual(len(self.manager.runs), 2)

        self.manager.clear_queue()
        self.assertEqual(len(self.manager.runs), 0)
        self.assertEqual(len(list((self.queue_dir / "configs").glob("*.yaml"))), 0)

    def test_get_pending_runs(self):
        r1 = LoraRunConfig(id="r1", status="queued")
        r2 = LoraRunConfig(id="r2", status="completed")
        r3 = LoraRunConfig(id="r3", status="queued")
        self.manager.add_run(r1)
        self.manager.add_run(r2)
        self.manager.add_run(r3)

        pending = self.manager.get_pending_runs()
        self.assertEqual(len(pending), 2)
        self.assertEqual([r.id for r in pending], ["r1", "r3"])

    def test_update_run_status(self):
        r = LoraRunConfig(id="r1", status="queued")
        self.manager.add_run(r)

        self.manager.update_run_status("r1", "running")
        self.assertEqual(self.manager.runs[0].status, "running")
        self.assertIsNotNone(self.manager.runs[0].started_at)

        self.manager.update_run_status("r1", "completed", exit_code=0)
        self.assertEqual(self.manager.runs[0].status, "completed")
        self.assertEqual(self.manager.runs[0].exit_code, 0)
        self.assertIsNotNone(self.manager.runs[0].finished_at)

    def test_generate_queue_script(self):
        r1 = LoraRunConfig(id="r1", name="Run One", iters=100)
        self.manager.add_run(r1)

        script_path = self.manager.generate_queue_script()
        self.assertTrue(script_path.exists())
        content = script_path.read_text(encoding="utf-8")
        self.assertIn("#!/usr/bin/env bash", content)
        self.assertIn("mlx_lm.lora --config", content)
        self.assertIn("r1.yaml", content)

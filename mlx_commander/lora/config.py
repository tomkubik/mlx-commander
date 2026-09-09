"""
LoRA Fine-Tuning Configuration Model for Apple MLX (mlx-lm).
Defines explicit training arguments, validation, YAML serialization,
and CLI command generation for mlx_lm.lora.
"""

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

POPULAR_MLX_MODELS = [
    ("mlx-community/Llama-3.2-3B-Instruct-4bit", "Llama-3.2-3B-Instruct (4-bit, fast & light)"),
    ("mlx-community/Llama-3.2-1B-Instruct-4bit", "Llama-3.2-1B-Instruct (4-bit, ultra-light)"),
    ("mlx-community/Qwen2.5-7B-Instruct-4bit", "Qwen2.5-7B-Instruct (4-bit, reasoning/code)"),
    ("mlx-community/Qwen2.5-3B-Instruct-4bit", "Qwen2.5-3B-Instruct (4-bit, balanced)"),
    ("mlx-community/Mistral-7B-Instruct-v0.3-4bit", "Mistral-7B-Instruct-v0.3 (4-bit)"),
    ("mlx-community/Phi-3.5-mini-instruct-4bit", "Phi-3.5-mini-instruct (4-bit, 3.8B)"),
]

FINE_TUNE_TYPES = ["lora", "dora", "full"]
OPTIMIZERS = ["adamw", "adam", "muon", "sgd", "adafactor"]


@dataclass
class LoraRunConfig:
    id: str = field(default_factory=lambda: f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}")
    name: str = ""
    model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    data: str = "mlx_dataset"
    train: bool = True
    test: bool = False
    fine_tune_type: str = "lora"
    optimizer: str = "adamw"
    iters: int = 1000
    batch_size: int = 4
    learning_rate: float = 1e-5
    num_layers: int = 16
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    max_seq_length: int = 2048
    grad_checkpoint: bool = True
    grad_accumulation_steps: int = 1
    mask_prompt: bool = True
    steps_per_report: int = 10
    steps_per_eval: int = 200
    val_batches: int = 25
    save_every: int = 100
    adapter_path: str = "adapters"
    seed: int = 0
    resume_adapter_file: Optional[str] = None

    # Queue execution tracking
    status: str = "queued"  # queued, running, completed, failed, cancelled
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    log_file: Optional[str] = None
    exit_code: Optional[int] = None
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name:
            model_slug = self.model.split("/")[-1].replace("-Instruct", "").replace("-4bit", "")
            self.name = f"{model_slug} (lr={self.learning_rate:g}, r={self.lora_rank})"
        if not self.adapter_path or self.adapter_path == "adapters":
            self.adapter_path = f"adapters/{self.id}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoraRunConfig":
        clean_data = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**clean_data)

    def to_mlx_yaml(self) -> str:
        """Generate YAML configuration compliant with mlx_lm.lora --config schema."""
        lines = [
            f'model: "{self.model}"',
            f'train: {"true" if self.train else "false"}',
            f'data: "{self.data}"',
            f'fine_tune_type: "{self.fine_tune_type}"',
            f'optimizer: "{self.optimizer}"',
            f'batch_size: {self.batch_size}',
            f'iters: {self.iters}',
            f'val_batches: {self.val_batches}',
            f'learning_rate: {self.learning_rate:g}',
            f'steps_per_report: {self.steps_per_report}',
            f'steps_per_eval: {self.steps_per_eval}',
            f'save_every: {self.save_every}',
            f'adapter_path: "{self.adapter_path}"',
            f'max_seq_length: {self.max_seq_length}',
            f'grad_checkpoint: {"true" if self.grad_checkpoint else "false"}',
            f'grad_accumulation_steps: {self.grad_accumulation_steps}',
            f'mask_prompt: {"true" if self.mask_prompt else "false"}',
            f'seed: {self.seed}',
            f'num_layers: {self.num_layers}',
            'lora_parameters:',
            f'  rank: {self.lora_rank}',
            f'  dropout: {self.lora_dropout}',
            f'  scale: {self.lora_alpha}',
        ]
        if self.test:
            lines.append('test: true')
        return "\n".join(lines) + "\n"

    def to_cli_command(self, config_file: Optional[str] = None) -> str:
        """Generate the equivalent mlx_lm.lora command invocation."""
        if config_file:
            return f"mlx_lm.lora --config {config_file}"
        cmd = [
            "mlx_lm.lora",
            f"--model {self.model}",
            f"--data {self.data}",
        ]
        if self.train:
            cmd.append("--train")
        if self.test:
            cmd.append("--test")
        cmd.extend([
            f"--fine-tune-type {self.fine_tune_type}",
            f"--optimizer {self.optimizer}",
            f"--batch-size {self.batch_size}",
            f"--iters {self.iters}",
            f"--learning-rate {self.learning_rate:g}",
            f"--num-layers {self.num_layers}",
            f"--max-seq-length {self.max_seq_length}",
            f"--adapter-path {self.adapter_path}",
            f"--save-every {self.save_every}",
            f"--steps-per-eval {self.steps_per_eval}",
            f"--steps-per-report {self.steps_per_report}",
        ])
        if self.grad_checkpoint:
            cmd.append("--grad-checkpoint")
        if self.mask_prompt:
            cmd.append("--mask-prompt")
        if self.resume_adapter_file:
            cmd.append(f"--resume-adapter-file {self.resume_adapter_file}")
        return " ".join(cmd)

    def validate(self) -> List[str]:
        """Validate config parameters and return list of human-readable errors."""
        errors: List[str] = []
        if self.iters <= 0:
            errors.append("Iterations must be > 0.")
        if self.batch_size <= 0:
            errors.append("Batch size must be > 0.")
        if self.learning_rate <= 0:
            errors.append("Learning rate must be > 0.")
        if self.lora_rank <= 0:
            errors.append("LoRA rank must be > 0.")
        if self.num_layers <= 0:
            errors.append("Number of layers must be > 0.")
        if not (0.0 <= self.lora_dropout <= 0.5):
            errors.append("Dropout must be between 0.0 and 0.5.")
        if self.max_seq_length < 64:
            errors.append("Max sequence length must be at least 64.")
        return errors

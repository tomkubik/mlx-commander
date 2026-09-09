"""
MLX Commander LoRA Fine-Tuning & Queue Orchestration Module.
"""

from .config import (
    FINE_TUNE_TYPES,
    OPTIMIZERS,
    POPULAR_MLX_MODELS,
    LoraRunConfig,
)
from .estimator import (
    calculate_implied_epochs,
    estimate_duration,
    estimate_peak_memory,
    get_apple_silicon_chip,
    get_hardware_memory_bytes,
)
from .queue import QueueManager
from .runner import run_lora_queue

__all__ = [
    "LoraRunConfig",
    "QueueManager",
    "run_lora_queue",
    "calculate_implied_epochs",
    "estimate_peak_memory",
    "estimate_duration",
    "get_hardware_memory_bytes",
    "get_apple_silicon_chip",
    "POPULAR_MLX_MODELS",
    "FINE_TUNE_TYPES",
    "OPTIMIZERS",
]

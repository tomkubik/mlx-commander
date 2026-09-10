"""
MLX Commander LoRA Fine-Tuning & Queue Orchestration Module.
"""

from .config import (
    FINE_TUNE_TYPES,
    OPTIMIZERS,
    POPULAR_MLX_MODELS,
    LoraRunConfig,
    format_learning_rate,
    generate_deterministic_run_name,
    sanitize_model_slug,
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
from .tracking import (
    WandbTracker,
    is_wandb_available,
    is_wandb_logged_in,
    parse_mlx_log_line,
)

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
    "generate_deterministic_run_name",
    "sanitize_model_slug",
    "format_learning_rate",
    "WandbTracker",
    "is_wandb_available",
    "is_wandb_logged_in",
    "parse_mlx_log_line",
]

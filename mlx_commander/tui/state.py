"""
Central Application State for MLX-Commander persistent TUI.
Maintains dataset info, format configuration, column mappings,
active panel focus, and reactive preview cache.
"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from mlx_commander.converter import ConversionResult, convert_and_save
from mlx_commander.formats import (
    ColumnMapping,
    MLXFormat,
    auto_detect_mapping,
    format_record,
    validate_mapping,
)
from mlx_commander.loader import LoadedDataset, load_local_dataset
from mlx_commander.splitter import (
    SplitConfig,
    calculate_split_counts,
    generate_random_seed,
)


class ActivePanel(Enum):
    LEFT = "left"    # Dataset source, schema, and columns
    RIGHT = "right"  # MLX format, column mappings, splits, output
    PREVIEW = "preview"


@dataclass
class CommanderState:
    dataset_path: str = ""
    loaded_dataset: Optional[LoadedDataset] = None
    target_format: MLXFormat = MLXFormat.PROMPT_COMPLETION
    mapping: ColumnMapping = field(default_factory=ColumnMapping)
    train_pct: float = 80.0
    valid_pct: float = 10.0
    test_pct: float = 10.0
    seed: int = field(default_factory=generate_random_seed)
    output_dir: str = "./mlx_dataset"

    # UI navigation state
    active_panel: ActivePanel = ActivePanel.RIGHT
    left_focus_idx: int = 0   # 0: Browse button, 1: Edit path button, 2: Columns list
    right_focus_idx: int = 0  # 0: Format, 1-3: Mappings, 4-6: Splits, 7: Seed, 8: Output, 9: Convert button
    column_scroll_offset: int = 0
    selected_column_idx: int = 0

    # Status / notification
    status_message: str = "Ready. Press Tab to switch panels, ↑/↓ to navigate, F5 to convert."
    status_is_error: bool = False

    # Reactive preview cache
    preview_cache: List[str] = field(default_factory=list)
    preview_error: Optional[str] = None

    def load_dataset(self, path_input: Any) -> bool:
        """Load dataset from disk (single file/folder or multiple merged files) and update state."""
        if not path_input:
            self.status_message = "Path cannot be empty."
            self.status_is_error = True
            return False

        try:
            self.status_message = "Loading dataset..."
            self.status_is_error = False
            ds = load_local_dataset(path_input)
            self.loaded_dataset = ds
            self.dataset_path = ds.source_path
            self.selected_column_idx = 0
            self.column_scroll_offset = 0

            # Auto-detect column mapping for current format
            self.mapping = auto_detect_mapping(self.target_format, ds.columns)
            if "Merged (" in ds.source_path:
                self.status_message = f"[OK] {ds.source_path}: {ds.total_rows:,} records merged. Schemas verified."
            else:
                self.status_message = f"Loaded {ds.total_rows:,} records with {len(ds.columns)} columns."
            self.status_is_error = False
            self.update_preview()
            return True
        except Exception as e:
            self.status_message = f"Failed to load: {e}"
            self.status_is_error = True
            return False

    def set_format(self, fmt: MLXFormat) -> None:
        """Change MLX target format and re-detect mappings if appropriate."""
        self.target_format = fmt
        if self.loaded_dataset and self.loaded_dataset.columns:
            self.mapping = auto_detect_mapping(fmt, self.loaded_dataset.columns)
        self.update_preview()

    def update_preview(self) -> None:
        """Generate live preview records based on current mapping and format."""
        self.preview_cache.clear()
        self.preview_error = None

        if not self.loaded_dataset or not self.loaded_dataset.sample_records:
            self.preview_cache = ["(No dataset loaded)"]
            return

        samples = self.loaded_dataset.sample_records[:3]
        errors = validate_mapping(self.target_format, self.mapping, self.loaded_dataset.columns)
        if errors:
            self.preview_error = f"Mapping incomplete: {', '.join(errors)}"
            return

        formatted = []
        for s in samples:
            try:
                rec = format_record(s, self.target_format, self.mapping)
                formatted.append(json.dumps(rec, ensure_ascii=False))
            except Exception as e:
                self.preview_error = f"Format error: {e}"
                return

        self.preview_cache = formatted

    def randomize_seed(self) -> None:
        """Generate a new random seed."""
        self.seed = generate_random_seed()
        self.status_message = f"Random seed generated: {self.seed}"
        self.status_is_error = False

    def get_split_counts(self) -> Dict[str, int]:
        """Calculate record counts for train / valid / test splits."""
        if not self.loaded_dataset:
            return {"train": 0, "valid": 0, "test": 0}
        n_train, n_valid, n_test = calculate_split_counts(
            self.loaded_dataset.total_rows,
            self.train_pct,
            self.valid_pct,
            self.test_pct,
        )
        return {"train": n_train, "valid": n_valid, "test": n_test}

    def get_mapping_fields_for_format(self) -> List[Dict[str, Any]]:
        """Return the column fields relevant to the current format."""
        cols = self.loaded_dataset.columns if self.loaded_dataset else []
        if self.target_format == MLXFormat.TEXT:
            return [
                {
                    "key": "text_col",
                    "label": "Text Column",
                    "current": self.mapping.text_col,
                    "help": "Column containing causal text",
                },
                {
                    "key": "text_template",
                    "label": "Text Template",
                    "current": self.mapping.text_template,
                    "help": "Optional template e.g. {input}\n{output}",
                },
            ]
        elif self.target_format == MLXFormat.PROMPT_COMPLETION:
            return [
                {
                    "key": "prompt_col",
                    "label": "Prompt / Question",
                    "current": self.mapping.prompt_col,
                    "help": "Column containing prompt or instruction",
                },
                {
                    "key": "completion_col",
                    "label": "Completion / Answer",
                    "current": self.mapping.completion_col,
                    "help": "Column containing response or completion",
                },
            ]
        elif self.target_format == MLXFormat.CHAT:
            if self.mapping.messages_col:
                return [
                    {
                        "key": "messages_col",
                        "label": "Messages List Col",
                        "current": self.mapping.messages_col,
                        "help": "List of role/content dicts",
                    },
                ]
            else:
                return [
                    {
                        "key": "user_col",
                        "label": "User Turn Col",
                        "current": self.mapping.user_col,
                        "help": "Column containing user message",
                    },
                    {
                        "key": "assistant_col",
                        "label": "Assistant Col",
                        "current": self.mapping.assistant_col,
                        "help": "Column containing assistant message",
                    },
                    {
                        "key": "system_col",
                        "label": "System Prompt Col",
                        "current": self.mapping.system_col,
                        "help": "Optional system prompt column",
                    },
                ]
        elif self.target_format == MLXFormat.DPO:
            return [
                {
                    "key": "prompt_col",
                    "label": "Prompt Col",
                    "current": self.mapping.dpo_prompt_col or self.mapping.prompt_col,
                    "help": "Column containing prompt",
                },
                {
                    "key": "chosen_col",
                    "label": "Chosen Response",
                    "current": self.mapping.chosen_col,
                    "help": "Preferred response",
                },
                {
                    "key": "rejected_col",
                    "label": "Rejected Response",
                    "current": self.mapping.rejected_col,
                    "help": "Dispreferred response",
                },
            ]
        return []

    def set_mapping_field(self, key: str, value: Optional[str]) -> None:
        """Update a specific mapping field and refresh preview."""
        if hasattr(self.mapping, key):
            setattr(self.mapping, key, value)
            if key == "prompt_col" and self.target_format == MLXFormat.DPO:
                self.mapping.dpo_prompt_col = value
            elif key == "dpo_prompt_col":
                self.mapping.prompt_col = value
            self.update_preview()

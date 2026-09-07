"""
MLX Dataset Format Definitions and Record Formatting Logic.
Supports Apple MLX (mlx-lm) fine-tuning formats:
- Text format: {"text": "..."}
- Chat / Messages format: {"messages": [{"role": "...", "content": "..."}]}
- Prompt & Completion format: {"prompt": "...", "completion": "..."}
- DPO / Preference format: {"prompt": "...", "chosen": "...", "rejected": "..."}
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class MLXFormat(str, Enum):
    TEXT = "text"
    CHAT = "chat"
    PROMPT_COMPLETION = "prompt_completion"
    DPO = "dpo"

    @classmethod
    def display_names(cls) -> Dict["MLXFormat", str]:
        return {
            cls.TEXT: "Text Format (Causal LM / Pre-training)",
            cls.CHAT: "Chat / Messages Format (Multi-turn conversations)",
            cls.PROMPT_COMPLETION: "Prompt & Completion Format (Instruction / Q&A)",
            cls.DPO: "DPO / Preference Format (Direct Preference Optimization)",
        }

    @classmethod
    def descriptions(cls) -> Dict["MLXFormat", str]:
        return {
            cls.TEXT: 'Schema: {"text": "..."} — Standard causal language modeling.',
            cls.CHAT: 'Schema: {"messages": [{"role": "system|user|assistant", "content": "..."}]} — Full dialogue fine-tuning.',
            cls.PROMPT_COMPLETION: 'Schema: {"prompt": "...", "completion": "..."} — Fine-tune model to complete specific prompts (supports --mask-prompt).',
            cls.DPO: 'Schema: {"prompt": "...", "chosen": "...", "rejected": "..."} — Preference alignment fine-tuning.',
        }


@dataclass
class ColumnMapping:
    # Text format
    text_col: Optional[str] = None
    text_template: Optional[str] = None  # e.g., "{instruction}\n\n{output}"

    # Chat format: Option A (pre-structured messages column)
    messages_col: Optional[str] = None
    role_key: str = "role"
    content_key: str = "content"
    role_map: Dict[str, str] = field(default_factory=lambda: {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "system": "system",
        "bot": "assistant",
    })

    # Chat format: Option B (separate role columns)
    system_col: Optional[str] = None
    user_col: Optional[str] = None
    assistant_col: Optional[str] = None

    # Prompt & Completion format
    prompt_col: Optional[str] = None
    completion_col: Optional[str] = None
    prompt_template: Optional[str] = None

    # DPO format
    dpo_prompt_col: Optional[str] = None
    chosen_col: Optional[str] = None
    rejected_col: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def extract_template_vars(template: str) -> List[str]:
    """Find all {variable_name} references in a format string."""
    return re.findall(r"\{([a-zA-Z0-9_]+)\}", template)


def parse_column_list(
    col_spec: Optional[str],
    available_columns: Optional[List[str]] = None,
) -> List[str]:
    """
    Parse a single column name, plus-separated ('a + b'), or comma-separated ('a, b') list.
    If the full col_spec exists directly in available_columns, it is treated as a single column name.
    """
    if not col_spec:
        return []
    cleaned = col_spec.strip()
    if not cleaned:
        return []
    if available_columns and cleaned in available_columns:
        return [cleaned]
    if "+" in cleaned:
        parts = [c.strip() for c in cleaned.split("+")]
        return [c for c in parts if c]
    if "," in cleaned:
        parts = [c.strip() for c in cleaned.split(",")]
        return [c for c in parts if c]
    return [cleaned]


def get_column_value(
    record: Dict[str, Any],
    col_spec: Optional[str],
    separator: str = "\n\n",
) -> str:
    """
    Extract and concatenate values from one or more columns in record.
    Joins non-empty string values using separator (default double newline \n\n).
    """
    if not col_spec:
        return ""
    cols = parse_column_list(col_spec)
    parts: List[str] = []
    for c in cols:
        if c in record and record[c] is not None:
            val = str(record[c]).strip()
            if val:
                parts.append(val)
    return separator.join(parts)


def format_template(template: str, record: Dict[str, Any]) -> str:
    """Format template string safely, filling missing keys with empty string."""
    class SafeDict(dict):
        def __missing__(self, key: str) -> str:
            return ""

    safe_record = SafeDict({k: str(v) if v is not None else "" for k, v in record.items()})
    try:
        return template.format_map(safe_record)
    except Exception:
        # Fallback simple replacement if braces in text cause issues
        res = template
        for k, v in record.items():
            res = res.replace(f"{{{k}}}", str(v) if v is not None else "")
        return res


def validate_mapping(
    format_type: MLXFormat,
    mapping: ColumnMapping,
    available_columns: List[str],
) -> List[str]:
    """Validate that the required columns or templates exist for the selected format."""
    errors = []
    col_set = set(available_columns)

    def check_cols(spec: Optional[str], field_name: str, required: bool = True) -> None:
        if not spec:
            if required:
                errors.append(f"{field_name} column must be specified.")
            return
        cols = parse_column_list(spec, available_columns)
        if not cols and required:
            errors.append(f"{field_name} column must be specified.")
            return
        for c in cols:
            if c not in col_set:
                errors.append(f"{field_name} column '{c}' not found in dataset columns.")

    if format_type == MLXFormat.TEXT:
        if mapping.text_template:
            vars_needed = extract_template_vars(mapping.text_template)
            missing = [v for v in vars_needed if v not in col_set]
            if missing:
                errors.append(f"Template references missing column(s): {', '.join(missing)}")
        elif mapping.text_col:
            check_cols(mapping.text_col, "Text", required=True)
        else:
            errors.append("Either a text column or a text template must be specified.")

    elif format_type == MLXFormat.CHAT:
        if mapping.messages_col:
            if mapping.messages_col not in col_set:
                errors.append(f"Messages column '{mapping.messages_col}' not found.")
        elif mapping.user_col and mapping.assistant_col:
            check_cols(mapping.user_col, "User", required=True)
            check_cols(mapping.assistant_col, "Assistant", required=True)
            if mapping.system_col:
                check_cols(mapping.system_col, "System", required=False)
        else:
            errors.append("Chat format requires either a messages list column OR user and assistant columns.")

    elif format_type == MLXFormat.PROMPT_COMPLETION:
        if mapping.prompt_template:
            vars_needed = extract_template_vars(mapping.prompt_template)
            missing = [v for v in vars_needed if v not in col_set]
            if missing:
                errors.append(f"Template references missing column(s): {', '.join(missing)}")
        else:
            check_cols(mapping.prompt_col, "Prompt", required=True)

        check_cols(mapping.completion_col, "Completion", required=True)

    elif format_type == MLXFormat.DPO:
        check_cols(mapping.dpo_prompt_col or mapping.prompt_col, "Prompt", required=True)
        check_cols(mapping.chosen_col, "Chosen", required=True)
        check_cols(mapping.rejected_col, "Rejected", required=True)

    return errors


def normalize_role(role_raw: Any, role_map: Dict[str, str]) -> str:
    """Normalize role names to standard MLX roles (system, user, assistant)."""
    val = str(role_raw).lower().strip()
    return role_map.get(val, val)


def format_record(
    record: Dict[str, Any],
    format_type: MLXFormat,
    mapping: ColumnMapping,
) -> Dict[str, Any]:
    """
    Format an individual dataset record into the selected MLX JSON schema.
    Returns a dictionary suitable for JSONL serialization.
    """
    if format_type == MLXFormat.TEXT:
        if mapping.text_template:
            text = format_template(mapping.text_template, record)
        elif mapping.text_col:
            text = get_column_value(record, mapping.text_col)
        else:
            text = ""
        return {"text": text}

    elif format_type == MLXFormat.CHAT:
        messages: List[Dict[str, str]] = []

        if mapping.messages_col and mapping.messages_col in record:
            raw_msgs = record[mapping.messages_col]
            if isinstance(raw_msgs, list):
                for item in raw_msgs:
                    if isinstance(item, dict):
                        raw_role = item.get(mapping.role_key, "user")
                        role = normalize_role(raw_role, mapping.role_map)
                        content = str(item.get(mapping.content_key, ""))
                        messages.append({"role": role, "content": content})
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        role = normalize_role(item[0], mapping.role_map)
                        content = str(item[1])
                        messages.append({"role": role, "content": content})
        else:
            # Multi-column mapping
            if mapping.system_col:
                sys_content = get_column_value(record, mapping.system_col)
                if sys_content:
                    messages.append({
                        "role": "system",
                        "content": sys_content,
                    })
            if mapping.user_col:
                user_content = get_column_value(record, mapping.user_col)
                messages.append({
                    "role": "user",
                    "content": user_content,
                })
            if mapping.assistant_col:
                asst_content = get_column_value(record, mapping.assistant_col)
                messages.append({
                    "role": "assistant",
                    "content": asst_content,
                })

        return {"messages": messages}

    elif format_type == MLXFormat.PROMPT_COMPLETION:
        if mapping.prompt_template:
            prompt = format_template(mapping.prompt_template, record)
        elif mapping.prompt_col:
            prompt = get_column_value(record, mapping.prompt_col)
        else:
            prompt = ""

        completion = get_column_value(record, mapping.completion_col) if mapping.completion_col else ""

        return {"prompt": prompt, "completion": completion}

    elif format_type == MLXFormat.DPO:
        p_col = mapping.dpo_prompt_col or mapping.prompt_col
        if p_col and p_col in record and isinstance(record[p_col], (dict, list)):
            p_val = record[p_col]
        else:
            p_val = get_column_value(record, p_col)

        if mapping.chosen_col and mapping.chosen_col in record and isinstance(record[mapping.chosen_col], (dict, list)):
            c_val = record[mapping.chosen_col]
        else:
            c_val = get_column_value(record, mapping.chosen_col)

        if mapping.rejected_col and mapping.rejected_col in record and isinstance(record[mapping.rejected_col], (dict, list)):
            r_val = record[mapping.rejected_col]
        else:
            r_val = get_column_value(record, mapping.rejected_col)

        return {
            "prompt": p_val,
            "chosen": c_val,
            "rejected": r_val,
        }

    raise ValueError(f"Unknown MLX format: {format_type}")


def auto_detect_mapping(format_type: MLXFormat, columns: List[str]) -> ColumnMapping:
    """Attempt to intelligently auto-detect column mappings based on common field names."""
    cols_lower = {col.lower(): col for col in columns}
    mapping = ColumnMapping()

    if format_type == MLXFormat.TEXT:
        for candidate in ["text", "content", "body", "document", "raw_text", "sentence"]:
            if candidate in cols_lower:
                mapping.text_col = cols_lower[candidate]
                break
        if not mapping.text_col and columns:
            mapping.text_col = columns[0]

    elif format_type == MLXFormat.CHAT:
        # Check for list of messages first
        for candidate in ["messages", "conversations", "dialog", "dialogue", "chat"]:
            if candidate in cols_lower:
                mapping.messages_col = cols_lower[candidate]
                # Check for ShareGPT style
                if candidate == "conversations":
                    mapping.role_key = "from"
                    mapping.content_key = "value"
                return mapping

        # Check for separate role columns
        for candidate in ["system", "system_prompt", "instruction_system"]:
            if candidate in cols_lower:
                mapping.system_col = cols_lower[candidate]
                break

        for candidate in ["user", "prompt", "instruction", "input", "query", "question"]:
            if candidate in cols_lower:
                mapping.user_col = cols_lower[candidate]
                break

        for candidate in ["assistant", "response", "output", "completion", "answer"]:
            if candidate in cols_lower:
                mapping.assistant_col = cols_lower[candidate]
                break

    elif format_type == MLXFormat.PROMPT_COMPLETION:
        for candidate in ["prompt", "instruction", "input", "query", "question", "context"]:
            if candidate in cols_lower:
                mapping.prompt_col = cols_lower[candidate]
                break

        for candidate in ["completion", "output", "response", "answer", "target"]:
            if candidate in cols_lower:
                mapping.completion_col = cols_lower[candidate]
                break

    elif format_type == MLXFormat.DPO:
        for candidate in ["prompt", "instruction", "input", "question"]:
            if candidate in cols_lower:
                mapping.dpo_prompt_col = cols_lower[candidate]
                break

        for candidate in ["chosen", "preferred", "accepted", "positive"]:
            if candidate in cols_lower:
                mapping.chosen_col = cols_lower[candidate]
                break

        for candidate in ["rejected", "dispreferred", "negative"]:
            if candidate in cols_lower:
                mapping.rejected_col = cols_lower[candidate]
                break

    return mapping

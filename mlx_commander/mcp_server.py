"""
Model Context Protocol (MCP) Server for MLX Commander.
Exposes dataset inspection, pre-populated interactive TUI launching,
and headless conversion tools to AI agents (Claude Desktop, Cursor, Antigravity, Cline, Zed).
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from mlx_commander import __version__
from mlx_commander.converter import convert_and_save
from mlx_commander.formats import ColumnMapping, MLXFormat, auto_detect_mapping, validate_mapping
from mlx_commander.loader import load_local_dataset
from mlx_commander.splitter import SplitConfig, generate_random_seed
from mlx_commander.terminal_spawner import is_macos, spawn_terminal_tui


def inspect_dataset_tool(dataset_path: str) -> Dict[str, Any]:
    """Inspect a dataset file or folder and return schema, rows, splits, and suggested mapping."""
    try:
        ds = load_local_dataset(dataset_path)
    except Exception as e:
        return {"status": "error", "message": f"Failed to load dataset: {e}"}

    sample_records = ds.get_all_records()[:3]

    # Auto-detect suggested mappings for each format
    suggested_mappings = {}
    for fmt in MLXFormat:
        m = auto_detect_mapping(fmt, ds.columns)
        errs = validate_mapping(fmt, m, ds.columns)
        suggested_mappings[fmt.value] = {
            "mapping": {
                k: v for k, v in m.__dict__.items() if v
            },
            "is_valid": len(errs) == 0,
            "validation_errors": errs,
        }

    return {
        "status": "success",
        "dataset_path": ds.source_path,
        "total_rows": ds.total_rows,
        "columns": ds.columns,
        "splits": ds.split_names,
        "split_counts": ds.split_counts,
        "sample_records": sample_records,
        "suggested_mappings": suggested_mappings,
    }


def launch_conversion_tui_tool(
    dataset_path: str,
    format: str = "prompt_completion",
    prompt_col: Optional[str] = None,
    completion_col: Optional[str] = None,
    messages_col: Optional[str] = None,
    text_col: Optional[str] = None,
    text_template: Optional[str] = None,
    user_col: Optional[str] = None,
    assistant_col: Optional[str] = None,
    system_col: Optional[str] = None,
    chosen_col: Optional[str] = None,
    rejected_col: Optional[str] = None,
    train_pct: float = 80.0,
    valid_pct: float = 10.0,
    test_pct: float = 10.0,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Pre-populates and launches the MLX Commander interactive TUI in a macOS Terminal window.
    The user can review live JSONL preview, adjust settings with arrow keys, and press Convert.
    Synchronously awaits completion and returns the conversion manifest.
    """
    if not is_macos():
        return {
            "status": "error",
            "message": "Interactive TUI spawning is supported on macOS. For Linux or headless environments, use convert_dataset_headless.",
        }

    args = [
        "--tui",
        "--dataset", dataset_path,
        "--format", format,
        "--train", str(train_pct),
        "--valid", str(valid_pct),
        "--test", str(test_pct),
    ]
    if prompt_col:
        args.extend(["--prompt-col", prompt_col])
    if completion_col:
        args.extend(["--completion-col", completion_col])
    if messages_col:
        args.extend(["--messages-col", messages_col])
    if text_col:
        args.extend(["--text-col", text_col])
    if text_template:
        args.extend(["--text-template", text_template])
    if user_col:
        args.extend(["--user-col", user_col])
    if assistant_col:
        args.extend(["--assistant-col", assistant_col])
    if system_col:
        args.extend(["--system-col", system_col])
    if chosen_col:
        args.extend(["--chosen-col", chosen_col])
    if rejected_col:
        args.extend(["--rejected-col", rejected_col])
    if seed is not None:
        args.extend(["--seed", str(seed)])
    if output_dir:
        args.extend(["--output", output_dir])

    # Determine manifest destination
    p = Path(dataset_path).resolve()
    base_dir = p if p.is_dir() else p.parent
    target_out = Path(output_dir).resolve() if output_dir else (base_dir / "mlx_dataset")
    target_out.mkdir(parents=True, exist_ok=True)
    manifest_file = target_out / "mlx_manifest.json"

    ec = spawn_terminal_tui(args, manifest_path=str(manifest_file))
    if ec == 0 and manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            return {"status": "error", "message": f"Failed reading manifest: {e}"}
    elif ec == 130:
        return {"status": "cancelled", "message": "User cancelled or closed the TUI session without converting."}
    else:
        return {"status": "error", "message": f"TUI session exited with status code {ec}."}


def convert_dataset_headless_tool(
    dataset_path: str,
    format: str = "prompt_completion",
    prompt_col: Optional[str] = None,
    completion_col: Optional[str] = None,
    messages_col: Optional[str] = None,
    text_col: Optional[str] = None,
    text_template: Optional[str] = None,
    user_col: Optional[str] = None,
    assistant_col: Optional[str] = None,
    system_col: Optional[str] = None,
    chosen_col: Optional[str] = None,
    rejected_col: Optional[str] = None,
    train_pct: float = 80.0,
    valid_pct: float = 10.0,
    test_pct: float = 10.0,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Perform automated headless conversion to MLX JSONL without opening the TUI."""
    try:
        ds = load_local_dataset(dataset_path)
    except Exception as e:
        return {"status": "error", "message": f"Failed to load dataset: {e}"}

    try:
        target_fmt = MLXFormat(format.lower().strip())
    except ValueError:
        return {"status": "error", "message": f"Invalid format '{format}'. Must be one of: prompt_completion, chat, text, dpo"}

    mapping = auto_detect_mapping(target_fmt, ds.columns)

    if prompt_col:
        mapping.prompt_col = prompt_col
    if completion_col:
        mapping.completion_col = completion_col
    if messages_col:
        mapping.messages_col = messages_col
    if text_col:
        mapping.text_col = text_col
    if text_template:
        mapping.text_template = text_template
    if user_col:
        mapping.user_col = user_col
    if assistant_col:
        mapping.assistant_col = assistant_col
    if system_col:
        mapping.system_col = system_col
    if chosen_col:
        mapping.chosen_col = chosen_col
    if rejected_col:
        mapping.rejected_col = rejected_col

    split_cfg = SplitConfig(
        train_pct=train_pct,
        valid_pct=valid_pct,
        test_pct=test_pct,
        seed=seed if seed is not None else generate_random_seed(),
    )

    out = output_dir or str(ds.default_output_dir)
    try:
        res = convert_and_save(
            dataset=ds,
            format_type=target_fmt,
            mapping=mapping,
            output_dir=out,
            split_config=split_cfg,
        )
        return res.to_manifest_dict()
    except Exception as e:
        return {"status": "error", "message": str(e)}


def create_mcp_server():
    """Build FastMCP server with registered tools and prompt."""
    from mcp.server.fastmcp import FastMCP

    server = FastMCP(
        name="mlx_commander",
        instructions="MLX Commander: Tool suite for inspecting and converting Hugging Face datasets into Apple MLX format with pre-populated TUI review.",
    )

    @server.tool()
    def inspect_dataset(dataset_path: str) -> dict:
        """Inspect a local Hugging Face dataset (parquet, jsonl, arrow, csv, sqlite)
        and return column names, row counts, detected format, and sample records."""
        return inspect_dataset_tool(dataset_path)

    @server.tool()
    def launch_conversion_tui(
        dataset_path: str,
        format: str = "prompt_completion",
        prompt_col: Optional[str] = None,
        completion_col: Optional[str] = None,
        messages_col: Optional[str] = None,
        train_pct: float = 80.0,
        valid_pct: float = 10.0,
        test_pct: float = 10.0,
        output_dir: Optional[str] = None,
    ) -> dict:
        """Launch the MLX Commander persistent Norton Commander TUI dashboard in a macOS Terminal window
        with pre-populated settings for user visual confirmation and live JSONL preview. Returns conversion manifest."""
        return launch_conversion_tui_tool(
            dataset_path=dataset_path,
            format=format,
            prompt_col=prompt_col,
            completion_col=completion_col,
            messages_col=messages_col,
            train_pct=train_pct,
            valid_pct=valid_pct,
            test_pct=test_pct,
            output_dir=output_dir,
        )

    @server.tool()
    def convert_dataset_headless(
        dataset_path: str,
        format: str = "prompt_completion",
        prompt_col: Optional[str] = None,
        completion_col: Optional[str] = None,
        messages_col: Optional[str] = None,
        train_pct: float = 80.0,
        valid_pct: float = 10.0,
        test_pct: float = 10.0,
        output_dir: Optional[str] = None,
    ) -> dict:
        """Directly convert a dataset to MLX JSONL in the background without UI interaction."""
        return convert_dataset_headless_tool(
            dataset_path=dataset_path,
            format=format,
            prompt_col=prompt_col,
            completion_col=completion_col,
            messages_col=messages_col,
            train_pct=train_pct,
            valid_pct=valid_pct,
            test_pct=test_pct,
            output_dir=output_dir,
        )

    @server.prompt()
    def prepare_dataset_for_mlx(dataset_path: str) -> str:
        """Instructions and workflow prompt for preparing a dataset for MLX fine-tuning."""
        return (
            f"Please inspect the dataset at '{dataset_path}' using the inspect_dataset tool, "
            f"choose the appropriate MLX format (prompt_completion, chat, text, or dpo), "
            f"and launch the TUI with launch_conversion_tui so the user can verify the preview and convert."
        )

    return server


def run_mcp_server() -> int:
    """Run MCP server over stdio transport."""
    try:
        server = create_mcp_server()
        server.run(transport="stdio")
        return 0
    except ImportError:
        sys.stderr.write(
            "Error: 'mcp' package is required to run the MCP server.\n"
            "Install it via:\n"
            "    pip install 'mlx_commander[mcp]'\n"
            "or run with uvx:\n"
            "    uvx --with mcp mlx_commander --mcp\n"
        )
        return 1


if __name__ == "__main__":
    sys.exit(run_mcp_server())

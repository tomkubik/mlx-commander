"""
Command Line Interface (CLI) for mlx_commander converter.
Dispatches between full-screen Curses TUI, interactive terminal wizard,
and automated headless conversion.
"""

import argparse
import json
import os
import sys

# Ensure ncurses escape delay is 25ms to make ESC instantaneous in all TUI pickers
os.environ.setdefault("ESCDELAY", "25")

import curses
from pathlib import Path
from typing import List, Optional

from mlx_commander import __version__
from mlx_commander.converter import ConversionResult, convert_and_save
from mlx_commander.formats import (
    ColumnMapping,
    MLXFormat,
    auto_detect_mapping,
    validate_mapping,
)
from mlx_commander.loader import load_local_dataset
from mlx_commander.splitter import SplitConfig, generate_random_seed
from mlx_commander.tui.app import launch_tui
from mlx_commander.tui.wizard_fallback import run_interactive_wizard


def parse_mapping_arg(mapping_str: str) -> ColumnMapping:
    """Parse JSON or key=val,key=val string into ColumnMapping."""
    clean = mapping_str.strip()
    if clean.startswith("{"):
        d = json.loads(clean)
        return ColumnMapping(**d)

    mapping = ColumnMapping()
    for pair in clean.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            k, v = k.strip(), v.strip()
            if hasattr(mapping, k):
                setattr(mapping, k, v)
            elif k == "prompt":
                mapping.prompt_col = v
            elif k == "completion":
                mapping.completion_col = v
            elif k == "text":
                mapping.text_col = v
            elif k == "messages":
                mapping.messages_col = v
            elif k == "user":
                mapping.user_col = v
            elif k == "assistant":
                mapping.assistant_col = v
            elif k == "system":
                mapping.system_col = v
            elif k == "chosen":
                mapping.chosen_col = v
            elif k == "rejected":
                mapping.rejected_col = v
    return mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-commander",
        description="Convert Hugging Face datasets into Apple MLX (mlx-lm) format with TUI or CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch interactive TUI wizard:
  mlx-commander

  # Launch line-by-line CLI wizard:
  mlx-commander --no-tui

  # Direct conversion from CLI:
  mlx-commander -d ./my_hf_dataset -f prompt_completion -o ./mlx_out \
         --prompt-col question --completion-col answer \
         --train 80 --valid 10 --test 10 --seed 42

  # Combine multiple dataset files with schema verification & re-splitting:
  mlx-commander -d train.jsonl test.jsonl -f prompt_completion -o ./mlx_out \
         --prompt-col question --completion-col answer
        """,
    )

    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")

    # Core conversion flags
    parser.add_argument(
        "-d", "--dataset",
        nargs="+",
        type=str,
        help="Path(s) to local Hugging Face dataset folder or data file(s) (.parquet, .arrow, .jsonl, .csv). Multiple files will be verified for schema consistency and merged.",
    )
    parser.add_argument(
        "-f", "--format",
        type=str,
        choices=["text", "chat", "prompt_completion", "dpo"],
        help="Target MLX format: 'text', 'chat', 'prompt_completion', or 'dpo'.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Destination directory where train.jsonl, valid.jsonl, test.jsonl will be saved.",
    )

    # Split parameters
    parser.add_argument("--train", type=float, default=None, help="Train split percentage (e.g. 80.0).")
    parser.add_argument("--valid", type=float, default=None, help="Validation split percentage (e.g. 10.0).")
    parser.add_argument("--test", type=float, default=None, help="Test split percentage (e.g. 10.0, or 0 to omit).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible shuffling.")
    parser.add_argument("--keep-splits", action="store_true", help="Preserve existing splits without re-splitting.")

    # Column mapping flags
    parser.add_argument("--mapping", type=str, help="Column mapping as JSON string or key=val,key=val.")
    parser.add_argument("--text-col", type=str, help="Source column for 'text' format.")
    parser.add_argument("--text-template", type=str, help="Template string for 'text' format (e.g. '{instruction}\\n{output}').")
    parser.add_argument("--prompt-col", type=str, help="Source column for prompt / question.")
    parser.add_argument("--completion-col", type=str, help="Source column for completion / answer.")
    parser.add_argument("--messages-col", type=str, help="Source column containing chat messages list.")
    parser.add_argument("--user-col", type=str, help="Source column for user turn in chat format.")
    parser.add_argument("--assistant-col", type=str, help="Source column for assistant turn in chat format.")
    parser.add_argument("--system-col", type=str, help="Source column for system prompt in chat format.")
    parser.add_argument("--chosen-col", type=str, help="Source column for chosen response in DPO format.")
    parser.add_argument("--rejected-col", type=str, help="Source column for rejected response in DPO format.")

    # UI mode flags
    parser.add_argument(
        "--commander", "--tui",
        action="store_true",
        dest="commander",
        help="Launch full-screen persistent MLX-Commander TUI dashboard (default in interactive terminal).",
    )
    parser.add_argument(
        "--wizard", "--no-tui", "--cli",
        action="store_true",
        dest="wizard",
        help="Run sequential step-by-step terminal wizard instead of persistent MLX-Commander dashboard.",
    )

    return parser


def run_direct_conversion(args: argparse.Namespace) -> ConversionResult:
    """Perform headless conversion using command-line arguments."""
    dataset = load_local_dataset(args.dataset)
    target_format = MLXFormat(args.format.lower())

    # Build mapping
    if args.mapping:
        mapping = parse_mapping_arg(args.mapping)
    else:
        mapping = auto_detect_mapping(target_format, dataset.columns)
        if args.text_col:
            mapping.text_col = args.text_col
        if args.text_template:
            mapping.text_template = args.text_template
        if args.prompt_col:
            mapping.prompt_col = args.prompt_col
        if args.completion_col:
            mapping.completion_col = args.completion_col
        if args.messages_col:
            mapping.messages_col = args.messages_col
        if args.user_col:
            mapping.user_col = args.user_col
        if args.assistant_col:
            mapping.assistant_col = args.assistant_col
        if args.system_col:
            mapping.system_col = args.system_col
        if args.chosen_col:
            mapping.chosen_col = args.chosen_col
        if args.rejected_col:
            mapping.rejected_col = args.rejected_col

    # Validate mapping
    errs = validate_mapping(target_format, mapping, dataset.columns)
    if errs:
        raise ValueError("Mapping error: " + "; ".join(errs))

    # Split config
    split_config = None
    if not (args.keep_splits and dataset.is_split):
        train_p = args.train if args.train is not None else 80.0
        valid_p = args.valid if args.valid is not None else 10.0
        test_p = args.test if args.test is not None else (100.0 - train_p - valid_p)
        seed = args.seed if args.seed is not None else generate_random_seed()
        split_config = SplitConfig(train_pct=train_p, valid_pct=valid_p, test_pct=test_p, seed=seed)
        v_errs = split_config.validate()
        if v_errs:
            raise ValueError("Split error: " + "; ".join(v_errs))

    return convert_and_save(
        dataset=dataset,
        format_type=target_format,
        mapping=mapping,
        output_dir_str=args.output,
        split_config=split_config,
        use_existing_splits=args.keep_splits,
    )


def is_interactive_tty() -> bool:
    """Check if stdout and stdin are interactive TTYs with adequate terminfo."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return False
    term = os.environ.get("TERM", "")
    if not term or term == "dumb":
        return False
    return True


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Normalize dataset path argument
    dataset_input = None
    if args.dataset:
        if isinstance(args.dataset, list):
            dataset_input = "\n".join(args.dataset) if len(args.dataset) > 1 else args.dataset[0]
        else:
            dataset_input = args.dataset

    # Case 1: All required CLI flags provided -> Direct Headless Run
    if args.dataset and args.format and args.output:
        try:
            result = run_direct_conversion(args)
            src_desc = f"{len(args.dataset)} files (merged)" if isinstance(args.dataset, list) and len(args.dataset) > 1 else (args.dataset[0] if isinstance(args.dataset, list) else str(args.dataset))
            print(f"✔ Successfully converted {src_desc} to {args.format} format in {result.output_dir}")
            for s_name, path in result.output_files.items():
                cnt = result.record_counts.get(s_name, 0)
                print(f"  • {path.name}: {cnt:,} records")
            print(f"\nMLX Fine-tuning command:\n{result.generate_mlx_lora_command()}\n")
            return 0
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return 130
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    # Case 2: Sequential wizard explicitly requested via --wizard or when non-interactive
    if args.wizard or not is_interactive_tty():
        try:
            run_interactive_wizard(
                dataset_path=dataset_input,
                format_arg=args.format,
                output_dir_arg=args.output,
                train_pct_arg=args.train,
                valid_pct_arg=args.valid,
                test_pct_arg=args.test,
                seed_arg=args.seed,
            )
            return 0
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")
            return 130
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)
            return 1

    # Case 3: Default — Launch persistent MLX-Commander full-screen TUI dashboard
    try:
        result = launch_tui(default_dataset_path=dataset_input)
        return 0 if result is not None else 130
    except (KeyboardInterrupt, curses.error):
        print("\nOperation cancelled.")
        return 130
    except Exception as e:
        print(f"\nTerminal notice: {e}. Falling back to CLI wizard...", file=sys.stderr)
        try:
            run_interactive_wizard(dataset_path=dataset_input)
            return 0
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")
            return 130


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        sys.exit(130)

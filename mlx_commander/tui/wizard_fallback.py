"""
Interactive Terminal Wizard (ANSI / Line-by-Line fallback).
Used when curses is unavailable (e.g. non-standard terminals, redirected stdin)
or when invoked with --no-tui.
"""

import json
import os
import select
import shutil
import sys
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

# ANSI Styling helpers
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


try:
    import termios
    import tty

    HAVE_TERMIOS = True
except ImportError:
    HAVE_TERMIOS = False

try:
    import readline
except ImportError:
    pass


def ask_input(prompt: str, default: Optional[str] = None) -> str:
    """Prompt user for input with an optional default value, supporting standard terminal editing."""
    def_hint = f" {DIM}[default: {default}]{RESET}" if default is not None else ""
    full_prompt = f"{BOLD}{prompt}{RESET}{def_hint}: "
    try:
        val = input(full_prompt)
        clean = val.strip()
        return clean if clean else (default or "")
    except EOFError:
        return default or ""


def _read_char(fd: int) -> str:
    """Read a single UTF-8 character or escape sequence from terminal in raw/cbreak mode."""
    ch = os.read(fd, 1).decode("utf-8", errors="ignore")
    if not ch:
        return "ENTER"
    if ch == "\x03":
        raise KeyboardInterrupt
    if ch == "\x1b":
        # Check if more bytes are immediately queued (escape sequence)
        r, _, _ = select.select([fd], [], [], 0.02)
        if not r:
            return "ESC"
        ch2 = os.read(fd, 1).decode("utf-8", errors="ignore")
        if ch2 in ("[", "O"):
            r, _, _ = select.select([fd], [], [], 0.02)
            if not r:
                return f"\x1b{ch2}"
            ch3 = os.read(fd, 1).decode("utf-8", errors="ignore")
            if ch3 == "A":
                return "UP"
            elif ch3 == "B":
                return "DOWN"
            elif ch3 == "C":
                return "RIGHT"
            elif ch3 == "D":
                return "LEFT"
            elif ch3 == "H":
                return "HOME"
            elif ch3 == "F":
                return "END"
            elif ch3 in ("1", "2", "3", "4", "5", "6"):
                r, _, _ = select.select([fd], [], [], 0.02)
                if r:
                    os.read(fd, 1)  # swallow trailing ~
                if ch3 == "5":
                    return "PAGE_UP"
                elif ch3 == "6":
                    return "PAGE_DOWN"
            return f"\x1b{ch2}{ch3}"
        return f"\x1b{ch2}"
    if ch in ("\r", "\n"):
        return "ENTER"
    if ch == " ":
        return "SPACE"
    if ch in ("\x7f", "\x08"):
        return "BACKSPACE"
    return ch


def _ask_choice_fallback(title: str, options: List[str], default_idx: int = 0) -> int:
    """Prompt user to choose from a numbered list of options (line-by-line fallback)."""
    print(f"\n{BOLD}{title}{RESET}")
    for idx, opt in enumerate(options):
        marker = f"{CYAN}*{RESET}" if idx == default_idx else " "
        print(f"  {marker} {BOLD}{idx + 1}{RESET}) {opt}")

    while True:
        choice_str = ask_input("Select an option number", default=str(default_idx + 1))
        try:
            val = int(choice_str) - 1
            if 0 <= val < len(options):
                return val
        except ValueError:
            pass
        print(f"{RED}Invalid choice. Please enter a number between 1 and {len(options)}.{RESET}")


def _ask_choice_interactive(title: str, options: List[str], default_idx: int = 0) -> int:
    """Interactive choice selection supporting Up/Down arrow keys, number keys, and Enter."""
    if not options:
        return 0

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    VISIBLE_LIMIT = 8
    selected_idx = max(0, min(default_idx, len(options) - 1))
    num_buffer = ""

    show_scroll = len(options) > VISIBLE_LIMIT
    total_lines = 1 + (VISIBLE_LIMIT + 2 if show_scroll else len(options))

    # Print title once
    print(f"\n{BOLD}{title}{RESET}")

    def render(lines_to_clear: int = 0):
        if lines_to_clear > 0:
            sys.stdout.write(f"\033[{lines_to_clear}A\r")

        term_cols = shutil.get_terminal_size(fallback=(80, 24)).columns
        if term_cols < 40:
            term_cols = 80
        max_opt_len = max(40, term_cols - 10)

        # Line 1: Hint
        num_display = f" {CYAN}[{num_buffer}]{RESET}" if num_buffer else ""
        sys.stdout.write(f"\033[2K  {DIM}(Use ↑/↓ arrows or type number, Enter to confirm){RESET}{num_display}\n")

        if show_scroll:
            half = VISIBLE_LIMIT // 2
            if selected_idx < half:
                start_idx = 0
            elif selected_idx >= len(options) - (VISIBLE_LIMIT - half):
                start_idx = len(options) - VISIBLE_LIMIT
            else:
                start_idx = selected_idx - half
            end_idx = start_idx + VISIBLE_LIMIT

            # Up indicator
            if start_idx > 0:
                sys.stdout.write(f"\033[2K    {DIM}▲ ({start_idx} more above){RESET}\n")
            else:
                sys.stdout.write("\033[2K\n")

            # Visible options
            for idx in range(start_idx, end_idx):
                opt = options[idx]
                if len(opt) > max_opt_len:
                    opt = opt[: max_opt_len - 1] + "…"
                if idx == selected_idx:
                    sys.stdout.write(f"\033[2K  {CYAN}❯{RESET} {BOLD}{CYAN}{idx + 1}) {opt}{RESET}\n")
                else:
                    sys.stdout.write(f"\033[2K    {DIM}{idx + 1}) {opt}{RESET}\n")

            # Down indicator
            below_count = len(options) - end_idx
            if below_count > 0:
                sys.stdout.write(f"\033[2K    {DIM}▼ ({below_count} more below){RESET}\n")
            else:
                sys.stdout.write("\033[2K\n")
        else:
            for idx, opt in enumerate(options):
                opt_str = opt
                if len(opt_str) > max_opt_len:
                    opt_str = opt_str[: max_opt_len - 1] + "…"
                if idx == selected_idx:
                    sys.stdout.write(f"\033[2K  {CYAN}❯{RESET} {BOLD}{CYAN}{idx + 1}) {opt_str}{RESET}\n")
                else:
                    sys.stdout.write(f"\033[2K    {DIM}{idx + 1}) {opt_str}{RESET}\n")

        sys.stdout.flush()

    try:
        # Hide cursor during navigation
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()

        # Unbuffered character input with interrupt signals intact
        tty.setcbreak(fd)

        # First render
        render(lines_to_clear=0)

        while True:
            key = _read_char(fd)

            if key in ("UP", "k", "K"):
                selected_idx = (selected_idx - 1) % len(options)
                num_buffer = ""
                render(lines_to_clear=total_lines)

            elif key in ("DOWN", "j", "J"):
                selected_idx = (selected_idx + 1) % len(options)
                num_buffer = ""
                render(lines_to_clear=total_lines)

            elif key in ("PAGE_UP", "HOME"):
                selected_idx = 0 if key == "HOME" else max(0, selected_idx - 5)
                num_buffer = ""
                render(lines_to_clear=total_lines)

            elif key in ("PAGE_DOWN", "END"):
                selected_idx = (len(options) - 1) if key == "END" else min(len(options) - 1, selected_idx + 5)
                num_buffer = ""
                render(lines_to_clear=total_lines)

            elif key in ("ENTER", "SPACE"):
                break

            elif key == "BACKSPACE":
                if num_buffer:
                    num_buffer = num_buffer[:-1]
                    if num_buffer:
                        try:
                            val = int(num_buffer) - 1
                            if 0 <= val < len(options):
                                selected_idx = val
                        except ValueError:
                            pass
                    render(lines_to_clear=total_lines)

            elif key == "ESC":
                num_buffer = ""
                render(lines_to_clear=total_lines)

            elif key.isdigit():
                new_buf = num_buffer + key
                try:
                    val = int(new_buf) - 1
                    if 0 <= val < len(options):
                        selected_idx = val
                        num_buffer = new_buf
                    elif len(key) == 1 and 0 <= int(key) - 1 < len(options):
                        selected_idx = int(key) - 1
                        num_buffer = key
                except ValueError:
                    pass
                render(lines_to_clear=total_lines)

    finally:
        # Guarantee terminal restoration and unhide cursor
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()

    # Clear options list and display the confirmed selection cleanly
    sys.stdout.write(f"\033[{total_lines}A\r")
    sys.stdout.write(f"\033[2K  {GREEN}*{RESET} {BOLD}{selected_idx + 1}) {options[selected_idx]}{RESET}\n")
    for _ in range(total_lines - 1):
        sys.stdout.write("\033[2K\n")
    sys.stdout.write(f"\033[{total_lines - 1}A\r")
    sys.stdout.flush()

    return selected_idx


def ask_choice(title: str, options: List[str], default_idx: int = 0) -> int:
    """Prompt user to choose from a numbered list of options.
    In interactive terminals (TTY), supports Up/Down cursor arrows, typing numbers,
    and Enter/Space to confirm, with automatic scrolling for long lists.
    In non-TTY environments (pipes, scripts, tests), uses standard line input.
    """
    if not (HAVE_TERMIOS and sys.stdin.isatty() and sys.stdout.isatty()):
        return _ask_choice_fallback(title, options, default_idx)
    try:
        return _ask_choice_interactive(title, options, default_idx)
    except Exception:
        return _ask_choice_fallback(title, options, default_idx)


def select_column_cli(
    prompt: str,
    columns: List[str],
    default_col: Optional[str] = None,
    allow_none: bool = False,
    allow_multiple: bool = True,
) -> Optional[str]:
    """Prompt user to pick or concatenate columns from the list."""
    opts = []
    if allow_none:
        opts.append("(None / Skip)")
    opts.extend(columns)
    if allow_multiple and len(columns) > 1:
        opts.append("[+] Combine multiple columns (Concatenate with ' + ')")

    def_idx = 0
    if default_col and default_col in columns:
        def_idx = columns.index(default_col) + (1 if allow_none else 0)

    choice = ask_choice(prompt, opts, default_idx=def_idx)
    if allow_none and choice == 0:
        return None
    offset = 1 if allow_none else 0
    if allow_multiple and choice == len(opts) - 1:
        print(f"\n{BOLD}Available columns:{RESET} " + ", ".join(f"{i+1}) {c}" for i, c in enumerate(columns)))
        resp = ask_input(
            "Enter column numbers or names to concatenate (e.g., '1, 2' or 'instruction + input')",
            default=default_col or columns[0],
        )
        from mlx_commander.formats import parse_column_list
        parts = parse_column_list(resp)
        resolved = []
        for p in parts:
            if p.isdigit() and 1 <= int(p) <= len(columns):
                resolved.append(columns[int(p) - 1])
            elif p in columns:
                resolved.append(p)
        return " + ".join(resolved) if resolved else default_col

    return columns[choice - offset]


def run_interactive_wizard(
    dataset_path: Optional[str] = None,
    format_arg: Optional[str] = None,
    output_dir_arg: Optional[str] = None,
    train_pct_arg: Optional[float] = None,
    valid_pct_arg: Optional[float] = None,
    test_pct_arg: Optional[float] = None,
    seed_arg: Optional[int] = None,
) -> ConversionResult:
    """Run interactive CLI wizard step-by-step."""
    print(f"\n{CYAN}{BOLD}======================================================{RESET}")
    print(f"{CYAN}{BOLD}       Hugging Face ➜ Apple MLX Dataset Converter     {RESET}")
    print(f"{CYAN}{BOLD}======================================================{RESET}\n")

    # Step 1: Dataset path
    from mlx_commander.gui_picker import pick_folder_gui, pick_file_gui
    import os

    dataset: Optional[LoadedDataset] = None
    target_path = dataset_path
    default_dir = os.getcwd()

    while dataset is None:
        if not target_path:
            print(f"\n{BOLD}Dataset Source Selection{RESET}")
            print(f"Current working folder: {CYAN}{default_dir}{RESET}")
            path_choice = ask_choice(
                "How would you like to select your dataset?",
                [
                    "Open GUI Finder Picker (Select Dataset Folder or File)",
                    f"Use Current Working Folder ({default_dir})",
                    "Enter path manually",
                ],
                default_idx=0,
            )
            if path_choice == 0:
                print(f"{CYAN}Opening Finder window...{RESET}")
                from mlx_commander.gui_picker import pick_dataset_gui
                chosen = pick_dataset_gui(default_dir=default_dir)
                if not chosen:
                    print(f"{YELLOW}No path chosen. Please choose an option or enter path manually.{RESET}")
                    continue
                target_path = chosen
            elif path_choice == 1:
                target_path = default_dir
            else:
                target_path = ask_input("Enter path to Hugging Face dataset folder or data file on disk", default=default_dir)

        try:
            dataset = load_local_dataset(target_path)
        except Exception as e:
            print(f"{RED}Error loading dataset from '{target_path}': {e}{RESET}")
            target_path = None

    print(f"\n{GREEN}[OK] Dataset loaded successfully!{RESET}")
    print(f"  • Source: {dataset.source_path}")
    print(f"  • Total records: {dataset.total_rows:,}")
    print(f"  • Splits found: [{', '.join(dataset.split_names)}]")
    print(f"  • Columns ({len(dataset.columns)}): {', '.join(dataset.columns)}")

    # Step 2: Format
    format_choices = [
        "Text Format (Causal LM / Pre-training) [{\"text\": \"...\"}]",
        "Chat / Messages Format (Instruction / Dialogue) [{\"messages\": [...]}]",
        "Prompt & Completion Format (Q&A / Instructions) [{\"prompt\": \"...\", \"completion\": \"...\"}]",
        "DPO / Preference Format (Direct Preference Optimization) [{\"prompt\": \"...\", \"chosen\": \"...\", \"rejected\": \"...\"}]",
    ]
    formats_list = [
        MLXFormat.TEXT,
        MLXFormat.CHAT,
        MLXFormat.PROMPT_COMPLETION,
        MLXFormat.DPO,
    ]

    if format_arg:
        try:
            target_format = MLXFormat(format_arg.lower())
        except ValueError:
            target_format = formats_list[ask_choice("Select MLX Target Format", format_choices, default_idx=0)]
    else:
        fmt_idx = ask_choice("Select MLX Target Format", format_choices, default_idx=0)
        target_format = formats_list[fmt_idx]

    # Step 3: Column Mapping
    mapping = auto_detect_mapping(target_format, dataset.columns)

    if target_format == MLXFormat.TEXT:
        mode = ask_choice(
            "How would you like to create the 'text' field?",
            [
                f"Use a single column (Auto-detected: '{mapping.text_col or 'None'}')",
                "Use a custom format template (e.g., 'Instruction: {instruction}\\nResponse: {output}')",
            ],
            default_idx=0 if mapping.text_col else 1,
        )
        if mode == 0:
            mapping.text_col = select_column_cli(
                "Select column for text", dataset.columns, default_col=mapping.text_col
            )
            mapping.text_template = None
        else:
            default_tmpl = " ".join([f"{{{c}}}" for c in dataset.columns[:2]])
            mapping.text_template = ask_input("Enter template with {column_name} variables", default=default_tmpl)
            mapping.text_col = None

    elif target_format == MLXFormat.CHAT:
        has_msg_col = bool(mapping.messages_col)
        chat_mode = ask_choice(
            "How are conversation turns stored in your dataset?",
            [
                f"Existing messages column with turns (Auto-detected: '{mapping.messages_col or 'None'}')",
                "Separate columns for roles (e.g. system, user, assistant)",
            ],
            default_idx=0 if has_msg_col else 1,
        )
        if chat_mode == 0:
            mapping.messages_col = select_column_cli(
                "Select column containing messages list", dataset.columns, default_col=mapping.messages_col
            )
        else:
            mapping.messages_col = None
            mapping.system_col = select_column_cli(
                "Select System column (optional)", dataset.columns, default_col=mapping.system_col, allow_none=True
            )
            mapping.user_col = select_column_cli(
                "Select User column (query / prompt)", dataset.columns, default_col=mapping.user_col or dataset.columns[0]
            )
            mapping.assistant_col = select_column_cli(
                "Select Assistant column (response)", dataset.columns, default_col=mapping.assistant_col or (dataset.columns[1] if len(dataset.columns) > 1 else dataset.columns[0])
            )

    elif target_format == MLXFormat.PROMPT_COMPLETION:
        mapping.prompt_col = select_column_cli(
            "Select Prompt / Question column", dataset.columns, default_col=mapping.prompt_col or dataset.columns[0]
        )
        mapping.completion_col = select_column_cli(
            "Select Completion / Answer column", dataset.columns, default_col=mapping.completion_col or (dataset.columns[1] if len(dataset.columns) > 1 else dataset.columns[0])
        )

    elif target_format == MLXFormat.DPO:
        mapping.dpo_prompt_col = select_column_cli("Select Prompt column", dataset.columns, default_col=mapping.dpo_prompt_col)
        mapping.chosen_col = select_column_cli("Select Chosen Response column", dataset.columns, default_col=mapping.chosen_col)
        mapping.rejected_col = select_column_cli("Select Rejected Response column", dataset.columns, default_col=mapping.rejected_col)

    # Validate mapping
    errs = validate_mapping(target_format, mapping, dataset.columns)
    if errs:
        raise ValueError("Invalid mapping: " + "; ".join(errs))

    # Step 4: Splitting & Random Seed
    use_existing = False
    if dataset.is_split and len(dataset.split_names) > 1:
        split_opt = ask_choice(
            f"Dataset already has splits: {', '.join(dataset.split_names)}. What would you like to do?",
            [
                "Re-split the entire dataset by custom % and random seed",
                "Keep existing splits and convert them directly",
            ],
            default_idx=0,
        )
        use_existing = (split_opt == 1)

    split_config: Optional[SplitConfig] = None
    if not use_existing:
        while True:
            tr_pct = train_pct_arg
            if tr_pct is None:
                tr_str = ask_input("Train split percentage (%)", default="80")
                try:
                    tr_pct = float(tr_str)
                except ValueError:
                    print(f"{RED}Please enter a valid number.{RESET}")
                    continue

            va_pct = valid_pct_arg
            if va_pct is None:
                va_str = ask_input("Validation split percentage (%)", default="10")
                try:
                    va_pct = float(va_str)
                except ValueError:
                    print(f"{RED}Please enter a valid number.{RESET}")
                    continue

            te_pct = test_pct_arg
            if te_pct is None:
                te_str = ask_input("Test split percentage (%) [optional, 0 for none]", default="10")
                try:
                    te_pct = float(te_str)
                except ValueError:
                    print(f"{RED}Please enter a valid number.{RESET}")
                    continue

            # Random Seed
            gen_seed = generate_random_seed()
            if seed_arg is not None:
                seed_val = seed_arg
            else:
                seed_input = ask_input(f"Random seed for shuffling [auto-generated: {gen_seed}]", default=str(gen_seed))
                try:
                    seed_val = int(seed_input)
                except ValueError:
                    print(f"{RED}Seed must be an integer.{RESET}")
                    continue

            cfg = SplitConfig(train_pct=tr_pct, valid_pct=va_pct, test_pct=te_pct, seed=seed_val)
            validation_errors = cfg.validate()
            if validation_errors:
                print(f"{RED}{'; '.join(validation_errors)}{RESET}")
                # Reset args to re-prompt
                train_pct_arg = None
                valid_pct_arg = None
                test_pct_arg = None
            else:
                split_config = cfg
                break

    # Step 5: Output Folder & Confirmation
    default_out = str(dataset.default_output_dir)
    if output_dir_arg:
        dest_dir = output_dir_arg
    else:
        out_choice = ask_choice(
            "How would you like to select output destination folder?",
            [
                f"Use Default Output Folder ({default_out})",
                "Open GUI Folder Picker (Choose in macOS Finder)",
                "Enter path manually",
            ],
            default_idx=0,
        )
        if out_choice == 0:
            dest_dir = default_out
        elif out_choice == 1:
            print(f"{CYAN}Opening Finder folder selection window...{RESET}")
            chosen = pick_folder_gui("Select Destination Folder", default_dir=default_out)
            dest_dir = chosen if chosen else default_out
        else:
            dest_dir = ask_input("Directory to save MLX dataset files", default=default_out)

    # Preview sample record
    preview_sample = dataset.sample_records[:2]
    print(f"\n{BOLD}── Converted Sample Preview ({target_format.value.upper()}) ─────────────────────{RESET}")
    for s in preview_sample:
        formatted = format_record(s, target_format, mapping)
        print(f"  {DIM}{json.dumps(formatted, ensure_ascii=False)[:120]}...{RESET}")
    print(f"{BOLD}─────────────────────────────────────────────────────────────────{RESET}\n")

    if not output_dir_arg:
        confirm = ask_input("Ready to convert and write files? (Y/n)", default="Y")
        if confirm.lower() not in ("y", "yes"):
            print(f"{YELLOW}Conversion cancelled by user.{RESET}")
            sys.exit(0)

    print(f"\n{CYAN}Converting dataset and writing to {dest_dir}...{RESET}")
    result = convert_and_save(
        dataset=dataset,
        format_type=target_format,
        mapping=mapping,
        output_dir_str=dest_dir,
        split_config=split_config,
        use_existing_splits=use_existing,
    )

    print(f"\n{GREEN}{BOLD}Conversion Complete!{RESET}\n")
    print(f"Saved to directory: {BOLD}{result.output_dir}{RESET}")
    for s_name, path in result.output_files.items():
        cnt = result.record_counts.get(s_name, 0)
        size_kb = result.file_sizes.get(s_name, 0) / 1024
        print(f"  • {GREEN}{path.name}{RESET}: {cnt:,} records ({size_kb:.1f} KB)")

    print(f"\n{BOLD}Fine-tune with Apple MLX using this command:{RESET}")
    print(f"{YELLOW}──────────────────────────────────────────────────────────{RESET}")
    for line in result.generate_mlx_lora_command().split("\n"):
        print(f"  {line}")
    print(f"{YELLOW}──────────────────────────────────────────────────────────\n{RESET}")

    return result

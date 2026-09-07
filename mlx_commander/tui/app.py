"""
MLX-Commander: Persistent Full-Screen Curses TUI Dashboard.
Dual-panel Norton Commander-style interface with real-time reactive JSONL preview.

Layout:
┌─ Dataset & Schema (Left Panel) ─────────┐┌─ MLX Format & Mappings (Right Panel) ─────┐
│ Path, Stats, and Scrollable Column List   ││ Format Radio, Column Dropdowns, Splits     │
└───────────────────────────────────────────┘└────────────────────────────────────────────┘
┌─ Live Converted Record Preview (Updates instantaneously as you edit fields) ──────────┐
│ {"prompt": "...", "completion": "..."}                                                │
└───────────────────────────────────────────────────────────────────────────────────────┘
 [Tab] Switch Pane   [↑/↓] Navigate   [Enter] Edit/Select   [F2] Finder   [F5] Convert   [F10] Exit
"""

import curses
import json
import os
import textwrap
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
from mlx_commander.tui.state import ActivePanel, CommanderState
from mlx_commander.tui.widgets import (
    configure_escdelay,
    draw_box_panel,
    draw_button,
    draw_field,
    draw_footer,
    draw_header,
    draw_mapping_pipeline_panel,
    draw_radio,
    get_color,
    safe_addstr,
    show_column_picker_dialog,
    show_error_dialog,
    show_help_dialog,
    show_message_dialog,
    show_output_destination_dialog,
    show_results_dialog,
    show_text_edit_dialog,
)


def init_colors() -> None:
    """Initialize curses color pairs for MLX-Commander."""
    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        # Pair 1: Header / Footer Banner (White on Blue or Black)
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)
        # Pair 2: Active / Focus / Border (Cyan)
        curses.init_pair(2, curses.COLOR_CYAN, -1)
        # Pair 3: Success / Selected / Check (Green)
        curses.init_pair(3, curses.COLOR_GREEN, -1)
        # Pair 4: Normal text (White)
        curses.init_pair(4, curses.COLOR_WHITE, -1)
        # Pair 5: Warning / Error (Red)
        curses.init_pair(5, curses.COLOR_RED, -1)
        # Pair 6: Muted / Dim (Yellow)
        curses.init_pair(6, curses.COLOR_YELLOW, -1)


def execute_conversion(stdscr: curses.window, state: CommanderState) -> Optional[ConversionResult]:
    """Validate mappings, execute conversion, and display the result modal."""
    if not state.loaded_dataset:
        state.status_message = "Please load a dataset before converting."
        state.status_is_error = True
        show_error_dialog(stdscr, "Conversion Error", "Please load a dataset before converting.")
        return None

    errors = validate_mapping(state.target_format, state.mapping, state.loaded_dataset.columns)
    if errors:
        state.status_message = f"Cannot convert: {errors[0]}"
        state.status_is_error = True
        show_error_dialog(stdscr, "Cannot Convert", errors)
        return None

    state.status_message = "Converting dataset and writing JSONL files..."
    state.status_is_error = False
    stdscr.refresh()

    split_cfg = SplitConfig(
        train_pct=state.train_pct,
        valid_pct=state.valid_pct,
        test_pct=state.test_pct,
        seed=state.seed,
    )
    try:
        res = convert_and_save(
            dataset=state.loaded_dataset,
            format_type=state.target_format,
            mapping=state.mapping,
            output_dir=Path(state.output_dir),
            split_config=split_cfg,
        )
        show_results_dialog(stdscr, res)
        state.status_message = f"[OK] Conversion complete! Saved to {res.output_dir}"
        state.status_is_error = False
        return res
    except Exception as e:
        show_error_dialog(stdscr, "Conversion Error", str(e))
        state.status_message = f"Conversion error: {e}"
        state.status_is_error = True
        return None


def run_commander_tui(
    stdscr: curses.window,
    default_dataset_path: Optional[str] = None,
    initial_state: Optional[CommanderState] = None,
    prefill: Optional[Dict[str, Any]] = None,
) -> Optional[ConversionResult]:
    """Main event loop for the persistent MLX-Commander dashboard."""
    configure_escdelay(25)
    init_colors()
    curses.curs_set(0)
    stdscr.keypad(True)

    state = initial_state if initial_state is not None else CommanderState()

    if prefill:
        state.apply_prefill(prefill)

    if not state.loaded_dataset:
        initial_path = default_dataset_path or (state.dataset_path if state.dataset_path else os.getcwd())
        # Attempt to auto-load if valid dataset files are in initial directory
        try:
            p = Path(initial_path).resolve()
            has_dataset = (
                p.is_file()
                or any(p.glob("*.parquet"))
                or any(p.glob("*.jsonl"))
                or any(p.glob("*.arrow"))
                or any(p.glob("*.csv"))
                or any(p.glob("*.tsv"))
                or any(p.glob("*.sqlite"))
                or any(p.glob("*.db"))
                or any(p.glob("*.tar"))
                or any(p.glob("*.tar.gz"))
                or any(p.glob("dataset_info.json"))
            )
            if has_dataset:
                state.load_dataset(str(p))
            else:
                state.dataset_path = str(p)
                if not state.has_custom_output_dir:
                    base = p if p.is_dir() else p.parent
                    state.output_dir = str(base / "mlx_dataset")
        except Exception:
            state.dataset_path = initial_path
            if not state.has_custom_output_dir:
                try:
                    p = Path(initial_path).resolve()
                    base = p if p.is_dir() else p.parent
                    state.output_dir = str(base / "mlx_dataset")
                except Exception:
                    state.output_dir = str(Path.cwd() / "mlx_dataset")

    conversion_result: Optional[ConversionResult] = None
    formats_list = [
        MLXFormat.PROMPT_COMPLETION,
        MLXFormat.CHAT,
        MLXFormat.TEXT,
        MLXFormat.DPO,
    ]

    while True:
        stdscr.erase()
        max_y, max_x = stdscr.getmaxyx()

        # Check for minimum terminal dimension
        if max_y < 16 or max_x < 70:
            safe_addstr(stdscr, 1, 2, "Terminal window too small for MLX-Commander.", curses.A_BOLD)
            safe_addstr(stdscr, 2, 2, f"Current: {max_x}x{max_y} (Minimum required: 70x16)", curses.A_DIM)
            safe_addstr(stdscr, 4, 2, "Please resize your terminal window or press [q] to exit.", curses.A_DIM)
            stdscr.refresh()
            k = stdscr.getch()
            if k in (ord("q"), ord("Q"), 27):
                return None
            continue

        # ----------------------------------------------------
        # 1. Header Banner
        # ----------------------------------------------------
        hdr_attr = (get_color(1) | curses.A_BOLD) if curses.has_colors() else curses.A_STANDOUT
        safe_addstr(stdscr, 0, 0, " " * max_x, hdr_attr)
        safe_addstr(stdscr, 0, 2, "MLX-Commander  ::  Persistent Dataset Conversion Dashboard", hdr_attr)
        hint_str = "[F1: Help | F2: Open | F3: Output | F5: Convert | F10: Exit]" if max_x >= 88 else "[F1: Help | F10: Exit]"
        safe_addstr(stdscr, 0, max(2, max_x - len(hint_str) - 2), hint_str, hdr_attr)

        # ----------------------------------------------------
        # Dimensions & Coordinates (3-Tier Responsive Layout)
        # ----------------------------------------------------
        left_w = max(34, max_x // 2)
        right_w = max_x - left_w

        # Tier 1: Top configuration panels
        mapping_fields = state.get_mapping_fields_for_format()
        needed_top_h = max(12, 10 + len(mapping_fields))
        max_possible_top = max(9, max_y - 12)
        panel_h = min(needed_top_h, max_possible_top)

        # Tier 2: Middle visual mapping pipeline
        remaining_y = max(8, max_y - 1 - (panel_h + 1))
        vis_h = max(5, min(10, remaining_y // 2))
        vis_y = panel_h + 1

        # Tier 3: Bottom live preview panel
        preview_y = vis_y + vis_h
        preview_h = max(4, max_y - 1 - preview_y)

        is_left = (state.active_panel == ActivePanel.LEFT)
        is_right = (state.active_panel == ActivePanel.RIGHT)

        # ----------------------------------------------------
        # 2. Left Panel: Dataset Source & Schema
        # ----------------------------------------------------
        draw_box_panel(
            stdscr,
            1,
            0,
            panel_h,
            left_w,
            "Dataset & Schema",
            is_focused=is_left,
            subtitle="Tab 1",
        )

        # Source path display
        disp_path = state.dataset_path or "<no path set>"
        max_path_w = left_w - 10
        if len(disp_path) > max_path_w:
            disp_path = "…" + disp_path[-(max_path_w - 1):]
        safe_addstr(stdscr, 2, 2, "Path: ", (get_color(4) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
        safe_addstr(stdscr, 2, 8, disp_path[:max_path_w], (get_color(4) | curses.A_DIM) if curses.has_colors() else curses.A_DIM)

        # Action Buttons
        f2_focus = is_left and state.left_focus_idx == 0
        edit_focus = is_left and state.left_focus_idx == 1
        draw_button(stdscr, 3, 2, "Finder (F2)", is_focused=f2_focus)
        draw_button(stdscr, 3, 18, "Change Path", is_focused=edit_focus)
        safe_addstr(stdscr, 4, 2, "Tip: You can load multiple files (select multiple or use commas)"[:left_w - 4], (get_color(4) | curses.A_DIM) if curses.has_colors() else curses.A_DIM)

        # Dataset Stats
        if state.loaded_dataset:
            ds = state.loaded_dataset
            safe_addstr(stdscr, 5, 2, f"Rows: {ds.total_rows:,}  |  Splits: {len(ds.split_names)}", (get_color(4) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
            split_summary = ", ".join(f"{s}: {ds.split_counts.get(s, 0):,}" for s in ds.split_names[:3])
            safe_addstr(stdscr, 6, 2, f"Found: [{split_summary}]", (get_color(4) | curses.A_DIM) if curses.has_colors() else curses.A_DIM)
            cols = ds.columns
        else:
            safe_addstr(stdscr, 5, 2, "No dataset loaded.", (get_color(4) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
            safe_addstr(stdscr, 6, 2, "Click Finder (F2) or Change Path.", (get_color(4) | curses.A_DIM) if curses.has_colors() else curses.A_DIM)
            cols = []

        # Available Columns List
        safe_addstr(stdscr, 7, 2, f"Columns ({len(cols)}):", (get_color(4) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
        col_list_focus = is_left and state.left_focus_idx == 2
        col_list_start_y = 8
        col_list_rows = max(1, panel_h - 11)

        if cols:
            # Scroll window calculation
            if state.selected_column_idx < state.column_scroll_offset:
                state.column_scroll_offset = state.selected_column_idx
            elif state.selected_column_idx >= state.column_scroll_offset + col_list_rows:
                state.column_scroll_offset = state.selected_column_idx - col_list_rows + 1

            for r in range(col_list_rows):
                c_idx = state.column_scroll_offset + r
                row_y = col_list_start_y + r
                if c_idx < len(cols):
                    c_name = cols[c_idx]
                    is_col_sel = col_list_focus and (c_idx == state.selected_column_idx)
                    prefix = " ▶ " if is_col_sel else "   "
                    text = f"{prefix}{c_idx + 1}. {c_name}"[:left_w - 4]
                    if is_col_sel:
                        c_attr = (get_color(2) | curses.A_STANDOUT | curses.A_BOLD) if curses.has_colors() else curses.A_STANDOUT
                    elif col_list_focus:
                        c_attr = (get_color(2) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD
                    else:
                        c_attr = get_color(2) if curses.has_colors() else 0
                    safe_addstr(stdscr, row_y, 2, text, c_attr)

            # Sample value preview for currently highlighted column
            if 0 <= state.selected_column_idx < len(cols) and state.loaded_dataset and state.loaded_dataset.sample_records:
                col_name = cols[state.selected_column_idx]
                sample_val = str(state.loaded_dataset.sample_records[0].get(col_name, ""))
                if len(sample_val) > left_w - 12:
                    sample_val = sample_val[:left_w - 13] + "…"
                safe_addstr(stdscr, panel_h - 1, 2, "Sample: ", (get_color(4) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
                safe_addstr(stdscr, panel_h - 1, 10, sample_val, (get_color(4) | curses.A_DIM) if curses.has_colors() else curses.A_DIM)
        else:
            safe_addstr(stdscr, col_list_start_y, 4, "(Load dataset to view schema)", (get_color(4) | curses.A_DIM) if curses.has_colors() else curses.A_DIM)

        # ----------------------------------------------------
        # 3. Right Panel: MLX Format, Mappings & Splits
        # ----------------------------------------------------
        draw_box_panel(
            stdscr,
            1,
            left_w,
            panel_h,
            right_w,
            "MLX Format & Mappings",
            is_focused=is_right,
            subtitle="Tab 2",
        )

        mapping_fields = state.get_mapping_fields_for_format()
        total_right_fields = 1 + len(mapping_fields) + 7

        # Field 0: Target Format Radios
        fmt_focus = is_right and state.right_focus_idx == 0
        safe_addstr(stdscr, 2, left_w + 2, "Format:", (get_color(4) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
        draw_radio(
            stdscr,
            2,
            left_w + 10,
            "prompt_comp",
            is_checked=(state.target_format == MLXFormat.PROMPT_COMPLETION),
            is_focused=(fmt_focus and state.target_format == MLXFormat.PROMPT_COMPLETION),
        )
        draw_radio(
            stdscr,
            2,
            left_w + 26,
            "chat",
            is_checked=(state.target_format == MLXFormat.CHAT),
            is_focused=(fmt_focus and state.target_format == MLXFormat.CHAT),
        )
        draw_radio(
            stdscr,
            3,
            left_w + 10,
            "text",
            is_checked=(state.target_format == MLXFormat.TEXT),
            is_focused=(fmt_focus and state.target_format == MLXFormat.TEXT),
        )
        draw_radio(
            stdscr,
            3,
            left_w + 26,
            "dpo",
            is_checked=(state.target_format == MLXFormat.DPO),
            is_focused=(fmt_focus and state.target_format == MLXFormat.DPO),
        )

        # Fields 1..N: Column Mappings
        safe_addstr(stdscr, 4, left_w + 2, "Column Mappings [Press Enter to Pick]:", (get_color(4) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
        for idx, f_info in enumerate(mapping_fields):
            row_y = 5 + idx
            is_f_focused = is_right and (state.right_focus_idx == 1 + idx)
            draw_field(
                stdscr,
                row_y,
                left_w + 2,
                f_info["label"],
                str(f_info["current"] or "<None>"),
                is_focused=is_f_focused,
                val_width=min(22, right_w - len(f_info["label"]) - 8),
                has_dropdown=True,
            )

        # Next rows: Splits & Output
        splits_start_y = 5 + len(mapping_fields)
        split_counts = state.get_split_counts()

        train_focus = is_right and state.right_focus_idx == 1 + len(mapping_fields)
        valid_focus = is_right and state.right_focus_idx == 2 + len(mapping_fields)
        test_focus = is_right and state.right_focus_idx == 3 + len(mapping_fields)

        safe_addstr(stdscr, splits_start_y, left_w + 2, "Splits (%):", (get_color(4) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
        draw_field(stdscr, splits_start_y, left_w + 14, "Train", f"{state.train_pct:.0f}%", is_focused=train_focus, val_width=8)
        draw_field(stdscr, splits_start_y, left_w + 28, "Valid", f"{state.valid_pct:.0f}%", is_focused=valid_focus, val_width=8)
        draw_field(stdscr, splits_start_y, left_w + 42, "Test", f"{state.test_pct:.0f}%", is_focused=test_focus, val_width=8)

        if state.loaded_dataset:
            safe_addstr(
                stdscr,
                splits_start_y + 1,
                left_w + 4,
                f"Records: {split_counts['train']:,} train / {split_counts['valid']:,} valid / {split_counts['test']:,} test",
                (get_color(4) | curses.A_DIM) if curses.has_colors() else curses.A_DIM,
            )

        # Seed & Randomize
        seed_y = splits_start_y + 2
        seed_focus = is_right and state.right_focus_idx == 4 + len(mapping_fields)
        rand_focus = is_right and state.right_focus_idx == 5 + len(mapping_fields)
        draw_field(stdscr, seed_y, left_w + 2, "Seed", str(state.seed), is_focused=seed_focus, val_width=12)
        draw_button(stdscr, seed_y, left_w + 22, "Randomize (r)", is_focused=rand_focus)

        # Output Folder
        out_y = seed_y + 1
        out_focus = is_right and state.right_focus_idx == 6 + len(mapping_fields)
        draw_field(stdscr, out_y, left_w + 2, "Output", state.output_dir, is_focused=out_focus, val_width=min(24, right_w - 14))

        # Convert Action Button
        btn_y = panel_h - 2
        conv_focus = is_right and state.right_focus_idx == 7 + len(mapping_fields)
        draw_button(stdscr, btn_y, left_w + 4, "Convert Dataset (F5)", is_focused=conv_focus)

        # ----------------------------------------------------
        # 4. Middle Panel: Visual Column Mapping Pipeline
        # ----------------------------------------------------
        draw_mapping_pipeline_panel(
            stdscr,
            vis_y,
            0,
            vis_h,
            max_x,
            state,
        )

        # ----------------------------------------------------
        # 5. Bottom Panel: Live Converted Record Preview
        # ----------------------------------------------------
        draw_box_panel(
            stdscr,
            preview_y,
            0,
            preview_h,
            max_x,
            "Live Converted Record Preview (MLX JSONL Format)",
            is_focused=False,
            subtitle=f"{state.target_format.value.upper()}",
        )

        if state.preview_error:
            safe_addstr(stdscr, preview_y + 1, 3, f"[!] {state.preview_error}", get_color(5) | curses.A_BOLD)
            safe_addstr(stdscr, preview_y + 2, 3, "Adjust column mappings in the Right Panel (Tab 2) to preview records.", (get_color(4) | curses.A_DIM) if curses.has_colors() else curses.A_DIM)
        elif not state.preview_cache:
            safe_addstr(stdscr, preview_y + 1, 3, "(No records to preview)", (get_color(4) | curses.A_DIM) if curses.has_colors() else curses.A_DIM)
        else:
            avail_w = max(20, max_x - 10)
            avail_rows = max(1, preview_h - 2)

            all_wrapped: List[List[str]] = []
            for line in state.preview_cache:
                wrapped = textwrap.wrap(
                    line,
                    width=avail_w,
                    break_long_words=True,
                    break_on_hyphens=False,
                )
                all_wrapped.append(wrapped if wrapped else [""])

            # Distribute available rows among all examples so all 3 examples are displayed
            n_examples = len(all_wrapped)
            lengths = [len(w) for w in all_wrapped]
            allocated = [0] * n_examples
            rem_rows = avail_rows

            # Guarantee at least 1 row per example if rows allow
            for i in range(n_examples):
                if rem_rows > 0:
                    allocated[i] = 1
                    rem_rows -= 1

            # Distribute remaining rows proportionally to longer examples
            while rem_rows > 0:
                candidates = [i for i in range(n_examples) if allocated[i] < lengths[i]]
                if not candidates:
                    break
                for i in candidates:
                    if rem_rows > 0 and allocated[i] < lengths[i]:
                        allocated[i] += 1
                        rem_rows -= 1

            # Render wrapped lines
            curr_row = preview_y + 1
            for i, wrapped_lines in enumerate(all_wrapped):
                num_lines = allocated[i]
                for line_idx in range(num_lines):
                    if curr_row >= preview_y + preview_h - 1:
                        break
                    line_text = wrapped_lines[line_idx]
                    # If this is the last line of a truncated example, append ellipsis
                    if line_idx == num_lines - 1 and num_lines < len(wrapped_lines):
                        if len(line_text) > avail_w - 5:
                            line_text = line_text[:avail_w - 5].rstrip() + " ...}"
                        else:
                            line_text = line_text + " ...}"

                    if line_idx == 0:
                        safe_addstr(stdscr, curr_row, 3, f"{i + 1}: ", (get_color(4) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
                        safe_addstr(stdscr, curr_row, 6, line_text, (get_color(4) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
                    else:
                        # Continuation line indented
                        safe_addstr(stdscr, curr_row, 6, line_text, get_color(4) if curses.has_colors() else 0)
                    curr_row += 1

        # ----------------------------------------------------
        # 6. Bottom Status / Hotkey Bar
        # ----------------------------------------------------
        footer_y = max_y - 1
        safe_addstr(stdscr, footer_y, 0, " " * max_x, hdr_attr)

        bar_shortcuts = "[Tab] Switch  [↑/↓] Move  [Enter] Edit  [F2] Open  [F3] Output  [F5] Convert" if max_x >= 92 else "[Tab] Switch  [F2] Open  [F5] Convert"
        shortcuts_x = max(10, max_x - len(bar_shortcuts) - 2)
        avail_status = max(10, shortcuts_x - 4)

        status_prefix = "[OK] " if not state.status_is_error else "[ERR] "
        status_text = f"{status_prefix}{state.status_message}"[:avail_status]
        safe_addstr(stdscr, footer_y, 2, status_text, hdr_attr | curses.A_BOLD)
        safe_addstr(stdscr, footer_y, shortcuts_x, bar_shortcuts, hdr_attr)

        stdscr.refresh()

        # ----------------------------------------------------
        # Input & Key Event Handling
        # ----------------------------------------------------
        key = stdscr.getch()

        if key == curses.KEY_RESIZE:
            stdscr.clear()
            continue

        elif key in (27, ord("q"), ord("Q"), curses.KEY_F10):
            break

        try:
            if key in (ord("?"), curses.KEY_F1):
                show_help_dialog(stdscr)

            elif key in (curses.KEY_F2, 15):  # F2 or Ctrl+O
                curses.def_prog_mode()
                curses.endwin()
                from mlx_commander.gui_picker import pick_dataset_gui
                chosen = pick_dataset_gui(default_dir=state.dataset_path or os.getcwd())
                curses.reset_prog_mode()
                stdscr.refresh()
                if chosen:
                    if not state.load_dataset(chosen):
                        show_error_dialog(stdscr, "Dataset Loading Failed", state.status_message)

            elif key in (curses.KEY_F3,):  # F3: Destination Folder
                curses.def_prog_mode()
                curses.endwin()
                from mlx_commander.gui_picker import is_macos, pick_folder_gui
                if is_macos():
                    start_dir = state.output_dir or (str(state.loaded_dataset.default_output_dir) if state.loaded_dataset else os.getcwd())
                    chosen = pick_folder_gui("Select Destination Folder", default_dir=start_dir)
                else:
                    chosen = None
                curses.reset_prog_mode()
                stdscr.refresh()
                if chosen:
                    state.output_dir = chosen.strip()
                    default_out = str(state.loaded_dataset.default_output_dir) if state.loaded_dataset else ""
                    state.has_custom_output_dir = (state.output_dir != default_out)
                    state.status_message = f"Output folder set to: {state.output_dir}"
                    state.status_is_error = False
                elif not is_macos():
                    default_out = str(state.loaded_dataset.default_output_dir) if state.loaded_dataset else state.output_dir
                    val = show_text_edit_dialog(
                        stdscr,
                        "Output Directory",
                        "Enter folder to save MLX JSONL datasets:",
                        default_val=state.output_dir or default_out,
                    )
                    if val:
                        state.output_dir = val.strip()
                        state.has_custom_output_dir = (state.output_dir != default_out)
                        state.status_message = f"Output folder set to: {state.output_dir}"
                        state.status_is_error = False

            elif key == curses.KEY_F5:
                res = execute_conversion(stdscr, state)
                if res is not None:
                    conversion_result = res

            elif key in (ord("r"), ord("R")):
                state.randomize_seed()

            elif key in (9, curses.KEY_BTAB):  # Tab / Shift-Tab
                state.active_panel = ActivePanel.RIGHT if state.active_panel == ActivePanel.LEFT else ActivePanel.LEFT

            # Navigation in Left Panel
            elif is_left:
                if key in (curses.KEY_UP, ord("k")):
                    if state.left_focus_idx == 2 and state.selected_column_idx > 0:
                        state.selected_column_idx -= 1
                    else:
                        state.left_focus_idx = max(0, state.left_focus_idx - 1)
                elif key in (curses.KEY_DOWN, ord("j")):
                    if state.left_focus_idx == 2:
                        if state.loaded_dataset and state.selected_column_idx < len(state.loaded_dataset.columns) - 1:
                            state.selected_column_idx += 1
                    else:
                        state.left_focus_idx = min(2, state.left_focus_idx + 1)
                elif key in (10, 13, curses.KEY_ENTER, 32):  # Enter or Space
                    if state.left_focus_idx == 0:  # Finder button
                        curses.def_prog_mode()
                        curses.endwin()
                        from mlx_commander.gui_picker import pick_dataset_gui
                        chosen = pick_dataset_gui(default_dir=state.dataset_path or os.getcwd())
                        curses.reset_prog_mode()
                        stdscr.refresh()
                        if chosen:
                            if not state.load_dataset(chosen):
                                show_error_dialog(stdscr, "Dataset Loading Failed", state.status_message)
                    elif state.left_focus_idx == 1:  # Edit path
                        new_path = show_text_edit_dialog(
                            stdscr,
                            "Change Dataset Path",
                            "Enter path(s) to local HF dataset (comma/newline separated for multiple):",
                            default_val=state.dataset_path or os.getcwd(),
                        )
                        if new_path:
                            if not state.load_dataset(new_path):
                                show_error_dialog(stdscr, "Dataset Loading Failed", state.status_message)

            # Navigation in Right Panel
            elif is_right:
                if key in (curses.KEY_UP, ord("k")):
                    state.right_focus_idx = (state.right_focus_idx - 1) % total_right_fields
                elif key in (curses.KEY_DOWN, ord("j")):
                    state.right_focus_idx = (state.right_focus_idx + 1) % total_right_fields
                elif key in (curses.KEY_LEFT, ord("h")):
                    if state.right_focus_idx == 0:
                        # Cycle format backwards
                        curr_i = formats_list.index(state.target_format)
                        state.set_format(formats_list[(curr_i - 1) % len(formats_list)])
                elif key in (curses.KEY_RIGHT, ord("l")):
                    if state.right_focus_idx == 0:
                        # Cycle format forwards
                        curr_i = formats_list.index(state.target_format)
                        state.set_format(formats_list[(curr_i + 1) % len(formats_list)])

                elif key in (10, 13, curses.KEY_ENTER, 32):  # Enter or Space
                    idx = state.right_focus_idx
                    if idx == 0:
                        # Toggle next format
                        curr_i = formats_list.index(state.target_format)
                        state.set_format(formats_list[(curr_i + 1) % len(formats_list)])

                    elif 1 <= idx <= len(mapping_fields):
                        # Column mapping picker
                        f_info = mapping_fields[idx - 1]
                        cols = state.loaded_dataset.columns if state.loaded_dataset else []
                        chosen = show_column_picker_dialog(
                            stdscr,
                            f"Select Column for '{f_info['label']}'",
                            cols,
                            current_val=f_info["current"],
                            allow_none=True,
                        )
                        state.set_mapping_field(f_info["key"], chosen)

                    elif idx == 1 + len(mapping_fields):  # Train %
                        val = show_text_edit_dialog(stdscr, "Train Split %", "Enter Train percentage (0-100):", f"{state.train_pct:.0f}", is_number=True)
                        if val:
                            try:
                                state.train_pct = float(val)
                            except ValueError:
                                pass

                    elif idx == 2 + len(mapping_fields):  # Valid %
                        val = show_text_edit_dialog(stdscr, "Valid Split %", "Enter Valid percentage (0-100):", f"{state.valid_pct:.0f}", is_number=True)
                        if val:
                            try:
                                state.valid_pct = float(val)
                            except ValueError:
                                pass

                    elif idx == 3 + len(mapping_fields):  # Test %
                        val = show_text_edit_dialog(stdscr, "Test Split %", "Enter Test percentage (0-100):", f"{state.test_pct:.0f}", is_number=True)
                        if val:
                            try:
                                state.test_pct = float(val)
                            except ValueError:
                                pass

                    elif idx == 4 + len(mapping_fields):  # Seed
                        val = show_text_edit_dialog(stdscr, "Random Seed", "Enter random seed integer:", str(state.seed), is_number=True)
                        if val:
                            try:
                                state.seed = int(val)
                            except ValueError:
                                pass

                    elif idx == 5 + len(mapping_fields):  # Randomize button
                        state.randomize_seed()

                    elif idx == 6 + len(mapping_fields):  # Output Dir
                        default_out = str(state.loaded_dataset.default_output_dir) if state.loaded_dataset else state.output_dir
                        chosen = show_output_destination_dialog(
                            stdscr,
                            current_output=state.output_dir,
                            default_output=default_out,
                        )
                        if chosen is not None:
                            state.output_dir = chosen.strip()
                            state.has_custom_output_dir = (state.output_dir != default_out)
                            state.status_message = f"Output folder set to: {state.output_dir}"
                            state.status_is_error = False

                    elif idx == 7 + len(mapping_fields):  # Convert button
                        res = execute_conversion(stdscr, state)
                        if res is not None:
                            conversion_result = res
        except curses.error:
            pass
        except Exception as e:
            state.status_message = f"Error: {e}"
            state.status_is_error = True
            show_error_dialog(stdscr, "Error Encountered", str(e))

    return conversion_result


def launch_tui(
    default_dataset_path: Optional[str] = None,
    initial_state: Optional[CommanderState] = None,
    prefill: Optional[Dict[str, Any]] = None,
) -> Optional[ConversionResult]:
    """Launch the MLX-Commander full-screen curses dashboard with optional prefill."""
    configure_escdelay(25)
    try:
        return curses.wrapper(run_commander_tui, default_dataset_path, initial_state, prefill)
    except KeyboardInterrupt:
        return None

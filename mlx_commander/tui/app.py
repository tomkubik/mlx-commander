"""
MLX Commander: Persistent Full-Screen Curses TUI Dashboard.
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
from typing import Any, Dict, List, Optional, Tuple

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
from mlx_commander.lora import (
    FINE_TUNE_TYPES,
    OPTIMIZERS,
    POPULAR_MLX_MODELS,
    LoraRunConfig,
    QueueManager,
)
from mlx_commander.terminal_spawner import is_macos, spawn_lora_queue_terminal
from mlx_commander.tui.state import ActivePanel, CommanderState, ThemeMode
from mlx_commander.tui.widgets import (
    COLOR_BANNER,
    COLOR_BORDER_JOINTS,
    COLOR_SUCCESS,
    COLOR_NORMAL_TEXT,
    COLOR_ERROR,
    COLOR_TITLE_ACCENT,
    COLOR_LABEL_GRAY,
    COLOR_INPUT_NORMAL,
    COLOR_INPUT_FOCUSED,
    COLOR_FN_NUMBER,
    COLOR_FN_LABEL,
    COLOR_PANEL_BG,
    configure_escdelay,
    draw_box_panel,
    draw_button,
    draw_field,
    draw_footer,
    draw_header,
    draw_mapping_pipeline_panel,
    draw_queue_table,
    draw_radio,
    get_color,
    safe_addstr,
    show_choice_dialog,
    show_column_picker_dialog,
    show_error_dialog,
    show_help_dialog,
    show_message_dialog,
    show_missing_dependency_dialog,
    show_model_picker_dialog,
    show_output_destination_dialog,
    show_results_dialog,
    show_text_edit_dialog,
)


def init_nc_palette() -> Tuple[int, int, int, int, int, int, int]:
    """
    Configure and return Norton Commander VGA color indices:
      - Background Blue: #0000AA (Classic VGA Blue)
      - Highlight / Cyan: #00AAAA (Cyan)
      - Main Text / White: #FFFFFF (Bright White)
      - Label / Light Gray: #AAAAAA (Light Gray)
      - Cursor / Yellow: #FFFF55 (Bright Yellow)
      - Prompt / Black: #000000 (Black)
      - Inactive Input Ice Blue: #55FFFF (Bright Cyan / Ice Blue)
    """
    if curses.can_change_color():
        # Exact RGB on 0..1000 curses scale
        curses.init_color(20, 0, 0, 666)         # #0000AA (Classic VGA Blue)
        curses.init_color(21, 0, 666, 666)       # #00AAAA (Cyan)
        curses.init_color(22, 1000, 1000, 1000)  # #FFFFFF (Bright White)
        curses.init_color(23, 666, 666, 666)     # #AAAAAA (Light Gray)
        curses.init_color(24, 1000, 1000, 333)   # #FFFF55 (Bright Yellow)
        curses.init_color(25, 0, 0, 0)           # #000000 (Black)
        curses.init_color(26, 333, 1000, 1000)   # #55FFFF (Ice Blue / Bright Cyan)
        return 20, 21, 22, 23, 24, 25, 26
    elif curses.COLORS >= 256:
        # Closest 256-color palette slots:
        # 19: #0000af (Blue), 37: #00afaf (Cyan), 15: #ffffff (White),
        # 248: #a8a8a8 (Light Gray), 227: #ffff5f (Yellow), 0: #000000 (Black), 51: #00ffff (Ice Blue)
        return 19, 37, 15, 248, 227, 0, 51
    else:
        # 16-color standard ANSI fallback
        return (
            curses.COLOR_BLUE,
            curses.COLOR_CYAN,
            curses.COLOR_WHITE,
            curses.COLOR_WHITE,
            curses.COLOR_YELLOW,
            curses.COLOR_BLACK,
            curses.COLOR_CYAN,
        )


def init_colors(theme_mode: Any = "modern") -> None:
    """Initialize curses color pairs for Modern or Norton Commander color schemes."""
    if not curses.has_colors():
        return
    curses.start_color()
    is_norton = ThemeMode.is_norton(theme_mode)

    if is_norton:
        c_blue, c_cyan, c_white, c_gray, c_yellow, c_black, c_ice_blue = init_nc_palette()
        # Norton Commander Classic EGA/VGA Palette:
        # Pair 1: Top Header Banner (Prompt / Black on Highlight / Cyan)
        curses.init_pair(COLOR_BANNER, c_black, c_cyan)
        # Pair 2: Highlight / Cyan Joints & Borders on Background Blue
        curses.init_pair(COLOR_BORDER_JOINTS, c_cyan, c_blue)
        # Pair 3: Success / Checked on Background Blue
        curses.init_pair(COLOR_SUCCESS, curses.COLOR_GREEN, c_blue)
        # Pair 4: Main Text / White (Standard file names and UI text) on Background Blue
        curses.init_pair(COLOR_NORMAL_TEXT, c_white, c_blue)
        # Pair 5: Error / Alerts (Main Text White on Red)
        curses.init_pair(COLOR_ERROR, c_white, curses.COLOR_RED)
        # Pair 6: Cursor / Yellow (Panel titles & accents) on Background Blue
        curses.init_pair(COLOR_TITLE_ACCENT, c_yellow, c_blue)
        # Pair 7: Label / Light Gray (Inactive elements, background text) on Background Blue
        curses.init_pair(COLOR_LABEL_GRAY, c_gray, c_blue)
        # Pair 8: User Inputs / Inactive Fields (Ice Blue on Deep Blue background)
        curses.init_pair(COLOR_INPUT_NORMAL, c_ice_blue, c_blue)
        # Pair 9: Active / Focused Field (Prompt / Black on Highlight / Cyan background)
        curses.init_pair(COLOR_INPUT_FOCUSED, c_black, c_cyan)
        # Pair 10: Function Key Number (White on Black)
        curses.init_pair(COLOR_FN_NUMBER, c_white, c_black)
        # Pair 11: Function Key Label (Black on Cyan)
        curses.init_pair(COLOR_FN_LABEL, c_black, c_cyan)
        # Pair 12: Background Blue (Panel/Window Fill: #0000AA)
        curses.init_pair(COLOR_PANEL_BG, c_white, c_blue)
    else:
        # Modern Dark Theme
        curses.use_default_colors()
        curses.init_pair(COLOR_BANNER, curses.COLOR_WHITE, curses.COLOR_BLUE)
        curses.init_pair(COLOR_BORDER_JOINTS, curses.COLOR_CYAN, -1)
        curses.init_pair(COLOR_SUCCESS, curses.COLOR_GREEN, -1)
        curses.init_pair(COLOR_NORMAL_TEXT, curses.COLOR_WHITE, -1)
        curses.init_pair(COLOR_ERROR, curses.COLOR_RED, -1)
        curses.init_pair(COLOR_TITLE_ACCENT, curses.COLOR_WHITE, -1)
        curses.init_pair(COLOR_LABEL_GRAY, curses.COLOR_WHITE, -1)
        curses.init_pair(COLOR_INPUT_NORMAL, curses.COLOR_CYAN, -1)
        curses.init_pair(COLOR_INPUT_FOCUSED, curses.COLOR_BLACK, curses.COLOR_CYAN)
        curses.init_pair(COLOR_FN_NUMBER, curses.COLOR_WHITE, -1)
        curses.init_pair(COLOR_FN_LABEL, curses.COLOR_CYAN, -1)
        curses.init_pair(COLOR_PANEL_BG, curses.COLOR_WHITE, -1)



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


def execute_lora_queue_action(stdscr: curses.window, state: CommanderState) -> None:
    """Execute queued LoRA fine-tuning runs sequentially via spawned macOS Terminal window."""
    if not state.queue_manager or not state.queue_manager.runs:
        show_error_dialog(
            stdscr,
            "Queue Empty",
            "The LoRA queue is empty. Configure parameters and press [F6] to add runs before executing.",
        )
        return

    pending = state.queue_manager.get_pending_runs()
    if not pending:
        show_error_dialog(
            stdscr,
            "No Pending Runs",
            "All runs in the queue have already completed. Add a new run [F6] or clone an existing run [c].",
        )
        return

    state.queue_manager.save()

    if is_macos():
        spawned = spawn_lora_queue_terminal(str(state.queue_manager.queue_dir))
        if spawned:
            show_message_dialog(
                stdscr,
                "LoRA Queue Launched",
                [
                    f"Sequential execution of {len(pending)} pending run(s) launched in macOS Terminal.",
                    "",
                    "• Runs execute sequentially to protect Apple Silicon Unified Memory.",
                    "• Real-time loss, throughput, and progress stream in the Terminal window.",
                    "• You may safely close MLX Commander without interrupting training.",
                ],
            )
            state.status_message = f"Queue running in Terminal window ({len(pending)} pending runs)."
            state.status_is_error = False
        else:
            show_error_dialog(stdscr, "Execution Error", "Failed to spawn external macOS Terminal.app window.")
    else:
        cmd = f"python3 -m mlx_commander --run-queue {state.queue_manager.queue_dir}"
        show_message_dialog(
            stdscr,
            "Execute LoRA Queue",
            [
                f"Queue saved to: {state.queue_manager.queue_dir}",
                "Run the following command in another terminal window:",
                "",
                cmd,
            ],
        )
        state.status_message = f"Queue saved. Run: {cmd}"




def _draw_mode1_dashboard(
    stdscr: curses.window,
    state: CommanderState,
    max_y: int,
    max_x: int,
    left_w: int,
    right_w: int,
) -> None:
    """Draw Mode 1: Dataset Converter dual-panel view with pipeline and preview."""
    mapping_fields = state.get_mapping_fields_for_format()
    min_top_h = 14 + len(mapping_fields)
    needed_top_h = max(16, min_top_h)
    max_possible_top = max(min_top_h, max_y - 10)
    panel_h = min(needed_top_h, max_possible_top)

    remaining_y = max(8, max_y - 1 - (panel_h + 1))
    vis_h = max(4, min(10, remaining_y // 2))
    vis_y = panel_h + 1

    preview_y = vis_y + vis_h
    preview_h = max(4, max_y - 1 - preview_y)

    is_left = (state.active_panel == ActivePanel.LEFT)
    is_right = (state.active_panel == ActivePanel.RIGHT)

    # 1. Left Panel: Dataset Source & Schema
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

    disp_path = state.dataset_path or "<no path set>"
    tab1_right_edge = left_w - 3
    max_path_w = max(10, left_w - 12)
    if len(disp_path) > max_path_w:
        disp_path = "…" + disp_path[-(max_path_w - 1):]
    path_x = max(8, tab1_right_edge - len(disp_path) + 1)
    safe_addstr(stdscr, 2, 2, "Path: ", (get_color(COLOR_LABEL_GRAY) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
    safe_addstr(stdscr, 2, path_x, disp_path, get_color(COLOR_NORMAL_TEXT) if curses.has_colors() else curses.A_DIM)

    f2_focus = is_left and state.left_focus_idx == 0
    edit_focus = is_left and state.left_focus_idx == 1
    draw_button(stdscr, 3, 2, "Finder (F2)", is_focused=f2_focus)
    draw_button(stdscr, 4, 2, "Change Path", is_focused=edit_focus)
    safe_addstr(stdscr, 5, 2, "Tip: You can load multiple files (select multiple or use commas)"[:left_w - 4], get_color(COLOR_LABEL_GRAY) if curses.has_colors() else curses.A_DIM)

    if state.loaded_dataset:
        ds = state.loaded_dataset
        safe_addstr(stdscr, 6, 2, "Rows: ", (get_color(COLOR_LABEL_GRAY) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
        safe_addstr(stdscr, 6, 8, f"{ds.total_rows:,}  ", get_color(COLOR_NORMAL_TEXT) if curses.has_colors() else 0)
        splits_x = 8 + len(f"{ds.total_rows:,}  ")
        safe_addstr(stdscr, 6, splits_x, "Splits: ", (get_color(COLOR_LABEL_GRAY) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
        safe_addstr(stdscr, 6, splits_x + 8, f"{len(ds.split_names)}", get_color(COLOR_NORMAL_TEXT) if curses.has_colors() else 0)
        split_summary = ", ".join(f"{s}: {ds.split_counts.get(s, 0):,}" for s in ds.split_names[:3])
        safe_addstr(stdscr, 7, 2, f"Found: [{split_summary}]", get_color(COLOR_LABEL_GRAY) if curses.has_colors() else curses.A_DIM)
        cols = ds.columns
    else:
        safe_addstr(stdscr, 6, 2, "No dataset loaded.", (get_color(COLOR_LABEL_GRAY) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
        safe_addstr(stdscr, 7, 2, "Use Finder (F2) or Change Path.", get_color(COLOR_LABEL_GRAY) if curses.has_colors() else curses.A_DIM)
        cols = []

    safe_addstr(stdscr, 8, 2, f"Columns ({len(cols)}):", (get_color(COLOR_LABEL_GRAY) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
    col_list_focus = is_left and state.left_focus_idx == 2
    col_list_start_y = 9
    col_list_rows = max(1, panel_h - 11)

    if cols:
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
                    c_attr = (get_color(COLOR_INPUT_FOCUSED) | curses.A_BOLD) if curses.has_colors() else curses.A_STANDOUT
                elif col_list_focus:
                    c_attr = (get_color(COLOR_BORDER_JOINTS) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD
                else:
                    c_attr = get_color(COLOR_NORMAL_TEXT) if curses.has_colors() else 0
                safe_addstr(stdscr, row_y, 2, text, c_attr)

        if 0 <= state.selected_column_idx < len(cols) and state.loaded_dataset and state.loaded_dataset.sample_records:
            col_name = cols[state.selected_column_idx]
            sample_val = str(state.loaded_dataset.sample_records[0].get(col_name, ""))
            if len(sample_val) > left_w - 12:
                sample_val = sample_val[:left_w - 13] + "…"
            safe_addstr(stdscr, panel_h - 1, 2, "Sample: ", (get_color(COLOR_LABEL_GRAY) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
            safe_addstr(stdscr, panel_h - 1, 10, sample_val, get_color(COLOR_NORMAL_TEXT) if curses.has_colors() else curses.A_DIM)
    else:
        safe_addstr(stdscr, col_list_start_y, 4, "(Load dataset to view schema)", get_color(COLOR_LABEL_GRAY) if curses.has_colors() else curses.A_DIM)

    # 2. Right Panel: MLX Format, Mappings & Splits
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

    total_right_fields = 11 + len(mapping_fields)
    if state.right_focus_idx >= total_right_fields:
        state.right_focus_idx = total_right_fields - 1

    safe_addstr(stdscr, 2, left_w + 2, "Format (MLX Target):", (get_color(COLOR_LABEL_GRAY) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
    format_options = [
        (MLXFormat.PROMPT_COMPLETION, "prompt_comp (Q&A)", "prompt_comp"),
        (MLXFormat.CHAT, "chat (Dialogue / Messages)", "chat"),
        (MLXFormat.TEXT, "text (Causal LM)", "text"),
        (MLXFormat.DPO, "dpo (Preference)", "dpo"),
    ]
    for f_i, (fmt_val, fmt_lbl, fmt_short) in enumerate(format_options):
        row_y = 3 + f_i
        lbl_text = fmt_short if right_w < 35 else fmt_lbl
        is_chk = (state.target_format == fmt_val)
        is_foc = (is_right and state.right_focus_idx == f_i)
        draw_radio(stdscr, row_y, left_w + 4, lbl_text, is_checked=is_chk, is_focused=is_foc)

    right_edge = left_w + right_w - 3
    safe_addstr(stdscr, 7, left_w + 2, "Column Mappings [Press Enter]:", (get_color(COLOR_LABEL_GRAY) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
    mapping_val_w = min(22, max(12, right_w - 24))

    for idx, f_info in enumerate(mapping_fields):
        row_y = 8 + idx
        is_f_focused = is_right and (state.right_focus_idx == 4 + idx)
        draw_field(
            stdscr,
            row_y,
            left_w + 2,
            f_info["label"],
            str(f_info["current"] or "<None>"),
            is_focused=is_f_focused,
            val_width=mapping_val_w,
            has_dropdown=True,
            right_edge=right_edge,
        )

    splits_start_y = 8 + len(mapping_fields)
    train_focus = is_right and state.right_focus_idx == 4 + len(mapping_fields)
    valid_focus = is_right and state.right_focus_idx == 5 + len(mapping_fields)
    test_focus = is_right and state.right_focus_idx == 6 + len(mapping_fields)

    safe_addstr(
        stdscr,
        splits_start_y,
        left_w + 2,
        "Dataset split:",
        (get_color(COLOR_LABEL_GRAY) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD,
    )
    split_label_x = min(left_w + 18, right_edge - 16)
    white_unbold = get_color(COLOR_NORMAL_TEXT) if curses.has_colors() else 0

    draw_field(stdscr, splits_start_y, split_label_x, "Train", f"{state.train_pct:.0f}%", is_focused=train_focus, val_width=6, right_edge=right_edge, lbl_attr=white_unbold)
    draw_field(stdscr, splits_start_y + 1, split_label_x, "Valid", f"{state.valid_pct:.0f}%", is_focused=valid_focus, val_width=6, right_edge=right_edge, lbl_attr=white_unbold)
    draw_field(stdscr, splits_start_y + 2, split_label_x, "Test", f"{state.test_pct:.0f}%", is_focused=test_focus, val_width=6, right_edge=right_edge, lbl_attr=white_unbold)

    seed_y = splits_start_y + 3
    seed_focus = is_right and state.right_focus_idx == 7 + len(mapping_fields)
    rand_focus = is_right and state.right_focus_idx == 8 + len(mapping_fields)

    btn_w = 17
    seed_box_w = 10
    btn_x = max(left_w + 18, right_edge - btn_w + 1)
    seed_field_x = max(left_w + 8, btn_x - 2 - seed_box_w)

    draw_field(stdscr, seed_y, left_w + 2, "Seed", str(state.seed), is_focused=seed_focus, val_width=8, field_x=seed_field_x)
    draw_button(stdscr, seed_y, btn_x, "Randomize (r)", is_focused=rand_focus)

    out_y = seed_y + 1
    out_focus = is_right and state.right_focus_idx == 9 + len(mapping_fields)
    draw_field(
        stdscr,
        out_y,
        left_w + 2,
        "Output",
        state.output_dir,
        is_focused=out_focus,
        val_width=mapping_val_w,
        has_dropdown=True,
        right_edge=right_edge,
    )

    conv_focus = is_right and state.right_focus_idx == 10 + len(mapping_fields)
    btn_y = out_y + 1
    draw_button(stdscr, btn_y, left_w + 4, "Convert Dataset (F5)", is_focused=conv_focus)

    # 3. Middle Panel: Visual Column Mapping Pipeline
    draw_mapping_pipeline_panel(
        stdscr,
        vis_y,
        0,
        vis_h,
        max_x,
        state,
    )

    # 4. Bottom Panel: Live Converted Record Preview
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

    if not state.loaded_dataset:
        safe_addstr(stdscr, preview_y + 1, 3, "(No dataset selected)", (get_color(COLOR_LABEL_GRAY) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
        safe_addstr(stdscr, preview_y + 2, 3, "Use Finder (F2) or Change Path in Tab 1 to select a dataset.", get_color(COLOR_LABEL_GRAY) if curses.has_colors() else curses.A_DIM)
    elif state.preview_error:
        safe_addstr(stdscr, preview_y + 1, 3, f"[!] {state.preview_error}", get_color(5) | curses.A_BOLD)
        safe_addstr(stdscr, preview_y + 2, 3, "Adjust column mappings in the Right Panel (Tab 2) to preview records.", get_color(COLOR_LABEL_GRAY) if curses.has_colors() else curses.A_DIM)
    elif not state.preview_cache:
        safe_addstr(stdscr, preview_y + 1, 3, "(No records to preview)", get_color(COLOR_LABEL_GRAY) if curses.has_colors() else curses.A_DIM)
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

        n_examples = len(all_wrapped)
        lengths = [len(w) for w in all_wrapped]
        allocated = [0] * n_examples
        rem_rows = avail_rows

        for i in range(n_examples):
            if rem_rows > 0:
                allocated[i] = 1
                rem_rows -= 1

        while rem_rows > 0:
            candidates = [i for i in range(n_examples) if allocated[i] < lengths[i]]
            if not candidates:
                break
            for i in candidates:
                if rem_rows > 0 and allocated[i] < lengths[i]:
                    allocated[i] += 1
                    rem_rows -= 1

        curr_row = preview_y + 1
        for i, wrapped_lines in enumerate(all_wrapped):
            num_lines = allocated[i]
            for line_idx in range(num_lines):
                if curr_row >= preview_y + preview_h - 1:
                    break
                line_text = wrapped_lines[line_idx]
                if line_idx == num_lines - 1 and num_lines < len(wrapped_lines):
                    if len(line_text) > avail_w - 5:
                        line_text = line_text[:avail_w - 5].rstrip() + " ...}"
                    else:
                        line_text = line_text + " ...}"

                if line_idx == 0:
                    safe_addstr(stdscr, curr_row, 3, f"{i + 1}: ", (get_color(COLOR_LABEL_GRAY) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
                    safe_addstr(stdscr, curr_row, 6, line_text, (get_color(COLOR_NORMAL_TEXT) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
                else:
                    safe_addstr(stdscr, curr_row, 6, line_text, get_color(COLOR_NORMAL_TEXT) if curses.has_colors() else 0)
                curr_row += 1


def _draw_mode2_dashboard(
    stdscr: curses.window,
    state: CommanderState,
    max_y: int,
    max_x: int,
    left_w: int,
    right_w: int,
) -> None:
    """Draw Mode 2: Apple MLX LoRA Fine-Tuning Dashboard & Queue."""
    panel_h = max(13, min(15, max_y - 12))
    vis_y = panel_h + 1
    remaining_y = max(6, max_y - 1 - (panel_h + 1))
    vis_h = max(5, min(9, remaining_y // 2))
    preview_y = vis_y + vis_h
    preview_h = max(4, max_y - 1 - preview_y)

    is_lora_left = (state.lora_active_panel == "left")
    is_lora_right = (state.lora_active_panel == "right")
    is_lora_queue = (state.lora_active_panel == "queue")

    # 1. Left Panel: Model & Dataset Selector
    draw_box_panel(
        stdscr,
        1,
        0,
        panel_h,
        left_w,
        "Model & Dataset Selector (LoRA Base)",
        is_focused=is_lora_left,
        subtitle="Step 1: Setup",
    )

    # Field 0: Base Model
    m_focus = is_lora_left and state.lora_left_focus_idx == 0
    model_disp = state.lora_config.model
    if len(model_disp) > left_w - 14:
        model_disp = "…" + model_disp[-(left_w - 15):]
    draw_field(stdscr, 2, 2, "Base Model", model_disp, is_focused=m_focus, val_width=left_w - 16, has_dropdown=True)

    # Field 1: Dataset Directory
    d_focus = is_lora_left and state.lora_left_focus_idx == 1
    data_disp = state.lora_config.data or "<no dataset path>"
    if len(data_disp) > left_w - 14:
        data_disp = "…" + data_disp[-(left_w - 15):]
    draw_field(stdscr, 4, 2, "Dataset", data_disp, is_focused=d_focus, val_width=left_w - 16, has_dropdown=True)

    # Field 2: Finder button
    f2_focus = is_lora_left and state.lora_left_focus_idx == 2
    draw_button(stdscr, 5, 2, "Finder (F2)", is_focused=f2_focus)

    # Field 3: Technique
    type_focus = is_lora_left and state.lora_left_focus_idx == 3
    draw_field(stdscr, 7, 2, "Method", state.lora_config.fine_tune_type.upper(), is_focused=type_focus, val_width=10, has_dropdown=True)

    # Field 4: Optimizer
    opt_focus = is_lora_left and state.lora_left_focus_idx == 4
    draw_field(stdscr, 8, 2, "Optimizer", state.lora_config.optimizer, is_focused=opt_focus, val_width=12, has_dropdown=True)

    # Field 5: Mode (Train / Test)
    mode_focus = is_lora_left and state.lora_left_focus_idx == 5
    mode_str = f"Train: {'[X]' if state.lora_config.train else '[ ]'}  Test: {'[X]' if state.lora_config.test else '[ ]'}"
    draw_field(stdscr, 9, 2, "Mode", mode_str, is_focused=mode_focus, val_width=20)

    # Field 6: Run Name
    name_focus = is_lora_left and state.lora_left_focus_idx == 6
    name_disp = state.lora_config.name or "<auto>"
    if len(name_disp) > left_w - 14:
        name_disp = "…" + name_disp[-(left_w - 15):]
    draw_field(stdscr, 10, 2, "Run Name", name_disp, is_focused=name_focus, val_width=left_w - 16)

    # 2. Right Panel: Hyperparameters & Unified Memory Estimator
    draw_box_panel(
        stdscr,
        1,
        left_w,
        panel_h,
        right_w,
        "LoRA Hyperparameters & Unified Memory Estimator",
        is_focused=is_lora_right,
        subtitle="Step 2: Config",
    )

    col1_x = left_w + 2
    col2_x = left_w + 2 + max(18, (right_w - 4) // 2)
    col_v_w = max(6, min(10, (right_w - 4) // 4 - 3))

    draw_field(stdscr, 2, col1_x, "Iters", str(state.lora_config.iters), is_focused=(is_lora_right and state.lora_right_focus_idx == 0), val_width=col_v_w)
    draw_field(stdscr, 2, col2_x, "Batch", str(state.lora_config.batch_size), is_focused=(is_lora_right and state.lora_right_focus_idx == 1), val_width=col_v_w)

    draw_field(stdscr, 3, col1_x, "LR", f"{state.lora_config.learning_rate:g}", is_focused=(is_lora_right and state.lora_right_focus_idx == 2), val_width=col_v_w)
    draw_field(stdscr, 3, col2_x, "Rank", str(state.lora_config.lora_rank), is_focused=(is_lora_right and state.lora_right_focus_idx == 3), val_width=col_v_w)

    draw_field(stdscr, 4, col1_x, "Alpha", f"{state.lora_config.lora_alpha:g}", is_focused=(is_lora_right and state.lora_right_focus_idx == 4), val_width=col_v_w)
    draw_field(stdscr, 4, col2_x, "Dropout", f"{state.lora_config.lora_dropout:g}", is_focused=(is_lora_right and state.lora_right_focus_idx == 5), val_width=col_v_w)

    draw_field(stdscr, 5, col1_x, "SeqLen", str(state.lora_config.max_seq_length), is_focused=(is_lora_right and state.lora_right_focus_idx == 6), val_width=col_v_w)
    draw_field(stdscr, 5, col2_x, "Layers", str(state.lora_config.num_layers), is_focused=(is_lora_right and state.lora_right_focus_idx == 7), val_width=col_v_w)

    chk_str = "[X] True" if state.lora_config.grad_checkpoint else "[ ] False"
    mask_str = "[X] True" if state.lora_config.mask_prompt else "[ ] False"
    draw_field(stdscr, 6, col1_x, "GradChk", chk_str, is_focused=(is_lora_right and state.lora_right_focus_idx == 8), val_width=col_v_w)
    draw_field(stdscr, 6, col2_x, "MaskPmt", mask_str, is_focused=(is_lora_right and state.lora_right_focus_idx == 9), val_width=col_v_w)

    draw_field(stdscr, 7, col1_x, "SaveEv", str(state.lora_config.save_every), is_focused=(is_lora_right and state.lora_right_focus_idx == 10), val_width=col_v_w)
    draw_field(stdscr, 7, col2_x, "StepsEval", str(state.lora_config.steps_per_eval), is_focused=(is_lora_right and state.lora_right_focus_idx == 11), val_width=col_v_w)

    ad_disp = state.lora_config.adapter_path
    if len(ad_disp) > col_v_w + 4:
        ad_disp = "…" + ad_disp[-(col_v_w + 3):]
    draw_field(stdscr, 8, col1_x, "Adapter", ad_disp, is_focused=(is_lora_right and state.lora_right_focus_idx == 12), val_width=col_v_w)
    draw_button(stdscr, 8, col2_x, "+ Add to Queue (F6)", is_focused=(is_lora_right and state.lora_right_focus_idx == 13))

    safe_addstr(stdscr, 9, col1_x, "─" * (right_w - 4), (get_color(COLOR_BORDER_JOINTS) | curses.A_DIM) if curses.has_colors() else curses.A_DIM)

    epochs = state.get_implied_epochs()
    if epochs is not None:
        safe_addstr(stdscr, 10, col1_x, f"• Implied Epochs: {epochs:.2f} epochs", (get_color(COLOR_TITLE_ACCENT) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
    else:
        safe_addstr(stdscr, 10, col1_x, "• Implied Epochs: (train.jsonl needed for exact calculation)", (get_color(COLOR_LABEL_GRAY) | curses.A_DIM) if curses.has_colors() else curses.A_DIM)

    mem_est = state.get_memory_estimate()
    lvl = mem_est.get("safety_level", "SAFE")
    if lvl == "SAFE":
        lvl_attr = (get_color(COLOR_SUCCESS) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD
    elif lvl == "TIGHT":
        lvl_attr = (get_color(COLOR_TITLE_ACCENT) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD
    else:
        lvl_attr = (get_color(COLOR_ERROR) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD
    safe_addstr(stdscr, 11, col1_x, f"• {mem_est.get('badge', '')}"[:right_w - 4], lvl_attr)

    dur_est = state.get_duration_estimate()
    dur_line = f"• Est. Duration: ~{dur_est.get('duration_str', '1m')} (ETA: {dur_est.get('eta_clock', 'N/A')})"
    safe_addstr(stdscr, 12, col1_x, dur_line[:right_w - 4], (get_color(COLOR_NORMAL_TEXT) | curses.A_DIM) if curses.has_colors() else curses.A_DIM)

    # 3. Middle Panel: Central Queue & Config Browser
    runs_list = state.queue_manager.runs if state.queue_manager else []
    draw_box_panel(
        stdscr,
        vis_y,
        0,
        vis_h,
        max_x,
        "Fine-Tuning Queue & Config Browser",
        is_focused=is_lora_queue,
        subtitle=f"{len(runs_list)} run(s) queued",
    )
    state.selected_queue_idx = draw_queue_table(
        stdscr,
        vis_y + 1,
        2,
        vis_h - 2,
        max_x - 4,
        runs_list,
        selected_idx=state.selected_queue_idx,
        is_focused=is_lora_queue,
    )

    # 4. Bottom Panel: Live MLX Command & YAML Preview
    draw_box_panel(
        stdscr,
        preview_y,
        0,
        preview_h,
        max_x,
        "Live LoRA Command & YAML Preview (mlx_lm.lora)",
        is_focused=False,
        subtitle="CLI / YAML",
    )
    cmd_text = state.lora_config.to_cli_command()
    safe_addstr(stdscr, preview_y + 1, 2, "Command: ", (get_color(COLOR_LABEL_GRAY) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
    safe_addstr(stdscr, preview_y + 1, 11, cmd_text[:max_x - 13], (get_color(COLOR_INPUT_NORMAL) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)

    if preview_h >= 4:
        cfg = state.lora_config
        summary_line = f"Config Spec: iters={cfg.iters} | batch={cfg.batch_size} | lr={cfg.learning_rate:g} | rank={cfg.lora_rank} | alpha={cfg.lora_alpha:g} | layers={cfg.num_layers} | grad_chk={cfg.grad_checkpoint} | mask_pmt={cfg.mask_prompt}"
        safe_addstr(stdscr, preview_y + 2, 2, summary_line[:max_x - 4], get_color(COLOR_NORMAL_TEXT) if curses.has_colors() else curses.A_DIM)


def _handle_mode1_input(
    stdscr: curses.window,
    state: CommanderState,
    key: int,
    formats_list: List[MLXFormat],
) -> Optional[ConversionResult]:
    """Handle keyboard input specific to Mode 1 (Dataset Conversion)."""
    mapping_fields = state.get_mapping_fields_for_format()
    total_right_fields = 11 + len(mapping_fields)
    is_left = (state.active_panel == ActivePanel.LEFT)
    is_right = (state.active_panel == ActivePanel.RIGHT)

    if key in (curses.KEY_F2, 15):  # F2 or Ctrl+O
        curses.def_prog_mode()
        curses.endwin()
        from mlx_commander.gui_picker import pick_dataset_gui
        chosen = pick_dataset_gui(default_dir=state.dataset_path or os.getcwd())
        curses.reset_prog_mode()
        stdscr.refresh()
        if chosen:
            if not state.load_dataset(chosen):
                if state.last_missing_dependency:
                    if show_missing_dependency_dialog(stdscr, state.last_missing_dependency):
                        state.load_dataset(chosen)
                else:
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

    elif key in (ord("r"), ord("R")):
        state.randomize_seed()

    elif key in (9, curses.KEY_BTAB):  # Tab / Shift-Tab
        state.active_panel = ActivePanel.RIGHT if state.active_panel == ActivePanel.LEFT else ActivePanel.LEFT

    # Navigation in Left Panel (Tab 1)
    elif is_left:
        num_cols = len(state.loaded_dataset.columns) if (state.loaded_dataset and state.loaded_dataset.columns) else 0
        if key in (curses.KEY_UP, curses.KEY_LEFT, ord("k"), ord("h")):
            if state.left_focus_idx == 2 and state.selected_column_idx > 0:
                state.selected_column_idx -= 1
            elif state.left_focus_idx == 2:
                state.left_focus_idx = 1
            elif state.left_focus_idx == 1:
                state.left_focus_idx = 0
            elif state.left_focus_idx == 0:
                state.active_panel = ActivePanel.RIGHT
                state.right_focus_idx = total_right_fields - 1
        elif key in (curses.KEY_DOWN, curses.KEY_RIGHT, ord("j"), ord("l")):
            if state.left_focus_idx == 0:
                state.left_focus_idx = 1
            elif state.left_focus_idx == 1:
                if num_cols > 0:
                    state.left_focus_idx = 2
                    state.selected_column_idx = 0
                else:
                    state.active_panel = ActivePanel.RIGHT
                    state.right_focus_idx = 0
            elif state.left_focus_idx == 2:
                if state.selected_column_idx < num_cols - 1:
                    state.selected_column_idx += 1
                else:
                    state.active_panel = ActivePanel.RIGHT
                    state.right_focus_idx = 0
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
                        if state.last_missing_dependency:
                            if show_missing_dependency_dialog(stdscr, state.last_missing_dependency):
                                state.load_dataset(chosen)
                        else:
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
                        if state.last_missing_dependency:
                            if show_missing_dependency_dialog(stdscr, state.last_missing_dependency):
                                state.load_dataset(new_path)
                        else:
                            show_error_dialog(stdscr, "Dataset Loading Failed", state.status_message)

    # Navigation in Right Panel (Tab 2)
    elif is_right:
        num_cols = len(state.loaded_dataset.columns) if (state.loaded_dataset and state.loaded_dataset.columns) else 0
        if key in (curses.KEY_UP, curses.KEY_LEFT, ord("k"), ord("h")):
            if state.right_focus_idx > 0:
                state.right_focus_idx -= 1
            else:
                state.active_panel = ActivePanel.LEFT
                if num_cols > 0:
                    state.left_focus_idx = 2
                    state.selected_column_idx = num_cols - 1
                else:
                    state.left_focus_idx = 1
        elif key in (curses.KEY_DOWN, curses.KEY_RIGHT, ord("j"), ord("l")):
            if state.right_focus_idx < total_right_fields - 1:
                state.right_focus_idx += 1
            else:
                state.active_panel = ActivePanel.LEFT
                state.left_focus_idx = 0
        elif key in (10, 13, curses.KEY_ENTER, 32):  # Enter or Space
            idx = state.right_focus_idx
            if 0 <= idx <= 3:
                state.set_format(formats_list[idx])
            elif 4 <= idx <= 3 + len(mapping_fields):
                f_info = mapping_fields[idx - 4]
                cols = state.loaded_dataset.columns if state.loaded_dataset else []
                chosen = show_column_picker_dialog(
                    stdscr,
                    f"Select Column for '{f_info['label']}'",
                    cols,
                    current_val=f_info["current"],
                    allow_none=True,
                )
                state.set_mapping_field(f_info["key"], chosen)
            elif idx == 4 + len(mapping_fields):  # Train %
                val = show_text_edit_dialog(stdscr, "Train Split %", "Enter Train percentage (0-100):", f"{state.train_pct:.0f}", is_number=True)
                if val:
                    try: state.train_pct = float(val)
                    except ValueError: pass
            elif idx == 5 + len(mapping_fields):  # Valid %
                val = show_text_edit_dialog(stdscr, "Valid Split %", "Enter Valid percentage (0-100):", f"{state.valid_pct:.0f}", is_number=True)
                if val:
                    try: state.valid_pct = float(val)
                    except ValueError: pass
            elif idx == 6 + len(mapping_fields):  # Test %
                val = show_text_edit_dialog(stdscr, "Test Split %", "Enter Test percentage (0-100):", f"{state.test_pct:.0f}", is_number=True)
                if val:
                    try: state.test_pct = float(val)
                    except ValueError: pass
            elif idx == 7 + len(mapping_fields):  # Seed
                val = show_text_edit_dialog(stdscr, "Random Seed", "Enter random seed integer:", str(state.seed), is_number=True)
                if val:
                    try: state.seed = int(val)
                    except ValueError: pass
            elif idx == 8 + len(mapping_fields):  # Randomize
                state.randomize_seed()
            elif idx == 9 + len(mapping_fields):  # Output Dir
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
            elif idx == 10 + len(mapping_fields):  # Convert button
                res = execute_conversion(stdscr, state)
                if res is not None:
                    return res

    return None


def _handle_mode2_input(
    stdscr: curses.window,
    state: CommanderState,
    key: int,
) -> None:
    """Handle keyboard input specific to Mode 2 (LoRA Fine-Tuning & Queue)."""
    if key in (9,):  # Tab
        if state.lora_active_panel == "left":
            state.lora_active_panel = "right"
        elif state.lora_active_panel == "right":
            state.lora_active_panel = "queue"
        else:
            state.lora_active_panel = "left"
    elif key in (curses.KEY_BTAB,):  # Shift-Tab
        if state.lora_active_panel == "left":
            state.lora_active_panel = "queue"
        elif state.lora_active_panel == "queue":
            state.lora_active_panel = "right"
        else:
            state.lora_active_panel = "left"

    elif key in (curses.KEY_F2, 15):  # F2 or Ctrl+O (Finder)
        curses.def_prog_mode()
        curses.endwin()
        from mlx_commander.gui_picker import pick_folder_gui
        chosen = pick_folder_gui("Select Dataset Folder", default_dir=state.lora_config.data or os.getcwd())
        curses.reset_prog_mode()
        stdscr.refresh()
        if chosen:
            state.lora_config.data = chosen.strip()
            state.status_message = f"Dataset folder set to: {chosen}"
            state.status_is_error = False

    # Navigation in Left Panel (Setup)
    elif state.lora_active_panel == "left":
        if key in (curses.KEY_UP, ord("k")):
            state.lora_left_focus_idx = max(0, state.lora_left_focus_idx - 1)
        elif key in (curses.KEY_DOWN, ord("j")):
            state.lora_left_focus_idx = min(6, state.lora_left_focus_idx + 1)
        elif key in (curses.KEY_RIGHT, ord("l")):
            state.lora_active_panel = "right"
        elif key in (10, 13, curses.KEY_ENTER, 32):  # Enter or Space
            idx = state.lora_left_focus_idx
            if idx == 0:  # Base Model
                chosen = show_model_picker_dialog(stdscr, state.lora_config.model)
                if chosen:
                    state.lora_config.model = chosen
                    model_slug = chosen.split("/")[-1].replace("-Instruct", "").replace("-4bit", "")
                    state.lora_config.name = f"{model_slug} (lr={state.lora_config.learning_rate:g}, r={state.lora_config.lora_rank})"
            elif idx == 1:  # Dataset path
                val = show_text_edit_dialog(
                    stdscr,
                    "Dataset Path",
                    "Enter directory containing train.jsonl / valid.jsonl:",
                    default_val=state.lora_config.data,
                )
                if val:
                    state.lora_config.data = val.strip()
            elif idx == 2:  # Finder button
                curses.def_prog_mode()
                curses.endwin()
                from mlx_commander.gui_picker import pick_folder_gui
                chosen = pick_folder_gui("Select Dataset Folder", default_dir=state.lora_config.data or os.getcwd())
                curses.reset_prog_mode()
                stdscr.refresh()
                if chosen:
                    state.lora_config.data = chosen.strip()
                    state.status_message = f"Dataset folder set to: {chosen}"
                    state.status_is_error = False
            elif idx == 3:  # Method
                chosen = show_choice_dialog(stdscr, "Fine-Tune Method", "Select technique:", FINE_TUNE_TYPES, state.lora_config.fine_tune_type)
                if chosen:
                    state.lora_config.fine_tune_type = chosen
            elif idx == 4:  # Optimizer
                chosen = show_choice_dialog(stdscr, "Optimizer", "Select training optimizer:", OPTIMIZERS, state.lora_config.optimizer)
                if chosen:
                    state.lora_config.optimizer = chosen
            elif idx == 5:  # Mode (toggle train/test)
                state.lora_config.train = not state.lora_config.train
            elif idx == 6:  # Run Name
                val = show_text_edit_dialog(stdscr, "Run Name", "Enter descriptive label for this run:", default_val=state.lora_config.name)
                if val:
                    state.lora_config.name = val.strip()

    # Navigation in Right Panel (Hyperparameters)
    elif state.lora_active_panel == "right":
        if key in (curses.KEY_UP, ord("k")):
            state.lora_right_focus_idx = max(0, state.lora_right_focus_idx - 2)
        elif key in (curses.KEY_DOWN, ord("j")):
            state.lora_right_focus_idx = min(13, state.lora_right_focus_idx + 2)
        elif key in (curses.KEY_LEFT, ord("h")):
            if state.lora_right_focus_idx % 2 == 1:
                state.lora_right_focus_idx -= 1
            else:
                state.lora_active_panel = "left"
        elif key in (curses.KEY_RIGHT, ord("l")):
            if state.lora_right_focus_idx % 2 == 0 and state.lora_right_focus_idx < 13:
                state.lora_right_focus_idx += 1
        elif key in (10, 13, curses.KEY_ENTER, 32):  # Enter or Space
            idx = state.lora_right_focus_idx
            if idx == 0:  # Iters
                val = show_text_edit_dialog(stdscr, "Training Iterations", "Enter number of iterations (e.g. 1000):", str(state.lora_config.iters), is_number=True)
                if val:
                    try: state.lora_config.iters = max(1, int(val))
                    except ValueError: pass
            elif idx == 1:  # Batch Size
                val = show_text_edit_dialog(stdscr, "Batch Size", "Enter micro-batch size (e.g. 4):", str(state.lora_config.batch_size), is_number=True)
                if val:
                    try: state.lora_config.batch_size = max(1, int(val))
                    except ValueError: pass
            elif idx == 2:  # Learning Rate
                val = show_text_edit_dialog(stdscr, "Learning Rate", "Enter learning rate (e.g. 1e-5):", f"{state.lora_config.learning_rate:g}")
                if val:
                    try: state.lora_config.learning_rate = float(val)
                    except ValueError: pass
            elif idx == 3:  # LoRA Rank
                val = show_text_edit_dialog(stdscr, "LoRA Rank", "Enter rank dimension r (e.g. 8, 16):", str(state.lora_config.lora_rank), is_number=True)
                if val:
                    try: state.lora_config.lora_rank = max(1, int(val))
                    except ValueError: pass
            elif idx == 4:  # LoRA Alpha
                val = show_text_edit_dialog(stdscr, "LoRA Scale / Alpha", "Enter alpha scaling factor (e.g. 16.0):", f"{state.lora_config.lora_alpha:g}")
                if val:
                    try: state.lora_config.lora_alpha = float(val)
                    except ValueError: pass
            elif idx == 5:  # LoRA Dropout
                val = show_text_edit_dialog(stdscr, "LoRA Dropout", "Enter dropout probability (0.0 to 0.5):", f"{state.lora_config.lora_dropout:g}")
                if val:
                    try: state.lora_config.lora_dropout = float(val)
                    except ValueError: pass
            elif idx == 6:  # Max Seq Len
                val = show_text_edit_dialog(stdscr, "Max Sequence Length", "Enter context window limit (e.g. 2048):", str(state.lora_config.max_seq_length), is_number=True)
                if val:
                    try: state.lora_config.max_seq_length = max(64, int(val))
                    except ValueError: pass
            elif idx == 7:  # Num Layers
                val = show_text_edit_dialog(stdscr, "LoRA Fine-Tuned Layers", "Enter number of top layers to adapt (e.g. 16):", str(state.lora_config.num_layers), is_number=True)
                if val:
                    try: state.lora_config.num_layers = max(1, int(val))
                    except ValueError: pass
            elif idx == 8:  # Grad Checkpoint
                state.lora_config.grad_checkpoint = not state.lora_config.grad_checkpoint
            elif idx == 9:  # Mask Prompt
                state.lora_config.mask_prompt = not state.lora_config.mask_prompt
            elif idx == 10:  # Save Every
                val = show_text_edit_dialog(stdscr, "Save Every", "Save adapter checkpoint every N steps:", str(state.lora_config.save_every), is_number=True)
                if val:
                    try: state.lora_config.save_every = max(1, int(val))
                    except ValueError: pass
            elif idx == 11:  # Steps per eval
                val = show_text_edit_dialog(stdscr, "Steps per Eval", "Evaluate on validation split every N steps:", str(state.lora_config.steps_per_eval), is_number=True)
                if val:
                    try: state.lora_config.steps_per_eval = max(1, int(val))
                    except ValueError: pass
            elif idx == 12:  # Adapter Out
                val = show_text_edit_dialog(stdscr, "Adapter Path", "Directory to store fine-tuned LoRA weights:", state.lora_config.adapter_path)
                if val:
                    state.lora_config.adapter_path = val.strip()
            elif idx == 13:  # Add to Queue
                added = state.add_current_lora_to_queue()
                state.status_message = f"Added '{added.name}' to queue ({len(state.queue_manager.runs)} run(s) queued)."
                state.status_is_error = False

    # Navigation in Queue Panel
    elif state.lora_active_panel == "queue":
        num_runs = len(state.queue_manager.runs) if state.queue_manager else 0
        if key in (curses.KEY_UP, ord("k")):
            if state.selected_queue_idx > 0:
                state.selected_queue_idx -= 1
        elif key in (curses.KEY_DOWN, ord("j")):
            if state.selected_queue_idx < num_runs - 1:
                state.selected_queue_idx += 1
        elif key in (ord("c"), ord("C")):  # Clone
            if state.queue_manager and state.queue_manager.runs:
                curr_r = state.queue_manager.runs[state.selected_queue_idx]
                cloned = state.queue_manager.clone_run(curr_r.id)
                if cloned:
                    state.selected_queue_idx = len(state.queue_manager.runs) - 1
                    state.status_message = f"Cloned run '{cloned.name}'."
                    state.status_is_error = False
        elif key in (ord("d"), ord("D")):  # Delete
            if state.queue_manager and state.queue_manager.runs:
                curr_r = state.queue_manager.runs[state.selected_queue_idx]
                state.queue_manager.delete_run(curr_r.id)
                if state.selected_queue_idx >= len(state.queue_manager.runs):
                    state.selected_queue_idx = max(0, len(state.queue_manager.runs) - 1)
                state.status_message = f"Deleted run '{curr_r.name}'."
                state.status_is_error = False
        elif key in (ord("x"), ord("X")):  # Clear
            if state.queue_manager and state.queue_manager.runs:
                state.queue_manager.clear_queue()
                state.selected_queue_idx = 0
                state.status_message = "LoRA queue cleared."
                state.status_is_error = False
        elif key in (10, 13, curses.KEY_ENTER):  # Load run to form
            if state.queue_manager and state.queue_manager.runs:
                curr_r = state.queue_manager.runs[state.selected_queue_idx]
                state.load_queue_run_into_form(curr_r.id)
                state.status_message = f"Loaded run '{curr_r.name}' into form editor."
                state.status_is_error = False


def run_commander_tui(
    stdscr: curses.window,
    default_dataset_path: Optional[str] = None,
    initial_state: Optional[CommanderState] = None,
    prefill: Optional[Dict[str, Any]] = None,
) -> Optional[ConversionResult]:
    """Main event loop for the persistent MLX Commander dashboard."""
    configure_escdelay(25)
    state = initial_state if initial_state is not None else CommanderState()

    if prefill:
        state.apply_prefill(prefill)

    init_colors(state.theme_mode)
    if ThemeMode.is_norton(state.theme_mode):
        try:
            stdscr.bkgd(" ", get_color(COLOR_PANEL_BG))
        except curses.error:
            pass

    curses.curs_set(0)
    stdscr.keypad(True)

    target_dataset_path = default_dataset_path or state.dataset_path
    if not state.loaded_dataset and target_dataset_path:
        try:
            p = Path(target_dataset_path).resolve()
            if p.exists():
                loaded = state.load_dataset(str(p))
                if not loaded and state.last_missing_dependency:
                    if show_missing_dependency_dialog(stdscr, state.last_missing_dependency):
                        state.load_dataset(str(p))
            else:
                state.dataset_path = str(p)
                if not state.has_custom_output_dir:
                    base = p if p.is_dir() else p.parent
                    state.output_dir = str(base / "mlx_dataset")
        except Exception:
            state.dataset_path = target_dataset_path
            if not state.has_custom_output_dir:
                try:
                    p = Path(target_dataset_path).resolve()
                    base = p if p.is_dir() else p.parent
                    state.output_dir = str(base / "mlx_dataset")
                except Exception:
                    state.output_dir = str(Path.cwd() / "mlx_dataset")
    elif not state.loaded_dataset:
        if not state.has_custom_output_dir and not state.output_dir:
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
            safe_addstr(stdscr, 1, 2, "Terminal window too small for MLX Commander.", curses.A_BOLD)
            safe_addstr(stdscr, 2, 2, f"Current: {max_x}x{max_y} (Minimum required: 70x16)", curses.A_DIM)
            safe_addstr(stdscr, 4, 2, "Please resize your terminal window or press [q] to exit.", curses.A_DIM)
            stdscr.refresh()
            k = stdscr.getch()
            if k in (ord("q"), ord("Q"), 27):
                return None
            continue

        # ----------------------------------------------------
        # 1. Header Banner with Dual Mode Switcher Tabs
        # ----------------------------------------------------
        hdr_attr = (get_color(COLOR_BANNER) | curses.A_BOLD) if curses.has_colors() else curses.A_STANDOUT
        safe_addstr(stdscr, 0, 0, " " * max_x, hdr_attr)

        title_prefix = "MLX Commander  ::  "
        safe_addstr(stdscr, 0, 2, title_prefix, hdr_attr)
        x_m1 = 2 + len(title_prefix)

        mode1_title = "[ 1: Dataset Converter ]"
        mode2_title = "[ 2: LoRA Fine-Tuning (F4) ]"
        if state.active_tab == 0:
            m1_attr = hdr_attr | curses.A_STANDOUT | curses.A_BOLD
            m2_attr = hdr_attr | curses.A_DIM
        else:
            m1_attr = hdr_attr | curses.A_DIM
            m2_attr = hdr_attr | curses.A_STANDOUT | curses.A_BOLD

        safe_addstr(stdscr, 0, x_m1, mode1_title, m1_attr)
        x_m2 = x_m1 + len(mode1_title) + 2
        safe_addstr(stdscr, 0, x_m2, mode2_title, m2_attr)

        hint_str = "[F1: Help | F4: Mode | F9: Scheme | F10: Exit]" if max_x >= 100 else "[F4: Mode | F10: Exit]"
        safe_addstr(stdscr, 0, max(2, max_x - len(hint_str) - 2), hint_str, hdr_attr)

        # ----------------------------------------------------
        # 2. Dimensions & Coordinates (3-Tier Responsive Layout)
        # ----------------------------------------------------
        left_w = max(34, max_x // 2)
        right_w = max_x - left_w

        if state.active_tab == 0:
            _draw_mode1_dashboard(stdscr, state, max_y, max_x, left_w, right_w)
        else:
            _draw_mode2_dashboard(stdscr, state, max_y, max_x, left_w, right_w)

        # ----------------------------------------------------
        # 3. Bottom Status / Hotkey Bar
        # ----------------------------------------------------
        footer_y = max_y - 1
        if ThemeMode.is_norton(state.theme_mode):
            if state.active_tab == 0:
                fn_items = [
                    ("1", "Help"),
                    ("2", "Open"),
                    ("3", "Output"),
                    ("4", "LoRA"),
                    ("5", "Convert"),
                    ("F9", "Scheme"),
                    ("10", "Exit"),
                ]
            else:
                fn_items = [
                    ("1", "Help"),
                    ("2", "Data"),
                    ("4", "Convert"),
                    ("5", "RunQueue"),
                    ("6", "AddRun"),
                    ("F9", "Scheme"),
                    ("10", "Exit"),
                ]
            safe_addstr(stdscr, footer_y, 0, " " * max_x, get_color(COLOR_PANEL_BG))
            cur_x = 0
            if max_x >= 100 and state.status_message:
                prefix = "[OK] " if not state.status_is_error else "[ERR] "
                short_status = f"{prefix}{state.status_message}"[:max_x - 65]
                stat_attr = (get_color(COLOR_TITLE_ACCENT) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD
                safe_addstr(stdscr, footer_y, 0, short_status, stat_attr)
                cur_x = len(short_status) + 2

            for num_str, lbl_str in fn_items:
                if cur_x + len(num_str) + len(lbl_str) + 2 >= max_x:
                    break
                num_attr = (get_color(COLOR_FN_NUMBER) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD
                lbl_attr = get_color(COLOR_FN_LABEL) if curses.has_colors() else curses.A_STANDOUT
                safe_addstr(stdscr, footer_y, cur_x, f" {num_str} ", num_attr)
                cur_x += len(num_str) + 2
                safe_addstr(stdscr, footer_y, cur_x, f"{lbl_str} ", lbl_attr)
                cur_x += len(lbl_str) + 1
        else:
            safe_addstr(stdscr, footer_y, 0, " " * max_x, hdr_attr)
            if state.active_tab == 0:
                bar_shortcuts = "[Tab] Switch  [F4] LoRA Mode  [Enter] Select  [F2] Open  [F3] Output  [F5] Convert  [F9] Scheme" if max_x >= 102 else "[Tab] Switch  [F4] LoRA  [F5] Convert  [F9] Scheme"
            else:
                bar_shortcuts = "[Tab] Switch Pane  [F4] Dataset Mode  [F5] Run Queue  [F6] Add Run  [c] Clone  [d] Del  [F9] Scheme" if max_x >= 102 else "[Tab] Pane  [F4] Convert  [F5] Run  [F6] Add  [c] Clone"
            shortcuts_x = max(10, max_x - len(bar_shortcuts) - 2)
            avail_status = max(10, shortcuts_x - 4)

            status_prefix = "[OK] " if not state.status_is_error else "[ERR] "
            status_text = f"{status_prefix}{state.status_message}"[:avail_status]
            safe_addstr(stdscr, footer_y, 2, status_text, hdr_attr | curses.A_BOLD)
            safe_addstr(stdscr, footer_y, shortcuts_x, bar_shortcuts, hdr_attr)

        stdscr.refresh()

        # ----------------------------------------------------
        # 4. Input & Key Event Handling
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

            elif key in (curses.KEY_F4, 20):  # F4 or Ctrl+T (Mode switcher)
                state.active_tab = 1 - state.active_tab
                if state.active_tab == 1:
                    state.sync_dataset_to_lora()
                    state.status_message = "Switched to LoRA Fine-Tuning Mode."
                else:
                    state.status_message = "Switched to Dataset Conversion Mode."
                state.status_is_error = False

            elif key == curses.KEY_F6:
                added = state.add_current_lora_to_queue()
                state.status_message = f"Added '{added.name}' to queue ({len(state.queue_manager.runs)} run(s) queued)."
                state.status_is_error = False

            elif key == curses.KEY_F5:
                if state.active_tab == 0:
                    res = execute_conversion(stdscr, state)
                    if res is not None:
                        conversion_result = res
                else:
                    execute_lora_queue_action(stdscr, state)

            elif key == curses.KEY_F9:
                state.toggle_theme()
                init_colors(state.theme_mode)
                if ThemeMode.is_norton(state.theme_mode):
                    try:
                        stdscr.bkgd(" ", get_color(COLOR_PANEL_BG))
                    except curses.error:
                        pass
                    state.status_message = "Theme: Norton Commander (Classic Blue)"
                else:
                    try:
                        stdscr.bkgd(" ", curses.color_pair(0))
                    except curses.error:
                        pass
                    state.status_message = "Theme: Modern Terminal"
                state.status_is_error = False
                try:
                    stdscr.touchwin()
                except curses.error:
                    pass
                stdscr.clear()

            elif state.active_tab == 0:
                res = _handle_mode1_input(stdscr, state, key, formats_list)
                if res is not None:
                    conversion_result = res
            else:
                _handle_mode2_input(stdscr, state, key)

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
    """Launch the MLX Commander full-screen curses dashboard with optional prefill."""
    configure_escdelay(25)
    try:
        return curses.wrapper(run_commander_tui, default_dataset_path, initial_state, prefill)
    except KeyboardInterrupt:
        return None

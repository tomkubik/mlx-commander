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
    draw_radio,
    get_color,
    safe_addstr,
    show_column_picker_dialog,
    show_error_dialog,
    show_help_dialog,
    show_message_dialog,
    show_missing_dependency_dialog,
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
        # 1. Header Banner
        # ----------------------------------------------------
        hdr_attr = (get_color(COLOR_BANNER) | curses.A_BOLD) if curses.has_colors() else curses.A_STANDOUT
        safe_addstr(stdscr, 0, 0, " " * max_x, hdr_attr)
        if ThemeMode.is_norton(state.theme_mode):
            safe_addstr(stdscr, 0, 2, "MLX Commander  ::  Norton Commander Classic Mode", hdr_attr)
            hint_str = "[F1: Help | F2: Open | F3: Output | F5: Convert | F9: Scheme | F10: Exit]" if max_x >= 92 else "[F1: Help | F9: Scheme]"
        else:
            safe_addstr(stdscr, 0, 2, "MLX Commander  ::  Persistent Dataset Conversion Dashboard", hdr_attr)
            hint_str = "[F1: Help | F2: Open | F3: Output | F5: Convert | F9: Scheme | F10: Exit]" if max_x >= 92 else "[F1: Help | F10: Exit]"
        safe_addstr(stdscr, 0, max(2, max_x - len(hint_str) - 2), hint_str, hdr_attr)

        # ----------------------------------------------------
        # Dimensions & Coordinates (3-Tier Responsive Layout)
        # ----------------------------------------------------
        left_w = max(34, max_x // 2)
        right_w = max_x - left_w

        # Tier 1: Top configuration panels
        mapping_fields = state.get_mapping_fields_for_format()
        min_top_h = 14 + len(mapping_fields)
        needed_top_h = max(16, min_top_h)
        max_possible_top = max(min_top_h, max_y - 10)
        panel_h = min(needed_top_h, max_possible_top)

        # Tier 2: Middle visual mapping pipeline
        remaining_y = max(8, max_y - 1 - (panel_h + 1))
        vis_h = max(4, min(10, remaining_y // 2))
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
        tab1_right_edge = left_w - 3
        max_path_w = max(10, left_w - 12)
        if len(disp_path) > max_path_w:
            disp_path = "…" + disp_path[-(max_path_w - 1):]
        path_x = max(8, tab1_right_edge - len(disp_path) + 1)
        safe_addstr(stdscr, 2, 2, "Path: ", (get_color(COLOR_LABEL_GRAY) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
        safe_addstr(stdscr, 2, path_x, disp_path, get_color(COLOR_NORMAL_TEXT) if curses.has_colors() else curses.A_DIM)

        # Action Buttons (stacked vertically)
        f2_focus = is_left and state.left_focus_idx == 0
        edit_focus = is_left and state.left_focus_idx == 1
        draw_button(stdscr, 3, 2, "Finder (F2)", is_focused=f2_focus)
        draw_button(stdscr, 4, 2, "Change Path", is_focused=edit_focus)
        safe_addstr(stdscr, 5, 2, "Tip: You can load multiple files (select multiple or use commas)"[:left_w - 4], get_color(COLOR_LABEL_GRAY) if curses.has_colors() else curses.A_DIM)

        # Dataset Stats
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

        # Available Columns List
        safe_addstr(stdscr, 8, 2, f"Columns ({len(cols)}):", (get_color(COLOR_LABEL_GRAY) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
        col_list_focus = is_left and state.left_focus_idx == 2
        col_list_start_y = 9
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
                        c_attr = (get_color(COLOR_INPUT_FOCUSED) | curses.A_BOLD) if curses.has_colors() else curses.A_STANDOUT
                    elif col_list_focus:
                        c_attr = (get_color(COLOR_BORDER_JOINTS) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD
                    else:
                        c_attr = get_color(COLOR_NORMAL_TEXT) if curses.has_colors() else 0
                    safe_addstr(stdscr, row_y, 2, text, c_attr)

            # Sample value preview for currently highlighted column
            if 0 <= state.selected_column_idx < len(cols) and state.loaded_dataset and state.loaded_dataset.sample_records:
                col_name = cols[state.selected_column_idx]
                sample_val = str(state.loaded_dataset.sample_records[0].get(col_name, ""))
                if len(sample_val) > left_w - 12:
                    sample_val = sample_val[:left_w - 13] + "…"
                safe_addstr(stdscr, panel_h - 1, 2, "Sample: ", (get_color(COLOR_LABEL_GRAY) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
                safe_addstr(stdscr, panel_h - 1, 10, sample_val, get_color(COLOR_NORMAL_TEXT) if curses.has_colors() else curses.A_DIM)
        else:
            safe_addstr(stdscr, col_list_start_y, 4, "(Load dataset to view schema)", get_color(COLOR_LABEL_GRAY) if curses.has_colors() else curses.A_DIM)


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
        total_right_fields = 11 + len(mapping_fields)
        if state.right_focus_idx >= total_right_fields:
            state.right_focus_idx = total_right_fields - 1

        # Fields 0..3: Target Format Radios (stacked vertically, one under another)
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

        # Fields 4..(3 + len(mapping_fields)): Column Mappings
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

        # Next rows: Splits & Output
        splits_start_y = 8 + len(mapping_fields)
        split_counts = state.get_split_counts()

        train_focus = is_right and state.right_focus_idx == 4 + len(mapping_fields)
        valid_focus = is_right and state.right_focus_idx == 5 + len(mapping_fields)
        test_focus = is_right and state.right_focus_idx == 6 + len(mapping_fields)

        # Splits layout:
        # Line 1: "Dataset split:" at left_w + 2, "Train" field right-aligned
        # Line 2: "Valid" field right-aligned
        # Line 3: "Test" field right-aligned
        # Labels for Train, Valid, Test are white font, unbolded.
        safe_addstr(
            stdscr,
            splits_start_y,
            left_w + 2,
            "Dataset split:",
            (get_color(COLOR_LABEL_GRAY) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD,
        )

        split_label_x = min(left_w + 18, right_edge - 16)
        white_unbold = get_color(COLOR_NORMAL_TEXT) if curses.has_colors() else 0

        draw_field(
            stdscr,
            splits_start_y,
            split_label_x,
            "Train",
            f"{state.train_pct:.0f}%",
            is_focused=train_focus,
            val_width=6,
            right_edge=right_edge,
            lbl_attr=white_unbold,
        )
        draw_field(
            stdscr,
            splits_start_y + 1,
            split_label_x,
            "Valid",
            f"{state.valid_pct:.0f}%",
            is_focused=valid_focus,
            val_width=6,
            right_edge=right_edge,
            lbl_attr=white_unbold,
        )
        draw_field(
            stdscr,
            splits_start_y + 2,
            split_label_x,
            "Test",
            f"{state.test_pct:.0f}%",
            is_focused=test_focus,
            val_width=6,
            right_edge=right_edge,
            lbl_attr=white_unbold,
        )

        # Seed & Randomize
        seed_y = splits_start_y + 3
        seed_focus = is_right and state.right_focus_idx == 7 + len(mapping_fields)
        rand_focus = is_right and state.right_focus_idx == 8 + len(mapping_fields)

        btn_w = 17  # len("[ Randomize (r) ]")
        seed_box_w = 10  # len("[ 123456 ]") with val_width=8
        btn_x = max(left_w + 18, right_edge - btn_w + 1)
        seed_field_x = max(left_w + 8, btn_x - 2 - seed_box_w)

        draw_field(stdscr, seed_y, left_w + 2, "Seed", str(state.seed), is_focused=seed_focus, val_width=8, field_x=seed_field_x)
        draw_button(stdscr, seed_y, btn_x, "Randomize (r)", is_focused=rand_focus)

        # Output Folder
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

        # Convert Action Button
        conv_focus = is_right and state.right_focus_idx == 10 + len(mapping_fields)
        btn_y = out_y + 1
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
                        safe_addstr(stdscr, curr_row, 3, f"{i + 1}: ", (get_color(COLOR_LABEL_GRAY) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
                        safe_addstr(stdscr, curr_row, 6, line_text, (get_color(COLOR_NORMAL_TEXT) | curses.A_BOLD) if curses.has_colors() else curses.A_BOLD)
                    else:
                        # Continuation line indented
                        safe_addstr(stdscr, curr_row, 6, line_text, get_color(COLOR_NORMAL_TEXT) if curses.has_colors() else 0)
                    curr_row += 1


        # ----------------------------------------------------
        # 6. Bottom Status / Hotkey Bar
        # ----------------------------------------------------
        footer_y = max_y - 1
        if ThemeMode.is_norton(state.theme_mode):
            # Classic Norton Commander Function Key Bar:
            # 1 Help  2 Open  3 Output  5 Convert  F9 Scheme  10 Exit
            fn_items = [
                ("1", "Help"),
                ("2", "Open"),
                ("3", "Output"),
                ("5", "Convert"),
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
            bar_shortcuts = "[Tab] Switch  [↑/↓ or ←/→] Navigate  [Enter] Select  [F2] Open  [F3] Output  [F5] Convert  [F9] Scheme" if max_x >= 102 else "[Tab] Switch  [F2] Open  [F5] Convert  [F9] Scheme"
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

            elif key == curses.KEY_F5:
                res = execute_conversion(stdscr, state)
                if res is not None:
                    conversion_result = res

            elif key in (ord("r"), ord("R")):
                state.randomize_seed()

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

            elif key in (9, curses.KEY_BTAB):  # Tab / Shift-Tab
                state.active_panel = ActivePanel.RIGHT if state.active_panel == ActivePanel.LEFT else ActivePanel.LEFT

            # Navigation in Left Panel (Tab 1)
            elif is_left:
                num_cols = len(state.loaded_dataset.columns) if (state.loaded_dataset and state.loaded_dataset.columns) else 0
                # Left arrow acts like Up arrow (switches UP)
                if key in (curses.KEY_UP, curses.KEY_LEFT, ord("k"), ord("h")):
                    if state.left_focus_idx == 2 and state.selected_column_idx > 0:
                        state.selected_column_idx -= 1
                    elif state.left_focus_idx == 2:
                        state.left_focus_idx = 1
                    elif state.left_focus_idx == 1:
                        state.left_focus_idx = 0
                    elif state.left_focus_idx == 0:
                        # Wrap UP to bottom of Tab 2 (Right Panel)
                        state.active_panel = ActivePanel.RIGHT
                        state.right_focus_idx = total_right_fields - 1
                # Right arrow acts like Down arrow (switches DOWN)
                elif key in (curses.KEY_DOWN, curses.KEY_RIGHT, ord("j"), ord("l")):
                    if state.left_focus_idx == 0:
                        state.left_focus_idx = 1
                    elif state.left_focus_idx == 1:
                        if num_cols > 0:
                            state.left_focus_idx = 2
                            state.selected_column_idx = 0
                        else:
                            # No columns: scroll DOWN directly into Tab 2
                            state.active_panel = ActivePanel.RIGHT
                            state.right_focus_idx = 0
                    elif state.left_focus_idx == 2:
                        if state.selected_column_idx < num_cols - 1:
                            state.selected_column_idx += 1
                        else:
                            # Reached bottom of Tab 1: scroll DOWN directly into Tab 2
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
                # Left arrow acts like Up arrow (switches UP)
                if key in (curses.KEY_UP, curses.KEY_LEFT, ord("k"), ord("h")):
                    if state.right_focus_idx > 0:
                        state.right_focus_idx -= 1
                    else:
                        # Reached top of Tab 2: scroll UP directly into Tab 1
                        state.active_panel = ActivePanel.LEFT
                        if num_cols > 0:
                            state.left_focus_idx = 2
                            state.selected_column_idx = num_cols - 1
                        else:
                            state.left_focus_idx = 1
                # Right arrow acts like Down arrow (switches DOWN)
                elif key in (curses.KEY_DOWN, curses.KEY_RIGHT, ord("j"), ord("l")):
                    if state.right_focus_idx < total_right_fields - 1:
                        state.right_focus_idx += 1
                    else:
                        # Reached bottom of Tab 2: wrap DOWN into top of Tab 1
                        state.active_panel = ActivePanel.LEFT
                        state.left_focus_idx = 0

                elif key in (10, 13, curses.KEY_ENTER, 32):  # Enter or Space
                    idx = state.right_focus_idx
                    if 0 <= idx <= 3:
                        # Select target format directly
                        state.set_format(formats_list[idx])

                    elif 4 <= idx <= 3 + len(mapping_fields):
                        # Column mapping picker
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
                            try:
                                state.train_pct = float(val)
                            except ValueError:
                                pass

                    elif idx == 5 + len(mapping_fields):  # Valid %
                        val = show_text_edit_dialog(stdscr, "Valid Split %", "Enter Valid percentage (0-100):", f"{state.valid_pct:.0f}", is_number=True)
                        if val:
                            try:
                                state.valid_pct = float(val)
                            except ValueError:
                                pass

                    elif idx == 6 + len(mapping_fields):  # Test %
                        val = show_text_edit_dialog(stdscr, "Test Split %", "Enter Test percentage (0-100):", f"{state.test_pct:.0f}", is_number=True)
                        if val:
                            try:
                                state.test_pct = float(val)
                            except ValueError:
                                pass

                    elif idx == 7 + len(mapping_fields):  # Seed
                        val = show_text_edit_dialog(stdscr, "Random Seed", "Enter random seed integer:", str(state.seed), is_number=True)
                        if val:
                            try:
                                state.seed = int(val)
                            except ValueError:
                                pass

                    elif idx == 8 + len(mapping_fields):  # Randomize button
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
    """Launch the MLX Commander full-screen curses dashboard with optional prefill."""
    configure_escdelay(25)
    try:
        return curses.wrapper(run_commander_tui, default_dataset_path, initial_state, prefill)
    except KeyboardInterrupt:
        return None

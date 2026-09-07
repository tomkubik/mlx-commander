"""
Curses TUI Widgets and UI Helpers.
Includes menus, text input fields, panels, and styled dialogs.
"""

import curses
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def safe_addstr(win: curses.window, y: int, x: int, text: str, attr: int = 0) -> None:
    """Safely print text inside a window without raising on border clip."""
    max_y, max_x = win.getmaxyx()
    if y < 0 or y >= max_y or x < 0 or x >= max_x:
        return
    avail = max_x - x
    if avail <= 0:
        return
    to_print = text[:avail]
    try:
        win.addstr(y, x, to_print, attr)
    except curses.error:
        pass


def safe_has_colors() -> bool:
    """Safely check if curses has colors initialized."""
    try:
        return curses.has_colors()
    except curses.error:
        return False


def get_color(pair_idx: int) -> int:
    """Safely return color pair attribute if colors are available."""
    try:
        if safe_has_colors():
            return curses.color_pair(pair_idx)
    except curses.error:
        pass
    return 0


def safe_curs_set(visibility: int) -> None:
    """Safely change cursor visibility without raising on uninitialized curses."""
    try:
        curses.curs_set(visibility)
    except curses.error:
        pass


def configure_escdelay(delay_ms: int = 25) -> None:
    """
    Configure ncurses ESC key delay to make exiting/canceling dialogs instantaneous.
    By default, ncurses waits 1000ms (1 full second) on an ESC keypress to disambiguate
    a standalone ESC from multi-byte escape sequences (such as arrow keys).
    Setting this to 25ms makes ESC exit instantaneous while still allowing
    arrow and function keys to be cleanly processed.
    """
    os.environ["ESCDELAY"] = str(delay_ms)
    try:
        curses.set_escdelay(delay_ms)
    except (AttributeError, curses.error):
        pass


def draw_header(stdscr: curses.window, title: str, step_info: str) -> None:
    """Draw top header banner with step indicator."""
    max_y, max_x = stdscr.getmaxyx()
    header_attr = get_color(1) | curses.A_BOLD if safe_has_colors() else curses.A_STANDOUT
    sub_attr = get_color(2) if safe_has_colors() else 0

    stdscr.attron(header_attr)
    safe_addstr(stdscr, 0, 0, " " * max_x, header_attr)
    safe_addstr(stdscr, 0, 2, f"MLX-Commander  ::  {title}", header_attr)
    stdscr.attroff(header_attr)

    if step_info:
        safe_addstr(stdscr, 1, 2, f"Step: {step_info}", sub_attr | curses.A_DIM)


def draw_footer(stdscr: curses.window, shortcuts: str) -> None:
    """Draw bottom footer bar with available keyboard shortcuts."""
    max_y, max_x = stdscr.getmaxyx()
    footer_attr = get_color(1) if safe_has_colors() else curses.A_STANDOUT
    y = max_y - 1
    safe_addstr(stdscr, y, 0, " " * max_x, footer_attr)
    safe_addstr(stdscr, y, 2, shortcuts, footer_attr)


def run_menu(
    stdscr: curses.window,
    title: str,
    subtitle: str,
    options: List[Tuple[str, str]],
    default_idx: int = 0,
) -> Optional[int]:
    """
    Display a navigable menu of options with [Title, Description].
    Returns selected index or None if canceled (ESC or 'q').
    """
    configure_escdelay(25)
    safe_curs_set(0)
    current_idx = default_idx
    max_idx = len(options) - 1

    while True:
        stdscr.erase()
        draw_header(stdscr, title, subtitle)
        max_y, max_x = stdscr.getmaxyx()

        y_start = 3
        safe_addstr(stdscr, y_start, 2, "Use [↑/↓] to Navigate, [Enter] to Select, [q] to Quit", curses.A_DIM)

        list_y = y_start + 2
        for idx, (label, desc) in enumerate(options):
            if list_y + idx * 2 >= max_y - 2:
                break
            is_selected = (idx == current_idx)
            indicator = " ▶ " if is_selected else "   "
            line = f"{indicator}{label}"

            if is_selected:
                attr = (get_color(3) | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT
            else:
                attr = get_color(4) if safe_has_colors() else 0

            safe_addstr(stdscr, list_y + idx * 2, 2, line, attr)
            if desc:
                safe_addstr(stdscr, list_y + idx * 2 + 1, 6, desc, curses.A_DIM)

        draw_footer(stdscr, "[↑/↓] Navigate  |  [Enter] Select  |  [Esc/q] Cancel")
        stdscr.refresh()

        key = stdscr.getch()
        if key in (curses.KEY_UP, ord("k")):
            current_idx = max(0, current_idx - 1)
        elif key in (curses.KEY_DOWN, ord("j")):
            current_idx = min(max_idx, current_idx + 1)
        elif key in (10, 13, curses.KEY_ENTER):
            return current_idx
        elif key in (27, ord("q"), ord("Q")):
            return None


def run_text_input(
    stdscr: curses.window,
    title: str,
    prompt: str,
    default_value: str = "",
    help_text: str = "",
    history: Optional[List[str]] = None,
    gui_picker_type: Optional[str] = None,  # "dataset" or "folder"
) -> Optional[str]:
    """
    Display a text input screen with cursor editing, auto-completion, and default value.
    Supports triggering a native GUI file/folder picker via [Ctrl+O] or [F2].
    """
    from mlx_commander.gui_picker import pick_dataset_gui, pick_folder_gui

    configure_escdelay(25)
    safe_curs_set(1)
    text = list(default_value)
    cursor_pos = len(text)

    while True:
        stdscr.erase()
        draw_header(stdscr, title, prompt)
        max_y, max_x = stdscr.getmaxyx()

        safe_addstr(stdscr, 3, 2, prompt, curses.A_BOLD)
        if help_text:
            safe_addstr(stdscr, 4, 2, help_text, curses.A_DIM)

        # Draw input box
        box_y = 6
        box_width = min(max_x - 6, 80)
        safe_addstr(stdscr, box_y - 1, 2, "┌" + "─" * box_width + "┐", curses.A_DIM)
        safe_addstr(stdscr, box_y, 2, "│" + " " * box_width + "│", curses.A_DIM)
        safe_addstr(stdscr, box_y + 1, 2, "└" + "─" * box_width + "┘", curses.A_DIM)

        # Draw text inside box
        current_str = "".join(text)
        visible_text = current_str
        if len(visible_text) > box_width - 4:
            visible_text = visible_text[-(box_width - 4):]

        safe_addstr(stdscr, box_y, 4, current_str[:box_width - 4], get_color(2) if safe_has_colors() else 0)

        # Draw hints / autocomplete suggestions if path-like
        if "/" in current_str or "~" in current_str or "." in current_str:
            expanded = Path(current_str).expanduser()
            parent = expanded.parent if not current_str.endswith("/") else expanded
            if parent.exists() and parent.is_dir():
                try:
                    candidates = [p.name for p in parent.iterdir() if p.name.startswith(expanded.name or "")][:4]
                    if candidates:
                        safe_addstr(stdscr, box_y + 3, 2, "Suggestions: " + ", ".join(candidates), curses.A_DIM)
                except Exception:
                    pass

        footer_text = "[Enter] Confirm  |  [Tab] Auto-complete  |  [Esc] Cancel"
        if gui_picker_type:
            footer_text = "[Enter] Confirm  |  [Ctrl+O / F2] Open Finder GUI  |  [Tab] Complete  |  [Esc] Cancel"
        draw_footer(stdscr, footer_text)

        # Position cursor
        cursor_x = min(4 + cursor_pos, box_width)
        stdscr.move(box_y, cursor_x)
        stdscr.refresh()

        key = stdscr.getch()
        if key in (10, 13, curses.KEY_ENTER):
            return "".join(text)
        elif key == 27:  # ESC
            return None
        elif gui_picker_type and key in (15, 6, curses.KEY_F2):  # Ctrl+O, Ctrl+F, or F2
            curses.def_prog_mode()
            curses.endwin()
            curr_path = "".join(text).strip() or os.getcwd()
            from mlx_commander.gui_picker import pick_dataset_gui, pick_folder_gui
            if gui_picker_type == "folder":
                chosen = pick_folder_gui("Select Destination Folder", default_dir=curr_path)
            else:
                chosen = pick_dataset_gui(default_dir=curr_path)
            curses.reset_prog_mode()
            stdscr.refresh()
            if chosen:
                text = list(chosen)
                cursor_pos = len(text)
        elif key in (curses.KEY_BACKSPACE, 127, 8):
            if cursor_pos > 0:
                text.pop(cursor_pos - 1)
                cursor_pos -= 1
        elif key == curses.KEY_DC:  # Delete
            if cursor_pos < len(text):
                text.pop(cursor_pos)
        elif key == curses.KEY_LEFT:
            cursor_pos = max(0, cursor_pos - 1)
        elif key == curses.KEY_RIGHT:
            cursor_pos = min(len(text), cursor_pos + 1)
        elif key == curses.KEY_HOME or key == 1:  # Ctrl+A
            cursor_pos = 0
        elif key == curses.KEY_END or key == 5:  # Ctrl+E
            cursor_pos = len(text)
        elif key == 9:  # TAB: simple path completion
            current_str = "".join(text)
            expanded = Path(current_str).expanduser()
            parent = expanded.parent if not current_str.endswith("/") else expanded
            if parent.exists() and parent.is_dir():
                try:
                    prefix = "" if current_str.endswith("/") else expanded.name
                    matches = [p for p in parent.iterdir() if p.name.startswith(prefix)]
                    if len(matches) == 1:
                        completed = str(matches[0]) + ("/" if matches[0].is_dir() else "")
                        text = list(completed)
                        cursor_pos = len(text)
                except Exception:
                    pass
        elif 32 <= key <= 126:
            text.insert(cursor_pos, chr(key))
            cursor_pos += 1


def show_message_dialog(
    stdscr: curses.window,
    title: str,
    message_lines: List[str],
    is_error: bool = False,
) -> None:
    """Display an informational or error modal dialog and wait for any key."""
    safe_curs_set(0)
    stdscr.erase()
    draw_header(stdscr, title, "Notice")
    max_y, max_x = stdscr.getmaxyx()

    attr = (get_color(5) | curses.A_BOLD) if (is_error and safe_has_colors()) else (get_color(3) | curses.A_BOLD)

    safe_addstr(stdscr, 3, 2, "┌" + "─" * (max_x - 6) + "┐", curses.A_DIM)
    for idx, line in enumerate(message_lines):
        if 4 + idx >= max_y - 4:
            break
        safe_addstr(stdscr, 4 + idx, 4, line, attr if idx == 0 and is_error else 0)

    safe_addstr(stdscr, max_y - 4, 2, "└" + "─" * (max_x - 6) + "┘", curses.A_DIM)
    draw_footer(stdscr, "Press any key to continue...")
    stdscr.refresh()
    stdscr.getch()


def draw_box_panel(
    win: curses.window,
    y: int,
    x: int,
    h: int,
    w: int,
    title: str,
    is_focused: bool = False,
    subtitle: str = "",
) -> None:
    """Draw a styled box panel with title and active focus indicator."""
    if h <= 2 or w <= 4:
        return

    border_attr = (get_color(2) | curses.A_BOLD) if (is_focused and safe_has_colors()) else (curses.A_DIM)
    title_attr = (get_color(2) | curses.A_BOLD) if (is_focused and safe_has_colors()) else (curses.A_BOLD)

    prefix = f"┌─ {title} "
    if subtitle:
        prefix += f"[{subtitle}] "
    avail_line = w - len(prefix) - 1
    if avail_line > 0:
        top_str = prefix + "─" * avail_line + "┐"
    else:
        top_str = "┌" + "─" * (w - 2) + "┐"

    safe_addstr(win, y, x, top_str, border_attr)

    for row in range(y + 1, y + h - 1):
        safe_addstr(win, row, x, "│", border_attr)
        safe_addstr(win, row, x + w - 1, "│", border_attr)

    safe_addstr(win, y + h - 1, x, "└" + "─" * (w - 2) + "┘", border_attr)


def draw_button(win: curses.window, y: int, x: int, label: str, is_focused: bool = False) -> None:
    """Draw an interactive button."""
    btn_str = f"[ {label} ]"
    if is_focused:
        attr = (get_color(3) | curses.A_STANDOUT | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT
    else:
        attr = (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD
    safe_addstr(win, y, x, btn_str, attr)


def draw_radio(
    win: curses.window,
    y: int,
    x: int,
    label: str,
    is_checked: bool = False,
    is_focused: bool = False,
) -> None:
    """Draw an interactive radio button option."""
    mark = "●" if is_checked else " "
    radio_str = f"({mark}) {label}"
    if is_focused:
        attr = (get_color(2) | curses.A_STANDOUT | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT
    elif is_checked:
        attr = (get_color(3) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD
    else:
        attr = curses.A_DIM
    safe_addstr(win, y, x, radio_str, attr)


def draw_field(
    win: curses.window,
    y: int,
    x: int,
    label: str,
    val_str: str,
    is_focused: bool = False,
    val_width: int = 24,
    has_dropdown: bool = False,
) -> None:
    """Draw a labeled form field with highlighted active box."""
    lbl = f"{label}: "
    safe_addstr(win, y, x, lbl, curses.A_BOLD)
    val_x = x + len(lbl)

    disp_val = val_str or "<none>"
    arrow = " ▾" if has_dropdown else ""
    max_len = val_width - len(arrow) - 2
    if len(disp_val) > max_len:
        disp_val = disp_val[: max_len - 1] + "…"

    pad = " " * max(0, max_len - len(disp_val))
    box_str = f"[ {disp_val}{pad}{arrow} ]"

    if is_focused:
        attr = (get_color(2) | curses.A_STANDOUT | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT
    else:
        attr = get_color(2) if safe_has_colors() else 0
    safe_addstr(win, y, val_x, box_str, attr)


def show_column_picker_dialog(
    stdscr: curses.window,
    title: str,
    columns: List[str],
    current_val: Optional[str] = None,
    allow_none: bool = False,
) -> Optional[str]:
    """
    Modal dialog overlay to pick or concatenate columns from a scrollable list.
    Supports:
    - [Space]: Toggle column selection. Multiple selections are ordered in the sequence
               they were pressed, displayed with [1], [2], etc., and concatenated with ' + '.
    - [Enter]: Confirm selection.
    - [Esc] / [q]: Cancel and return previous value without delay.
    """
    from mlx_commander.formats import parse_column_list

    configure_escdelay(25)
    options = []
    if allow_none:
        options.append("(None / Skip)")
    options.extend(columns)

    if not options:
        return None

    # Track ordered selected columns
    initial_cols = [c for c in parse_column_list(current_val, columns) if c in columns]
    selected_cols: List[str] = list(initial_cols)
    space_used = False

    sel_idx = 0
    if selected_cols:
        for i, opt in enumerate(options):
            if opt == selected_cols[0]:
                sel_idx = i
                break
    elif current_val:
        for i, opt in enumerate(options):
            if opt == current_val:
                sel_idx = i
                break

    initial_sel_idx = sel_idx

    max_y, max_x = stdscr.getmaxyx()
    h = min(len(options) + 7, max_y - 4, 18)
    longest_col = max((len(c) for c in options), default=10)
    w = min(max(longest_col + 16, 54), max_x - 6)
    start_y = max(1, (max_y - h) // 2)
    start_x = max(1, (max_x - w) // 2)

    scroll_offset = max(0, sel_idx - (h - 7) // 2)
    safe_curs_set(0)

    while True:
        safe_addstr(stdscr, start_y, start_x, "╔" + "═" * (w - 2) + "╗", get_color(2) | curses.A_BOLD)
        safe_addstr(stdscr, start_y, start_x + 2, f" {title} ", get_color(2) | curses.A_BOLD)

        visible_rows = h - 5
        if sel_idx < scroll_offset:
            scroll_offset = sel_idx
        elif sel_idx >= scroll_offset + visible_rows:
            scroll_offset = sel_idx - visible_rows + 1

        for r in range(visible_rows):
            row_y = start_y + 1 + r
            opt_idx = scroll_offset + r
            safe_addstr(stdscr, row_y, start_x, "║" + " " * (w - 2) + "║", get_color(2))
            if opt_idx < len(options):
                opt_name = options[opt_idx]
                is_focused = (opt_idx == sel_idx)
                prefix = " ▶ " if is_focused else "   "

                if opt_name == "(None / Skip)":
                    box = "   "
                elif opt_name in selected_cols:
                    if len(selected_cols) > 1:
                        order_num = selected_cols.index(opt_name) + 1
                        box = f"[{order_num}]"
                    else:
                        box = "[*]"
                else:
                    box = "[ ]"

                line_text = f"{prefix}{box} {opt_name}"[: w - 4]
                if is_focused:
                    attr = (get_color(3) | curses.A_STANDOUT | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT
                elif opt_name in selected_cols:
                    attr = (get_color(3) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD
                else:
                    attr = 0

                safe_addstr(stdscr, row_y, start_x + 1, line_text, attr)

        # Concatenation selection summary
        safe_addstr(stdscr, start_y + h - 3, start_x, "║" + " " * (w - 2) + "║", get_color(2))
        if selected_cols:
            summary_txt = "Selection: " + " + ".join(selected_cols)
            safe_addstr(stdscr, start_y + h - 3, start_x + 2, summary_txt[: w - 4], get_color(3) | curses.A_BOLD)
        else:
            safe_addstr(stdscr, start_y + h - 3, start_x + 2, "Selection: <none>", curses.A_DIM)

        # Footer
        safe_addstr(stdscr, start_y + h - 2, start_x, "║" + " " * (w - 2) + "║", get_color(2))
        safe_addstr(stdscr, start_y + h - 2, start_x + 2, "[Space] Toggle/Order  [Enter] OK  [Esc] Cancel"[: w - 4], curses.A_DIM)
        safe_addstr(stdscr, start_y + h - 1, start_x, "╚" + "═" * (w - 2) + "╝", get_color(2) | curses.A_BOLD)
        stdscr.refresh()

        k = stdscr.getch()
        if k in (curses.KEY_UP, ord("k")):
            sel_idx = max(0, sel_idx - 1)
        elif k in (curses.KEY_DOWN, ord("j")):
            sel_idx = min(len(options) - 1, sel_idx + 1)
        elif k == 32:  # Spacebar: toggle selection
            space_used = True
            chosen = options[sel_idx]
            if chosen == "(None / Skip)":
                selected_cols.clear()
            else:
                if chosen in selected_cols:
                    selected_cols.remove(chosen)
                else:
                    selected_cols.append(chosen)
        elif k in (10, 13, curses.KEY_ENTER):
            if space_used:
                if not selected_cols:
                    if options[sel_idx] == "(None / Skip)":
                        return None
                    return None if allow_none else options[sel_idx]
                elif len(selected_cols) == 1:
                    return selected_cols[0]
                else:
                    return " + ".join(selected_cols)
            else:
                # Space was not pressed: single-select navigation
                chosen = options[sel_idx]
                if allow_none and chosen == "(None / Skip)":
                    return None
                if len(selected_cols) > 1 and sel_idx == initial_sel_idx:
                    return " + ".join(selected_cols)
                return chosen
        elif k in (27, ord("q")):
            return current_val


def show_text_edit_dialog(
    stdscr: curses.window,
    title: str,
    prompt: str,
    default_val: str = "",
    is_number: bool = False,
) -> Optional[str]:
    """Modal dialog overlay to edit a single value in-place."""
    configure_escdelay(25)
    max_y, max_x = stdscr.getmaxyx()
    h = 7
    w = min(max_x - 8, 60)
    start_y = max(1, (max_y - h) // 2)
    start_x = max(1, (max_x - w) // 2)

    safe_curs_set(1)
    text_chars = list(default_val)
    cursor_pos = len(text_chars)

    while True:
        safe_addstr(stdscr, start_y, start_x, "╔" + "═" * (w - 2) + "╗", get_color(2) | curses.A_BOLD)
        safe_addstr(stdscr, start_y, start_x + 2, f" {title} ", get_color(2) | curses.A_BOLD)
        for r in range(1, h - 1):
            safe_addstr(stdscr, start_y + r, start_x, "║" + " " * (w - 2) + "║", get_color(2))
        safe_addstr(stdscr, start_y + h - 1, start_x, "╚" + "═" * (w - 2) + "╝", get_color(2) | curses.A_BOLD)

        safe_addstr(stdscr, start_y + 1, start_x + 2, prompt, curses.A_BOLD)

        val_str = "".join(text_chars)
        box_w = w - 6
        disp = val_str[-(box_w):] if len(val_str) > box_w else val_str
        safe_addstr(stdscr, start_y + 3, start_x + 3, disp + " " * (box_w - len(disp)), get_color(2) | curses.A_STANDOUT)
        safe_addstr(stdscr, start_y + 5, start_x + 2, "[Enter] OK   [Esc] Cancel", curses.A_DIM)

        cursor_x = start_x + 3 + min(cursor_pos, box_w)
        stdscr.move(start_y + 3, cursor_x)
        stdscr.refresh()

        k = stdscr.getch()
        if k in (10, 13, curses.KEY_ENTER):
            safe_curs_set(0)
            return "".join(text_chars)
        elif k == 27:  # ESC
            safe_curs_set(0)
            return None
        elif k in (curses.KEY_BACKSPACE, 127, 8):
            if cursor_pos > 0:
                text_chars.pop(cursor_pos - 1)
                cursor_pos -= 1
        elif k == curses.KEY_DC:
            if cursor_pos < len(text_chars):
                text_chars.pop(cursor_pos)
        elif k == curses.KEY_LEFT:
            cursor_pos = max(0, cursor_pos - 1)
        elif k == curses.KEY_RIGHT:
            cursor_pos = min(len(text_chars), cursor_pos + 1)
        elif 32 <= k <= 126:
            char = chr(k)
            if is_number and char not in "0123456789.":
                continue
            text_chars.insert(cursor_pos, char)
            cursor_pos += 1


def show_results_dialog(stdscr: curses.window, result: Any) -> None:
    """Modal dialog displaying conversion results and mlx_lm.lora command."""
    configure_escdelay(25)
    safe_curs_set(0)
    max_y, max_x = stdscr.getmaxyx()
    h = min(max_y - 4, 20)
    w = min(max_x - 6, 80)
    start_y = max(1, (max_y - h) // 2)
    start_x = max(1, (max_x - w) // 2)

    while True:
        safe_addstr(stdscr, start_y, start_x, "╔" + "═" * (w - 2) + "╗", get_color(3) | curses.A_BOLD)
        safe_addstr(stdscr, start_y, start_x + 2, " [OK] Conversion Successful! ", get_color(3) | curses.A_BOLD)
        for r in range(1, h - 1):
            safe_addstr(stdscr, start_y + r, start_x, "║" + " " * (w - 2) + "║", get_color(3))
        safe_addstr(stdscr, start_y + h - 1, start_x, "╚" + "═" * (w - 2) + "╝", get_color(3) | curses.A_BOLD)

        safe_addstr(stdscr, start_y + 2, start_x + 3, f"Saved dataset to: {result.output_dir}", curses.A_BOLD)
        row = start_y + 4
        for split_name, file_path in result.output_files.items():
            cnt = result.record_counts.get(split_name, 0)
            sz = result.file_sizes.get(split_name, 0) / 1024.0
            safe_addstr(stdscr, row, start_x + 3, f"• {file_path.name}: {cnt:,} records ({sz:.1f} KB)", curses.A_DIM)
            row += 1

        row += 1
        safe_addstr(stdscr, row, start_x + 3, "MLX Fine-tuning Command:", get_color(2) | curses.A_BOLD)
        row += 1
        cmd_lines = result.generate_mlx_lora_command().strip().split("\n")
        for line in cmd_lines[:6]:
            if row < start_y + h - 2:
                safe_addstr(stdscr, row, start_x + 5, line[: w - 8], get_color(2))
                row += 1

        safe_addstr(stdscr, start_y + h - 2, start_x + 3, "Press [Enter], [Space], or [Esc] to return", curses.A_STANDOUT)
        stdscr.refresh()

        k = stdscr.getch()
        if k in (10, 13, 32, 27, ord("q"), ord("Q")):
            break


def show_help_dialog(stdscr: curses.window) -> None:
    """Modal dialog displaying MLX-Commander shortcut help."""
    configure_escdelay(25)
    safe_curs_set(0)
    max_y, max_x = stdscr.getmaxyx()
    h = min(max_y - 4, 18)
    w = min(max_x - 6, 76)
    start_y = max(1, (max_y - h) // 2)
    start_x = max(1, (max_x - w) // 2)

    shortcuts = [
        ("Tab / Shift-Tab", "Switch focus between Left (Dataset) and Right (Config) panels"),
        ("↑ / ↓ (or k / j)", "Navigate through fields, options, and buttons in active panel"),
        ("← / → (or h / l)", "Toggle MLX format radio options"),
        ("Enter / Space", "Open column dropdown, edit field value, or trigger button"),
        ("F2", "Open native macOS Finder upload / dataset picker"),
        ("F5", "Run conversion and write train/valid/test JSONL files"),
        ("r / R", "Randomize split seed"),
        ("? / F1", "Show this help screen"),
        ("F10 / q / Esc", "Exit MLX-Commander"),
    ]

    while True:
        safe_addstr(stdscr, start_y, start_x, "╔" + "═" * (w - 2) + "╗", get_color(2) | curses.A_BOLD)
        safe_addstr(stdscr, start_y, start_x + 2, " MLX-Commander Keyboard Shortcuts ", get_color(2) | curses.A_BOLD)
        for r in range(1, h - 1):
            safe_addstr(stdscr, start_y + r, start_x, "║" + " " * (w - 2) + "║", get_color(2))
        safe_addstr(stdscr, start_y + h - 1, start_x, "╚" + "═" * (w - 2) + "╝", get_color(2) | curses.A_BOLD)

        for i, (key, desc) in enumerate(shortcuts):
            row = start_y + 2 + i
            if row < start_y + h - 2:
                safe_addstr(stdscr, row, start_x + 3, f"{key:<20}", get_color(3) | curses.A_BOLD)
                safe_addstr(stdscr, row, start_x + 24, desc[: w - 27], curses.A_DIM)

        safe_addstr(stdscr, start_y + h - 2, start_x + 3, "Press [Enter], [Space], or [Esc] to close", curses.A_STANDOUT)
        stdscr.refresh()

        k = stdscr.getch()
        if k in (10, 13, 32, 27, ord("q"), ord("Q")):
            break


def get_schema_mapping_targets(
    target_format: Any,
    mapping: Any,
    available_columns: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Extract structured target fields and unmapped source columns.
    Returns:
      (targets, unmapped_columns)
    Each target is:
      {
        "key": str,          # e.g. "prompt", "completion", "chosen", etc.
        "label": str,        # display label
        "type_str": str,     # e.g. "string", "string/dict", "list[dict]"
        "required": bool,    # whether this field is required
        "cols": List[str],   # mapped source columns (parsed from col_spec)
        "is_concat": bool,   # True if 2 or more columns are combined
      }
    """
    from mlx_commander.formats import MLXFormat, parse_column_list

    avail = available_columns or []
    targets: List[Dict[str, Any]] = []

    if target_format == MLXFormat.PROMPT_COMPLETION:
        prompt_spec = getattr(mapping, "prompt_col", None)
        comp_spec = getattr(mapping, "completion_col", None)
        p_cols = parse_column_list(prompt_spec, avail) if prompt_spec else []
        c_cols = parse_column_list(comp_spec, avail) if comp_spec else []
        targets.append({
            "key": "prompt",
            "label": "prompt",
            "type_str": "string",
            "required": True,
            "cols": p_cols,
            "is_concat": len(p_cols) > 1,
        })
        targets.append({
            "key": "completion",
            "label": "completion",
            "type_str": "string",
            "required": True,
            "cols": c_cols,
            "is_concat": len(c_cols) > 1,
        })

    elif target_format == MLXFormat.CHAT:
        messages_col = getattr(mapping, "messages_col", None)
        if messages_col:
            m_cols = parse_column_list(messages_col, avail) if messages_col else []
            targets.append({
                "key": "messages",
                "label": "messages",
                "type_str": "list[dict]",
                "required": True,
                "cols": m_cols,
                "is_concat": len(m_cols) > 1,
            })
        else:
            u_cols = parse_column_list(getattr(mapping, "user_col", None), avail)
            a_cols = parse_column_list(getattr(mapping, "assistant_col", None), avail)
            s_cols = parse_column_list(getattr(mapping, "system_col", None), avail)
            targets.append({
                "key": "user",
                "label": "user (turn)",
                "type_str": "string",
                "required": True,
                "cols": u_cols,
                "is_concat": len(u_cols) > 1,
            })
            targets.append({
                "key": "assistant",
                "label": "assistant (turn)",
                "type_str": "string",
                "required": True,
                "cols": a_cols,
                "is_concat": len(a_cols) > 1,
            })
            targets.append({
                "key": "system",
                "label": "system (prompt)",
                "type_str": "string (opt)",
                "required": False,
                "cols": s_cols,
                "is_concat": len(s_cols) > 1,
            })

    elif target_format == MLXFormat.DPO:
        p_raw = getattr(mapping, "dpo_prompt_col", None) or getattr(mapping, "prompt_col", None)
        p_cols = parse_column_list(p_raw, avail) if p_raw else []
        c_cols = parse_column_list(getattr(mapping, "chosen_col", None), avail)
        r_cols = parse_column_list(getattr(mapping, "rejected_col", None), avail)
        targets.append({
            "key": "prompt",
            "label": "prompt",
            "type_str": "string/dict",
            "required": True,
            "cols": p_cols,
            "is_concat": len(p_cols) > 1,
        })
        targets.append({
            "key": "chosen",
            "label": "chosen (pref)",
            "type_str": "string/dict",
            "required": True,
            "cols": c_cols,
            "is_concat": len(c_cols) > 1,
        })
        targets.append({
            "key": "rejected",
            "label": "rejected (disp)",
            "type_str": "string/dict",
            "required": True,
            "cols": r_cols,
            "is_concat": len(r_cols) > 1,
        })

    elif target_format == MLXFormat.TEXT:
        t_raw = getattr(mapping, "text_col", None)
        t_cols = parse_column_list(t_raw, avail) if t_raw else []
        targets.append({
            "key": "text",
            "label": "text",
            "type_str": "string",
            "required": True,
            "cols": t_cols,
            "is_concat": len(t_cols) > 1,
        })

    # Unmapped columns: original dataset columns that are not assigned to any target
    mapped_set = set()
    for t in targets:
        for c in t["cols"]:
            mapped_set.add(c)
    unmapped = [c for c in avail if c not in mapped_set]

    return targets, unmapped


def plan_mapping_panel_rows(
    targets: List[Dict[str, Any]],
    unmapped: List[str],
    avail_rows: int,
) -> Tuple[List[int], bool]:
    """
    Determine row allocation for each target and whether to show unmapped columns.
    Guarantees total rendered lines fits within avail_rows.
    """
    n = len(targets)
    if n == 0 or avail_rows < 3:
        return ([1] * max(1, n), False)

    min_needed = 2 + (n - 1) + n  # 1 top border + 1 bottom border + (n - 1) dividers + n target rows
    if avail_rows < min_needed:
        return ([1] * n, False)

    show_unmapped = False
    if unmapped and avail_rows >= min_needed + 2:
        show_unmapped = True

    used = min_needed + (2 if show_unmapped else 0)
    extra = avail_rows - used
    heights = [1] * n

    for i, t in enumerate(targets):
        if extra > 0 and len(t["cols"]) > 1:
            heights[i] += 1
            extra -= 1

    return heights, show_unmapped


def draw_mapping_pipeline_panel(
    win: curses.window,
    y: int,
    x: int,
    h: int,
    w: int,
    state: Any,
) -> None:
    """
    Draw the visual column mapping pipeline panel with:
    - Bundled original columns reordered per MLX target field
    - Matching jointed divider blocks on both source and target sides
    - Clear flow routing joints connecting source bundles to MLX target blocks
    - Concat badges and bracket tree joints for multi-column mappings
    - Clean typography with zero emojis
    """
    if h < 4 or w < 50:
        return

    from mlx_commander.formats import MLXFormat

    target_fmt = getattr(state, "target_format", MLXFormat.PROMPT_COMPLETION)
    mapping = getattr(state, "mapping", None)
    loaded_ds = getattr(state, "loaded_dataset", None)
    avail_cols = loaded_ds.columns if loaded_ds else []

    targets, unmapped = get_schema_mapping_targets(target_fmt, mapping, avail_cols)

    # 1. Outer Box Panel
    draw_box_panel(
        win,
        y,
        x,
        h,
        w,
        "Column Mapping Pipeline (Source -> Target MLX Schema)",
        is_focused=False,
        subtitle=target_fmt.value.upper(),
    )

    # 2. Dimensions and Header Row
    box_w = max(22, min(28, (w - 24) // 2))
    left_x = x + 2
    right_x = x + w - box_w - 2
    flow_x = left_x + box_w
    flow_w = right_x - flow_x

    header_y = y + 1
    safe_addstr(win, header_y, left_x + 1, "ORIGINAL COLUMNS (BUNDLED)", curses.A_BOLD | get_color(4))
    flow_title = "FLOW JOINTS"
    safe_addstr(win, header_y, flow_x + max(0, (flow_w - len(flow_title)) // 2), flow_title, curses.A_DIM | get_color(2))
    safe_addstr(win, header_y, right_x + 1, "TARGET MLX SCHEMA", curses.A_BOLD | get_color(4))

    # 3. Row Allocation
    start_y = y + 2
    avail_rows = h - 3
    if avail_rows < 2:
        return

    n = len(targets)
    heights, show_unmapped = plan_mapping_panel_rows(targets, unmapped, avail_rows)

    border_attr = get_color(2) if safe_has_colors() else curses.A_DIM
    header_box_attr = (get_color(2) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD
    green_attr = (get_color(3) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD
    cyan_attr = (get_color(2) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD
    yellow_attr = (get_color(6) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD
    red_attr = (get_color(5) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD
    dim_attr = curses.A_DIM

    curr_y = start_y

    def fmt_border(left_char: str, right_char: str, title: str, width: int) -> str:
        if title:
            prefix = f"{left_char}─ {title} "
            rem = width - len(prefix) - 1
            if rem >= 0:
                return prefix + "─" * rem + right_char
        return left_char + "─" * (width - 2) + right_char

    # Top Border of Boxes
    if targets and curr_y < y + h - 1:
        t0 = targets[0]
        left_top = fmt_border("┌", "┐", f"Source: {t0['key']}", box_w)
        right_top = fmt_border("┌", "┐", f"Target: {t0['key']}", box_w)
        safe_addstr(win, curr_y, left_x, left_top, header_box_attr)
        safe_addstr(win, curr_y, right_x, right_top, header_box_attr)
        curr_y += 1

    # Render each target block
    for i, t in enumerate(targets):
        cols = t["cols"]
        b_h = heights[i]

        for r in range(b_h):
            if curr_y >= y + h - 1:
                break

            # Left Box Content
            if not loaded_ds:
                l_text = " (No dataset loaded)" if r == 0 else ""
                l_attr = dim_attr
            elif not cols:
                l_text = " ! (none selected)" if r == 0 else ""
                l_attr = red_attr if t["required"] else dim_attr
            elif len(cols) == 1:
                l_text = f" • {cols[0]}" if r == 0 else "   [1 column]"
                l_attr = green_attr if r == 0 else dim_attr
            elif len(cols) >= 2:
                if b_h >= 2:
                    c_name = cols[r] if r < len(cols) else cols[-1]
                    l_text = f" • {c_name}"
                    l_attr = cyan_attr
                else:
                    c_joined = " + ".join(cols)
                    l_text = f" • {c_joined}"
                    l_attr = cyan_attr

            l_padded = f"│ {l_text:<{box_w - 4}} │"
            safe_addstr(win, curr_y, left_x, l_padded, border_attr)
            if l_text:
                safe_addstr(win, curr_y, left_x + 2, l_text[:box_w - 4], l_attr)

            # Right Box Content
            if not loaded_ds:
                r_text = f" ○ {t['key']} (awaiting)" if r == 0 else ""
                r_attr = dim_attr
            elif not cols:
                r_status = "missing" if t["required"] else "optional"
                r_text = f" ○ {t['key']} ({r_status})" if r == 0 else ""
                r_attr = red_attr if t["required"] else dim_attr
            else:
                if r == 0:
                    r_text = f" ● {t['key']}"
                    r_attr = green_attr
                else:
                    r_text = f"   Type: {t['type_str']}"
                    r_attr = dim_attr

            r_padded = f"│ {r_text:<{box_w - 4}} │"
            safe_addstr(win, curr_y, right_x, r_padded, border_attr)
            if r_text:
                safe_addstr(win, curr_y, right_x + 2, r_text[:box_w - 4], r_attr)

            # Center Flow Joints
            if not loaded_ds:
                msg = "(awaiting dataset)"
                rem_flow = flow_w - len(msg) - 8
                dash1 = max(1, rem_flow // 2)
                dash2 = max(1, rem_flow - dash1)
                c_flow = f" {'- ' * (dash1 // 2)}{msg}{' -' * (dash2 // 2)}► "
                safe_addstr(win, curr_y, flow_x, c_flow[:flow_w], dim_attr)
            elif not cols:
                c_flow = f" {'- ' * max(1, flow_w // 2 - 2)}► "
                safe_addstr(win, curr_y, flow_x, c_flow[:flow_w], red_attr if t["required"] else dim_attr)
            elif len(cols) == 1:
                bar_len = max(2, flow_w - 4)
                c_flow = f" {'─' * bar_len}► "
                safe_addstr(win, curr_y, flow_x, c_flow[:flow_w], green_attr)
            elif len(cols) >= 2:
                badge = "[ + Concat ]"
                lead = " ──┴──► "
                lead_len = len(lead)
                badge_len = len(badge)
                bar_len = max(2, flow_w - lead_len - badge_len - 3)
                tail = f" {'─' * bar_len}► "

                if b_h >= 2 and r == 0:
                    c_flow = " ──╮"
                    safe_addstr(win, curr_y, flow_x, c_flow, cyan_attr)
                else:
                    safe_addstr(win, curr_y, flow_x, lead, cyan_attr)
                    safe_addstr(win, curr_y, flow_x + lead_len, badge, cyan_attr | curses.A_STANDOUT)
                    safe_addstr(win, curr_y, flow_x + lead_len + badge_len, tail, green_attr)

            curr_y += 1

        # Divider joint between target blocks
        if curr_y < y + h - 1:
            if i < n - 1:
                next_t = targets[i + 1]
                left_div = fmt_border("├", "┤", f"Source: {next_t['key']}", box_w)
                right_div = fmt_border("├", "┤", f"Target: {next_t['key']}", box_w)
                safe_addstr(win, curr_y, left_x, left_div, header_box_attr)
                safe_addstr(win, curr_y, right_x, right_div, header_box_attr)
                curr_y += 1

    # Close Right Box
    if curr_y < y + h - 1:
        right_bot = fmt_border("└", "┘", "", box_w)
        safe_addstr(win, curr_y, right_x, right_bot, border_attr)

    # Unmapped columns section
    if show_unmapped and curr_y < y + h - 2:
        left_unmap_div = fmt_border("├", "┤", "Unmapped Columns", box_w)
        safe_addstr(win, curr_y, left_x, left_unmap_div, header_box_attr)
        curr_y += 1

        if curr_y < y + h - 1:
            unmapped_str = " · " + ", ".join(unmapped)
            avail_unmap = box_w - 4
            if len(unmapped_str) > avail_unmap:
                unmapped_str = unmapped_str[:avail_unmap - 1] + "…"
            l_unmap_padded = f"│ {unmapped_str:<{box_w - 4}} │"
            safe_addstr(win, curr_y, left_x, l_unmap_padded, border_attr)
            safe_addstr(win, curr_y, left_x + 2, unmapped_str, yellow_attr | dim_attr)

            unmap_flow = " - - - (ignored / excluded)"
            safe_addstr(win, curr_y, flow_x, unmap_flow[:flow_w], dim_attr)
            curr_y += 1

    # Close Left Box
    if curr_y < y + h:
        left_bot = fmt_border("└", "┘", "", box_w)
        safe_addstr(win, curr_y, left_x, left_bot, border_attr)

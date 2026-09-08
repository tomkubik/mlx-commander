"""
Curses TUI Widgets and UI Helpers.
Includes menus, text input fields, panels, and styled dialogs.
"""

import curses
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Semantic color pair constants
COLOR_BANNER = 1         # Top Header Banner
COLOR_BORDER_JOINTS = 2  # Borders, Frame, and Routing Joints (Highlight / Cyan #00AAAA)
COLOR_SUCCESS = 3        # Success, Checks
COLOR_NORMAL_TEXT = 4    # Main Text / White #FFFFFF (Standard file names and UI text)
COLOR_ERROR = 5          # Error, Alerts
COLOR_TITLE_ACCENT = 6   # Panel Titles / Accents (Cursor / Yellow #FFFF55)
COLOR_LABEL_GRAY = 7     # Label / Light Gray #AAAAAA (Inactive elements, background text)
COLOR_INPUT_NORMAL = 8   # User Inputs / Inactive Fields (Ice Blue #55FFFF on Background Blue #0000AA)
COLOR_INPUT_FOCUSED = 9  # Active / Focused Field (Prompt / Black on Highlight / Cyan)
COLOR_FN_NUMBER = 10     # Norton Hotkey Bar: Key Number (White on Black)
COLOR_FN_LABEL = 11      # Norton Hotkey Bar: Key Label (Black on Cyan)
COLOR_PANEL_BG = 12      # Background Blue Fill #0000AA (Classic VGA Blue)


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
    safe_addstr(stdscr, 0, 2, f"MLX Commander  ::  {title}", header_attr)
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
        if key in (curses.KEY_UP, curses.KEY_LEFT, ord("k"), ord("h")):
            current_idx = max(0, current_idx - 1)
        elif key in (curses.KEY_DOWN, curses.KEY_RIGHT, ord("j"), ord("l")):
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


def show_error_dialog(
    stdscr: curses.window,
    title: str,
    message: Any,
) -> None:
    """
    Display a centered modal error dialog overlay on top of MLX Commander.
    Preserves the background dashboard without erasing the screen.
    """
    import textwrap
    configure_escdelay(25)
    safe_curs_set(0)

    max_y, max_x = stdscr.getmaxyx()

    if isinstance(message, str):
        raw_lines = [l for l in message.split("\n")]
    elif isinstance(message, (list, tuple)):
        raw_lines = [str(m) for m in message]
    else:
        raw_lines = [str(message)]

    w = min(max_x - 6, 76)
    w = max(40, w)
    inner_w = w - 6

    wrapped_lines: List[str] = []
    for line in raw_lines:
        if not line.strip():
            wrapped_lines.append("")
            continue
        wrapped = textwrap.wrap(line, width=inner_w, break_long_words=True)
        wrapped_lines.extend(wrapped if wrapped else [""])

    max_text_lines = max(1, min(len(wrapped_lines), max_y - 8))
    h = max_text_lines + 5

    start_y = max(1, (max_y - h) // 2)
    start_x = max(1, (max_x - w) // 2)

    border_attr = (get_color(5) | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT
    title_attr = (get_color(5) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD
    text_attr = (get_color(4) | curses.A_BOLD) if safe_has_colors() else 0

    while True:
        safe_addstr(stdscr, start_y, start_x, "╔" + "═" * (w - 2) + "╗", border_attr)
        title_str = f" [!] {title} "[: w - 4]
        safe_addstr(stdscr, start_y, start_x + 2, title_str, title_attr)

        for r in range(1, h - 1):
            safe_addstr(stdscr, start_y + r, start_x, "║" + " " * (w - 2) + "║", border_attr)

        safe_addstr(stdscr, start_y + h - 1, start_x, "╚" + "═" * (w - 2) + "╝", border_attr)

        for idx in range(max_text_lines):
            safe_addstr(stdscr, start_y + 2 + idx, start_x + 3, wrapped_lines[idx][:inner_w], text_attr)

        hint = "[Enter] OK   [Esc] Dismiss"
        safe_addstr(stdscr, start_y + h - 2, start_x + 3, hint, (get_color(2) | curses.A_STANDOUT | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT)

        stdscr.refresh()

        k = stdscr.getch()
        if k in (10, 13, 32, 27, ord("q"), ord("Q")):
            break


def show_missing_dependency_dialog(
    stdscr: curses.window,
    exc: Any,
) -> None:
    """
    Display a centered modal interstitial dialog when an unsupported dataset format
    requires an uninstalled optional dependency (such as pyarrow, duckdb, or pylance).
    Preserves the background dashboard without erasing the screen.
    Inactive labels and titles are displayed in white; active interactive elements in teal.
    """
    import textwrap
    configure_escdelay(25)
    safe_curs_set(0)

    max_y, max_x = stdscr.getmaxyx()

    format_name = getattr(exc, "format_name", "Required Format")
    package_name = getattr(exc, "package_name", "package")
    install_command = getattr(exc, "install_command", f"pip install {package_name}")
    extra_name = getattr(exc, "extra_name", package_name.replace("py", ""))
    description = getattr(exc, "description", "")

    w = min(max_x - 4, 76)
    w = max(48, w)
    inner_w = w - 6

    raw_lines = [
        f"The '{format_name}' format requires the '{package_name}' package.",
    ]
    if description:
        raw_lines.append(description)
    raw_lines.append("")
    raw_lines.append("To enable support for this format, install the package:")

    wrapped_lines: List[str] = []
    for line in raw_lines:
        if not line.strip():
            wrapped_lines.append("")
            continue
        wrapped = textwrap.wrap(line, width=inner_w, break_long_words=True)
        wrapped_lines.extend(wrapped if wrapped else [""])

    extra_cmd = f"pip install 'mlx_commander[{extra_name}]'"
    cmd_box_width = min(inner_w, max(len(install_command) + 6, len(extra_cmd) + 6, 38))

    h = len(wrapped_lines) + 3 + 4 + 3
    h = min(h, max_y - 4)

    start_y = max(1, (max_y - h) // 2)
    start_x = max(1, (max_x - w) // 2)

    border_attr = (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT
    title_attr = (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD
    label_attr = get_color(4) if safe_has_colors() else 0
    active_teal = (get_color(2) | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT
    action_attr = (get_color(2) | curses.A_STANDOUT | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT

    while True:
        safe_addstr(stdscr, start_y, start_x, "╔" + "═" * (w - 2) + "╗", border_attr)
        title_str = f" [!] Dependency Required: {format_name} "[: w - 4]
        safe_addstr(stdscr, start_y, start_x + 2, title_str, title_attr)

        for r in range(1, h - 1):
            safe_addstr(stdscr, start_y + r, start_x, "║" + " " * (w - 2) + "║", border_attr)

        safe_addstr(stdscr, start_y + h - 1, start_x, "╚" + "═" * (w - 2) + "╝", border_attr)

        curr_y = start_y + 2
        for l in wrapped_lines:
            if curr_y >= start_y + h - 6:
                break
            safe_addstr(stdscr, curr_y, start_x + 3, l[:inner_w], label_attr)
            curr_y += 1

        curr_y += 1
        if curr_y < start_y + h - 4:
            safe_addstr(stdscr, curr_y, start_x + 4, "┌" + "─" * (cmd_box_width - 2) + "┐", active_teal)
            cmd_text = f"  {install_command}".ljust(cmd_box_width - 2)
            safe_addstr(stdscr, curr_y + 1, start_x + 4, f"│{cmd_text}│", active_teal)
            safe_addstr(stdscr, curr_y + 2, start_x + 4, "└" + "─" * (cmd_box_width - 2) + "┘", active_teal)
            curr_y += 3

        if curr_y < start_y + h - 2:
            hint_str = f"Or install optional extra: {extra_cmd}"
            safe_addstr(stdscr, curr_y, start_x + 4, hint_str[:inner_w], label_attr)

        hint = "[I] Install with pip   [Enter] OK / Dismiss"
        safe_addstr(stdscr, start_y + h - 2, start_x + 3, hint, action_attr)

        stdscr.refresh()

        k = stdscr.getch()
        if k in (ord("i"), ord("I")):
            installing_str = f"Installing {package_name} via pip...".ljust(w - 6)
            safe_addstr(stdscr, start_y + h - 2, start_x + 3, installing_str[: w - 6], title_attr)
            stdscr.refresh()
            import subprocess
            import sys
            try:
                res = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package_name],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if res.returncode == 0:
                    ok_str = f"Installed {package_name} successfully!".ljust(w - 6)
                    safe_addstr(stdscr, start_y + h - 2, start_x + 3, ok_str[: w - 6], active_teal)
                    stdscr.refresh()
                    curses.napms(800)
                    return True
                else:
                    err_hint = f"pip install failed (code {res.returncode}). Press any key."
                    safe_addstr(stdscr, start_y + h - 2, start_x + 3, err_hint[: w - 6], action_attr)
                    stdscr.refresh()
                    stdscr.getch()
            except Exception as pe:
                err_hint = f"Error: {pe}"[: w - 6]
                safe_addstr(stdscr, start_y + h - 2, start_x + 3, err_hint, action_attr)
                stdscr.refresh()
                stdscr.getch()
            return False
        elif k in (10, 13, 32, 27, ord("q"), ord("Q")):
            return False
    return False


def show_message_dialog(
    stdscr: curses.window,
    title: str,
    message_lines: List[str],
    is_error: bool = False,
) -> None:
    """Display an informational or error modal dialog overlay on top of the screen."""
    if is_error:
        show_error_dialog(stdscr, title, message_lines)
        return

    import textwrap
    configure_escdelay(25)
    safe_curs_set(0)

    max_y, max_x = stdscr.getmaxyx()
    w = min(max_x - 6, 76)
    w = max(40, w)
    inner_w = w - 6

    wrapped_lines: List[str] = []
    for line in message_lines:
        wrapped = textwrap.wrap(str(line), width=inner_w, break_long_words=True)
        wrapped_lines.extend(wrapped if wrapped else [""])

    max_text_lines = max(1, min(len(wrapped_lines), max_y - 8))
    h = max_text_lines + 5

    start_y = max(1, (max_y - h) // 2)
    start_x = max(1, (max_x - w) // 2)

    border_attr = (get_color(2) | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT
    title_attr = (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD
    text_attr = get_color(4) if safe_has_colors() else 0

    while True:
        safe_addstr(stdscr, start_y, start_x, "╔" + "═" * (w - 2) + "╗", border_attr)
        title_str = f" {title} "[: w - 4]
        safe_addstr(stdscr, start_y, start_x + 2, title_str, title_attr)

        for r in range(1, h - 1):
            safe_addstr(stdscr, start_y + r, start_x, "║" + " " * (w - 2) + "║", border_attr)

        safe_addstr(stdscr, start_y + h - 1, start_x, "╚" + "═" * (w - 2) + "╝", border_attr)

        for idx in range(max_text_lines):
            safe_addstr(stdscr, start_y + 2 + idx, start_x + 3, wrapped_lines[idx][:inner_w], text_attr)

        hint = "[Enter] OK   [Esc] Dismiss"
        safe_addstr(stdscr, start_y + h - 2, start_x + 3, hint, (get_color(2) | curses.A_STANDOUT | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT)

        stdscr.refresh()

        k = stdscr.getch()
        if k in (10, 13, 32, 27, ord("q"), ord("Q")):
            break


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
    """Draw a styled box panel with title and active focus border."""
    if h <= 2 or w <= 4:
        return

    border_attr = (get_color(COLOR_BORDER_JOINTS) | curses.A_BOLD) if (is_focused and safe_has_colors()) else get_color(COLOR_BORDER_JOINTS)
    title_attr = (get_color(COLOR_TITLE_ACCENT) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD

    safe_addstr(win, y, x, "┌─ ", border_attr)
    title_text = f"{title} "
    safe_addstr(win, y, x + 3, title_text, title_attr)
    curr_x = x + 3 + len(title_text)
    if subtitle:
        sub_text = f"[{subtitle}] "
        safe_addstr(win, y, curr_x, sub_text, (get_color(COLOR_LABEL_GRAY) | curses.A_DIM) if safe_has_colors() else curses.A_DIM)
        curr_x += len(sub_text)
    avail_line = max(0, (x + w - 1) - curr_x)
    if avail_line > 0:
        safe_addstr(win, y, curr_x, "─" * avail_line + "┐", border_attr)
    else:
        safe_addstr(win, y, x + w - 1, "┐", border_attr)

    for row in range(y + 1, y + h - 1):
        safe_addstr(win, row, x, "│", border_attr)
        safe_addstr(win, row, x + w - 1, "│", border_attr)

    safe_addstr(win, y + h - 1, x, "└" + "─" * (w - 2) + "┘", border_attr)


def draw_button(win: curses.window, y: int, x: int, label: str, is_focused: bool = False) -> None:
    """Draw an interactive button with deep blue background and active white font."""
    btn_str = f"[ {label} ]"
    if is_focused:
        attr = (get_color(COLOR_INPUT_FOCUSED) | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT
    else:
        attr = get_color(COLOR_INPUT_NORMAL) if safe_has_colors() else 0
    safe_addstr(win, y, x, btn_str, attr)


def draw_radio(
    win: curses.window,
    y: int,
    x: int,
    label: str,
    is_checked: bool = False,
    is_focused: bool = False,
) -> None:
    """Draw an interactive radio button option with input field color or active focus."""
    mark = "●" if is_checked else " "
    radio_str = f"({mark}) {label}"
    if is_focused:
        attr = (get_color(COLOR_INPUT_FOCUSED) | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT
    elif is_checked:
        attr = (get_color(COLOR_INPUT_NORMAL) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD
    else:
        attr = get_color(COLOR_INPUT_NORMAL) if safe_has_colors() else 0
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
    right_edge: Optional[int] = None,
    field_x: Optional[int] = None,
    lbl_attr: Optional[int] = None,
) -> None:
    """Draw a labeled form field with deep blue background and active bright white font.

    If right_edge is specified, aligns the field box so its right bracket is at right_edge,
    while keeping the label aligned at x (the left side).
    If field_x is specified, renders the box starting explicitly at field_x.
    If lbl_attr is specified, overrides the default bold gray label styling.
    """
    lbl = f"{label}: "
    used_lbl_attr = (
        lbl_attr
        if lbl_attr is not None
        else ((get_color(COLOR_LABEL_GRAY) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
    )
    safe_addstr(win, y, x, lbl, used_lbl_attr)

    if right_edge is not None and field_x is None:
        avail_for_box = right_edge - (x + len(lbl)) + 1
        if avail_for_box > 4 and val_width + 2 > avail_for_box:
            val_width = max(4, avail_for_box - 2)

    disp_val = val_str or "<none>"
    arrow = " ▾" if has_dropdown else ""
    max_len = val_width - len(arrow) - 2
    if len(disp_val) > max_len:
        if "/" in disp_val or "\\" in disp_val:
            disp_val = "…" + disp_val[-(max_len - 1):]
        else:
            disp_val = disp_val[: max_len - 1] + "…"

    pad = " " * max(0, max_len - len(disp_val))
    box_str = f"[ {disp_val}{pad}{arrow} ]"

    if field_x is not None:
        val_x = field_x
    elif right_edge is not None:
        box_len = len(box_str)
        min_val_x = x + len(lbl)
        val_x = max(min_val_x, right_edge - box_len + 1)
    else:
        val_x = x + len(lbl)

    if is_focused:
        attr = (get_color(COLOR_INPUT_FOCUSED) | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT
    else:
        attr = get_color(COLOR_INPUT_NORMAL) if safe_has_colors() else 0
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
        safe_addstr(stdscr, start_y, start_x + 2, f" {title} ", (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)

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
                    attr = (get_color(2) | curses.A_STANDOUT | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT
                elif opt_name in selected_cols:
                    attr = (get_color(2) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD
                else:
                    attr = get_color(2) if safe_has_colors() else 0

                safe_addstr(stdscr, row_y, start_x + 1, line_text, attr)

        # Concatenation selection summary
        safe_addstr(stdscr, start_y + h - 3, start_x, "║" + " " * (w - 2) + "║", get_color(2))
        safe_addstr(stdscr, start_y + h - 3, start_x + 2, "Selection: ", (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        if selected_cols:
            summary_txt = " + ".join(selected_cols)
            safe_addstr(stdscr, start_y + h - 3, start_x + 13, summary_txt[: w - 15], (get_color(2) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        else:
            safe_addstr(stdscr, start_y + h - 3, start_x + 13, "<none>", (get_color(4) | curses.A_DIM) if safe_has_colors() else curses.A_DIM)

        # Footer
        safe_addstr(stdscr, start_y + h - 2, start_x, "║" + " " * (w - 2) + "║", get_color(2))
        safe_addstr(stdscr, start_y + h - 2, start_x + 2, "[Space] Toggle/Order  [Enter] OK  [Esc] Cancel"[: w - 4], (get_color(4) | curses.A_DIM) if safe_has_colors() else curses.A_DIM)
        safe_addstr(stdscr, start_y + h - 1, start_x, "╚" + "═" * (w - 2) + "╝", get_color(2) | curses.A_BOLD)
        stdscr.refresh()

        k = stdscr.getch()
        if k in (curses.KEY_UP, curses.KEY_LEFT, ord("k"), ord("h")):
            sel_idx = max(0, sel_idx - 1)
        elif k in (curses.KEY_DOWN, curses.KEY_RIGHT, ord("j"), ord("l")):
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
        safe_addstr(stdscr, start_y, start_x + 2, f" {title} ", (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        for r in range(1, h - 1):
            safe_addstr(stdscr, start_y + r, start_x, "║" + " " * (w - 2) + "║", get_color(2))
        safe_addstr(stdscr, start_y + h - 1, start_x, "╚" + "═" * (w - 2) + "╝", get_color(2) | curses.A_BOLD)

        safe_addstr(stdscr, start_y + 1, start_x + 2, prompt, (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)

        val_str = "".join(text_chars)
        box_w = w - 6
        disp = val_str[-(box_w):] if len(val_str) > box_w else val_str
        safe_addstr(stdscr, start_y + 3, start_x + 3, disp + " " * (box_w - len(disp)), (get_color(2) | curses.A_STANDOUT) if safe_has_colors() else curses.A_STANDOUT)
        safe_addstr(stdscr, start_y + 5, start_x + 2, "[Enter] OK   [Esc] Cancel", (get_color(4) | curses.A_DIM) if safe_has_colors() else curses.A_DIM)

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


def show_output_destination_dialog(
    stdscr: curses.window,
    current_output: str,
    default_output: str,
) -> Optional[str]:
    """
    Dialog allowing user to choose how to specify the destination output folder:
      1) Open macOS Finder (GUI Folder Picker) [if macOS]
      2) Enter folder path manually
      3) Reset to default (<dataset_dir>/mlx_dataset)
    Returns:
      Selected path string, or None if cancelled with Esc.
    """
    from mlx_commander.gui_picker import is_macos, pick_folder_gui

    options: List[Tuple[str, str]] = []
    if is_macos():
        options.append(("Finder (GUI Folder Picker)", "Open native macOS Finder to select destination folder"))
    options.append(("Manual Path Entry", "Type or paste custom destination directory path"))
    options.append(("Reset to Default", f"Use default: {default_output}"))

    sel_idx = 0
    max_y, max_x = stdscr.getmaxyx()
    h = min(12, max_y - 4)
    w = min(74, max_x - 4)
    start_y = max(1, (max_y - h) // 2)
    start_x = max(1, (max_x - w) // 2)

    safe_curs_set(0)

    while True:
        safe_addstr(stdscr, start_y, start_x, "╔" + "═" * (w - 2) + "╗", (get_color(2) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        safe_addstr(stdscr, start_y, start_x + 2, " Destination Output Folder ", (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        for r in range(1, h - 1):
            safe_addstr(stdscr, start_y + r, start_x, "║" + " " * (w - 2) + "║", get_color(2) if safe_has_colors() else 0)
        safe_addstr(stdscr, start_y + h - 1, start_x, "╚" + "═" * (w - 2) + "╝", (get_color(2) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)

        # Summary of current and default paths
        curr_disp = current_output or "<none>"
        if len(curr_disp) > w - 16:
            curr_disp = "…" + curr_disp[-(w - 17):]
        def_disp = default_output or "<none>"
        if len(def_disp) > w - 16:
            def_disp = "…" + def_disp[-(w - 17):]

        safe_addstr(stdscr, start_y + 1, start_x + 3, "Current: ", (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        safe_addstr(stdscr, start_y + 1, start_x + 12, curr_disp, (get_color(2) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        safe_addstr(stdscr, start_y + 2, start_x + 3, "Default: ", (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        safe_addstr(stdscr, start_y + 2, start_x + 12, def_disp, (get_color(4) | curses.A_DIM) if safe_has_colors() else curses.A_DIM)

        safe_addstr(stdscr, start_y + 3, start_x + 2, "╟" + "─" * (w - 4) + "╢", (get_color(2) | curses.A_DIM) if safe_has_colors() else curses.A_DIM)

        # Draw option rows
        for i, (label, desc) in enumerate(options):
            row_y = start_y + 4 + i
            is_focused = (i == sel_idx)
            prefix = " ▶ " if is_focused else "   "
            line_str = f"{prefix}{label:<28} {desc}"[: w - 6]
            if is_focused:
                attr = (get_color(2) | curses.A_STANDOUT | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT
            else:
                attr = get_color(2) if safe_has_colors() else 0
            safe_addstr(stdscr, row_y, start_x + 2, line_str, attr)

        safe_addstr(stdscr, start_y + h - 2, start_x + 3, "[Enter/Space] Select   [Esc] Cancel", (get_color(4) | curses.A_DIM) if safe_has_colors() else curses.A_DIM)
        stdscr.refresh()

        k = stdscr.getch()
        if k in (curses.KEY_UP, curses.KEY_LEFT, ord("k"), ord("h")):
            sel_idx = (sel_idx - 1) % len(options)
        elif k in (curses.KEY_DOWN, curses.KEY_RIGHT, ord("j"), ord("l")):
            sel_idx = (sel_idx + 1) % len(options)
        elif k in (27, ord("q"), ord("Q")):
            return None
        elif k in (10, 13, 32):  # Enter or Space
            chosen_label = options[sel_idx][0]
            if "Finder" in chosen_label:
                curses.def_prog_mode()
                curses.endwin()
                start_dir = current_output or default_output or os.getcwd()
                res = pick_folder_gui("Select Destination Folder", default_dir=start_dir)
                curses.reset_prog_mode()
                stdscr.refresh()
                return res if res else None
            elif "Manual" in chosen_label:
                val = show_text_edit_dialog(
                    stdscr,
                    "Output Directory",
                    "Enter folder to save MLX JSONL datasets:",
                    default_val=current_output or default_output,
                )
                return val.strip() if val else None
            elif "Reset" in chosen_label:
                return default_output


def show_results_dialog(stdscr: curses.window, result: Any, *args: Any) -> None:
    """Modal dialog displaying conversion results."""
    if not hasattr(result, "output_dir") or result is False:
        msg = args[0] if args else (str(result) if result else "Operation failed.")
        show_error_dialog(stdscr, "Operation Error", msg)
        return

    configure_escdelay(25)
    safe_curs_set(0)

    max_y, max_x = stdscr.getmaxyx()
    num_files = len(getattr(result, "output_files", {}))
    h = min(max(9, 7 + num_files), max_y - 2)
    w = min(74, max_x - 4)
    start_y = max(1, (max_y - h) // 2)
    start_x = max(1, (max_x - w) // 2)

    while True:
        safe_addstr(stdscr, start_y, start_x, "╔" + "═" * (w - 2) + "╗", get_color(3) | curses.A_BOLD)
        safe_addstr(stdscr, start_y, start_x + 2, " [OK] Conversion Successful! ", (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        for r in range(1, h - 1):
            safe_addstr(stdscr, start_y + r, start_x, "║" + " " * (w - 2) + "║", get_color(3))
        safe_addstr(stdscr, start_y + h - 1, start_x, "╚" + "═" * (w - 2) + "╝", get_color(3) | curses.A_BOLD)

        safe_addstr(stdscr, start_y + 2, start_x + 3, f"Saved dataset to: {result.output_dir}"[: w - 6], (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        row = start_y + 4
        for split_name, file_path in getattr(result, "output_files", {}).items():
            if row < start_y + h - 2:
                cnt = getattr(result, "record_counts", {}).get(split_name, 0)
                sz = getattr(result, "file_sizes", {}).get(split_name, 0) / 1024.0
                safe_addstr(stdscr, row, start_x + 3, f"• {file_path.name}: {cnt:,} records ({sz:.1f} KB)"[: w - 6], (get_color(4) | curses.A_DIM) if safe_has_colors() else curses.A_DIM)
                row += 1

        safe_addstr(stdscr, start_y + h - 2, start_x + 3, "Press [Enter], [Space], or [Esc] to return", (get_color(2) | curses.A_STANDOUT | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT)
        stdscr.refresh()

        k = stdscr.getch()
        if k in (10, 13, 32, 27, ord("q"), ord("Q")):
            break


def show_help_dialog(stdscr: curses.window) -> None:
    """Modal dialog displaying all keyboard shortcuts."""
    max_y, max_x = stdscr.getmaxyx()
    h = min(18, max_y - 2)
    w = min(72, max_x - 4)
    start_y = max(1, (max_y - h) // 2)
    start_x = max(1, (max_x - w) // 2)

    shortcuts = [
        ("Tab / Shift-Tab", "Switch focus between Left (Dataset) and Right (Config) panels"),
        ("↑ / ↓ (or k / j)", "Navigate vertically through fields, options, and buttons"),
        ("← / → (or h / l)", "Vertical navigation (← switches up, → switches down)"),
        ("Enter / Space", "Select format toggle, open column dropdown, or edit field"),
        ("F2", "Open native macOS Finder upload / dataset picker"),
        ("F3", "Open macOS Finder to choose output destination folder"),
        ("F5", "Run conversion and write train/valid/test JSONL files"),
        ("F9", "Toggle color scheme (Norton Commander <-> Modern)"),
        ("r / R", "Randomize split seed"),
        ("? / F1", "Show this help screen"),
        ("F10 / q / Esc", "Exit MLX Commander"),
    ]

    while True:
        safe_addstr(stdscr, start_y, start_x, "╔" + "═" * (w - 2) + "╗", get_color(2) | curses.A_BOLD)
        safe_addstr(stdscr, start_y, start_x + 2, " MLX Commander Keyboard Shortcuts ", (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
        for r in range(1, h - 1):
            safe_addstr(stdscr, start_y + r, start_x, "║" + " " * (w - 2) + "║", get_color(2))
        safe_addstr(stdscr, start_y + h - 1, start_x, "╚" + "═" * (w - 2) + "╝", get_color(2) | curses.A_BOLD)

        for i, (key, desc) in enumerate(shortcuts):
            row = start_y + 2 + i
            if row < start_y + h - 2:
                safe_addstr(stdscr, row, start_x + 3, f"{key:<20}", (get_color(4) | curses.A_BOLD) if safe_has_colors() else curses.A_BOLD)
                safe_addstr(stdscr, row, start_x + 24, desc[: w - 27], (get_color(4) | curses.A_DIM) if safe_has_colors() else curses.A_DIM)

        safe_addstr(stdscr, start_y + h - 2, start_x + 3, "Press [Enter], [Space], or [Esc] to close", (get_color(2) | curses.A_STANDOUT | curses.A_BOLD) if safe_has_colors() else curses.A_STANDOUT)
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
    safe_addstr(win, header_y, left_x + 1, "ORIGINAL COLUMNS", curses.A_BOLD | get_color(4))
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
        left_top = fmt_border("┌", "┐", "", box_w)
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
                left_div = fmt_border("├", "┤", "", box_w)
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

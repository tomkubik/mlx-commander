"""
Terminal Spawner for MLX-Commander.
Enables AI agents (Antigravity, Claude Desktop, Cursor, MCP clients) and non-interactive
subshells to spawn an interactive macOS Terminal window running the curses TUI,
awaiting completion via a file-based status handshake.
"""

import json
import os
import platform
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import List, Optional


def is_macos() -> bool:
    """Return True if running on macOS (Darwin)."""
    return platform.system() == "Darwin"


def is_terminal_interactive() -> bool:
    """Check if current process has an interactive TTY with adequate terminfo."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return False
    term = os.environ.get("TERM", "")
    if not term or term == "dumb":
        return False
    return True


def build_terminal_script(
    python_exe: str,
    args: List[str],
    status_file: str,
    working_dir: str,
) -> str:
    """
    Construct the shell script executed inside the spawned macOS Terminal window.
    Ensures exit status and completion are safely written to status_file.
    """
    clean_args = [a for a in args if a not in ("--spawn-terminal", "--external-terminal")]
    if "--tui" not in clean_args and "--commander" not in clean_args:
        clean_args.insert(0, "--tui")

    cmd_parts = [shlex.quote(python_exe), "-m", "mlx_commander"] + [shlex.quote(a) for a in clean_args]
    exec_cmd = " ".join(cmd_parts)
    escaped_cwd = shlex.quote(working_dir)

    script_lines = [
        "#!/usr/bin/env bash",
        f"cd {escaped_cwd}",
        exec_cmd,
        "EC=$?",
        f'python3 -c "import json; json.dump({{\\"exit_code\\": $EC, \\"finished\\": True}}, open({json.dumps(status_file)}, \\"w\\"))"',
        "if [ $EC -eq 0 ]; then",
        '  echo "\n[OK] Conversion completed. Closing window in 2s..."; sleep 2; exit 0;',
        "else",
        '  echo "\nSession ended (code $EC). Closing window..."; sleep 1; exit $EC;',
        "fi",
    ]
    return "\n".join(script_lines) + "\n"


def spawn_terminal_tui(
    args: List[str],
    manifest_path: Optional[str] = None,
    timeout: Optional[float] = None,
    working_dir: Optional[str] = None,
) -> int:
    """
    Spawn the MLX-Commander TUI in a dedicated macOS Terminal.app window
    and block synchronously until user completes conversion or exits.

    Returns:
        Exit code: 0 on successful conversion, 130 on cancel, 1 on error.
    """
    if not is_macos():
        raise RuntimeError("Terminal spawning via AppleScript is only supported on macOS.")

    cwd = working_dir or os.getcwd()
    run_id = uuid.uuid4().hex[:10]
    status_file = os.path.join(tempfile.gettempdir(), f"mlx_commander_status_{run_id}.json")
    script_file = os.path.join(tempfile.gettempdir(), f"mlx_commander_run_{run_id}.sh")

    # Forward manifest path if provided
    forward_args = list(args)
    if manifest_path and "--manifest-file" not in forward_args:
        forward_args.extend(["--manifest-file", str(manifest_path)])

    script_content = build_terminal_script(
        python_exe=sys.executable,
        args=forward_args,
        status_file=status_file,
        working_dir=cwd,
    )

    with open(script_file, "w", encoding="utf-8") as f:
        f.write(script_content)
    os.chmod(script_file, 0o755)

    applescript = (
        f'tell application "Terminal"\n'
        f'    activate\n'
        f'    do script "{script_file}"\n'
        f'end tell\n'
    )

    try:
        proc = subprocess.run(
            ["osascript", "-e", applescript],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            sys.stderr.write(f"Failed to spawn Terminal via AppleScript: {proc.stderr}\n")
            return 1

        # Poll status file
        start_time = time.time()
        while True:
            if timeout and (time.time() - start_time) > timeout:
                sys.stderr.write("Timed out waiting for MLX-Commander TUI session.\n")
                return 1

            if os.path.exists(status_file):
                try:
                    with open(status_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if data.get("finished"):
                        return int(data.get("exit_code", 0))
                except (json.JSONDecodeError, OSError):
                    pass

            time.sleep(0.25)
    except KeyboardInterrupt:
        sys.stderr.write("\nProcess interrupted.\n")
        return 130
    finally:
        for fpath in (status_file, script_file):
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except OSError:
                    pass

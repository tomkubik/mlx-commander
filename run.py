#!/usr/bin/env python3
"""Launcher for MLX-Commander.

Run directly with Python:
    python3 run.py
"""
import os
import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent

# Auto-detect local virtual environment if present and not already active
venv_python = base_dir / ".venv" / "bin" / "python"
if venv_python.is_file() and sys.executable != str(venv_python):
    os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve())] + sys.argv[1:])

if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from mlx_commander.cli import main

if __name__ == "__main__":
    sys.exit(main())

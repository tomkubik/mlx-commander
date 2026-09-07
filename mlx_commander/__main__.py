"""Top-level package execution entry point.

Allows running MLX-Commander via:
    python3 -m mlx_commander
"""
import sys
from mlx_commander.cli import main

if __name__ == "__main__":
    sys.exit(main())

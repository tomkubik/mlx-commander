"""Top-level package execution entry point.

Allows running MLX-Commander via:
    python3 -m hf2mlx
"""
import sys
from hf2mlx.cli import main

if __name__ == "__main__":
    sys.exit(main())

from setuptools import setup, find_packages

setup(
    name="mlx-commander",
    version="0.2.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "mlx-commander = mlx_commander.cli:main",
        ],
    },
)

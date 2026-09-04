from setuptools import setup, find_packages

setup(
    name="hf2mlx",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "hf2mlx = hf2mlx.cli:main",
        ],
    },
)

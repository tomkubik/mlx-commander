"""
Custom exceptions for MLX Commander.
"""

from typing import Optional


class MissingDependencyError(RuntimeError):
    """
    Raised when an operation or dataset format requires an optional library
    that is not currently installed in the Python environment.
    """

    def __init__(
        self,
        format_name: str,
        package_name: str,
        install_command: str,
        description: str = "",
        extra_name: Optional[str] = None,
    ):
        self.format_name = format_name
        self.package_name = package_name
        self.install_command = install_command
        self.description = description
        self.extra_name = extra_name or package_name.replace("py", "")

        msg = (
            f"The '{format_name}' format requires package '{package_name}'. "
            f"Install it with: {install_command}"
        )
        if description:
            msg += f" ({description})"
        super().__init__(msg)

    @property
    def pip_extra(self) -> str:
        """Return the pip command using mlx_commander extra syntax."""
        return f"pip install 'mlx_commander[{self.extra_name}]'"

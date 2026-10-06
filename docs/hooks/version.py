"""MkDocs build hook: the site name carries the package version.

The header and every page title read "TorchSONN <version>", the version
taken from ``pyproject.toml``, so a reader can tell which version a
published page describes.
"""

from __future__ import annotations

import tomllib
from pathlib import Path


def on_config(config):
    """Append the version in ``pyproject.toml`` to ``site_name``."""
    pyproject = Path(config.config_file_path).resolve().parent / "pyproject.toml"
    version = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]
    # `mkdocs serve` may hand the same config to a rebuild; append once.
    if not config.site_name.endswith(f" {version}"):
        config.site_name = f"{config.site_name} {version}"
    return config

"""YAML configuration loading for repeatable segmentation runs."""
from __future__ import annotations
from pathlib import Path
import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text()) or {}
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")
    config["_config_dir"] = str(path.parent)
    return config


def dataset_options(config: dict) -> dict:
    data = dict(config.get("data", {}))
    if not data.get("root"):
        raise ValueError("data.root is required in the YAML configuration.")
    root = Path(data["root"])
    if not root.is_absolute():
        root = Path(config["_config_dir"]) / root
    data["root"] = str(root.resolve())
    return data
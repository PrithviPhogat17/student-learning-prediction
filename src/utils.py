"""Small shared helpers: paths, config, seeds and JSON I/O."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Read the YAML configuration file."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(path: str | Path) -> Path:
    """Turn a config path into an absolute path (relative paths start at the project root)."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def get_dirs(cfg: dict) -> dict[str, Path]:
    """Absolute models / reports / processed directories from the config."""
    return {
        "models": resolve_path(cfg["paths"]["models_dir"]),
        "reports": resolve_path(cfg["paths"]["reports_dir"]),
        "processed": resolve_path(cfg["paths"]["processed_dir"]),
    }


def set_global_seed(seed: int) -> None:
    """Seed Python and NumPy. (TensorFlow is seeded separately in model.py.)"""
    random.seed(seed)
    np.random.seed(seed)


class _Encoder(json.JSONEncoder):
    def default(self, o: Any):  # noqa: D401
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, cls=_Encoder)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)

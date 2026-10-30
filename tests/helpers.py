"""Shared test setup: a temporary project layout filled with SYNTHETIC demo data."""
import tempfile
from pathlib import Path

from src.demo_data import DEMO_FILENAME, generate_demo_dataframe
from src.utils import load_config


def make_temp_cfg(max_epochs: int = 40, patience: int = 5):
    root = Path(tempfile.mkdtemp(prefix="slp_test_"))
    (root / "demo").mkdir()
    generate_demo_dataframe(n_rows=300, seed=1).to_csv(root / "demo" / DEMO_FILENAME, index=False)
    cfg = load_config()
    cfg["dataset"]["demo_dir"] = str(root / "demo")
    cfg["dataset"]["raw_dir"] = str(root / "raw_missing")
    cfg["paths"] = {"models_dir": str(root / "models"), "reports_dir": str(root / "reports"),
                    "processed_dir": str(root / "processed")}
    cfg["model"]["max_epochs"] = max_epochs
    cfg["model"]["patience"] = patience
    return cfg, root

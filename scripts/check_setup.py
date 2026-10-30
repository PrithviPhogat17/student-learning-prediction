"""Preflight check: is this computer ready to run the project? Run: python scripts/check_setup.py

Reports Python version, required packages, the dataset file (and its columns), and trained artifacts.
Exit code 0 = everything needed for training is present.
"""
import _bootstrap  # noqa: F401

import importlib
import sys

from src.utils import get_dirs, load_config

PACKAGES = ["tensorflow", "numpy", "pandas", "sklearn", "matplotlib", "streamlit", "joblib", "yaml"]


def main() -> int:
    ok = True
    v = sys.version_info
    print(f"Python {v.major}.{v.minor}.{v.micro}", "(OK)" if (3, 10) <= (v.major, v.minor) <= (3, 12)
          else "(TensorFlow works best with Python 3.10-3.12)")
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            print(f"  [ok]      {name} {getattr(mod, '__version__', '')}")
        except Exception as exc:
            print(f"  [MISSING] {name}  ({type(exc).__name__}) -> pip install -r requirements.txt")
            ok = False

    cfg = load_config()
    from src.data_loader import dataset_path
    path = dataset_path(cfg)
    if path.exists():
        import pandas as pd
        from src.schema import REQUIRED_COLUMNS
        df = pd.read_csv(path, sep=cfg["dataset"]["separator"])
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        print(f"  [ok]      dataset {path.name}: {df.shape[0]} rows x {df.shape[1]} columns")
        if missing:
            print(f"  [PROBLEM] missing expected columns: {missing}")
            ok = False
    else:
        print(f"  [MISSING] dataset {path} -> python scripts/download_data.py")
        ok = False

    models = get_dirs(cfg)["models"]
    print("  trained model:", "found" if (models / "metadata.json").exists()
          else "not yet (run: python scripts/train.py)")
    print("\nReady to train." if ok else "\nFix the items marked MISSING/PROBLEM, then run this again.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

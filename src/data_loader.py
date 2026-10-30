"""Dataset location, loading, inspection and cleaning (no modelling here)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .demo_data import DEMO_FILENAME
from .schema import (ALL_FEATURES, LABEL_COLUMN, RECORD_ID, REQUIRED_COLUMNS,
                     TARGET_COLUMN)
from .utils import resolve_path

DOWNLOAD_HELP = (
    "The real dataset was not found.\n"
    "  Option 1 (recommended): run  python scripts/download_data.py\n"
    "  Option 2 (manual): download https://archive.ics.uci.edu/static/public/320/student+performance.zip,\n"
    "           unzip it (it contains a nested student.zip) and copy student-mat.csv into data/raw/\n"
    "  Option 3 (demo only, synthetic data): python scripts/train.py --demo"
)


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the dataset file does not exist."""


def dataset_path(cfg: dict, demo: bool = False) -> Path:
    if demo:
        return resolve_path(cfg["dataset"]["demo_dir"]) / DEMO_FILENAME
    return resolve_path(cfg["dataset"]["raw_dir"]) / cfg["dataset"]["course_file"]


def file_md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def load_raw_dataset(cfg: dict, demo: bool = False) -> pd.DataFrame:
    """Load the CSV and keep only the columns the project needs (plus an anonymous id)."""
    path = dataset_path(cfg, demo)
    if not path.exists():
        msg = f"File not found: {path}\n" + (
            "Generate it with: python scripts/make_demo_data.py" if demo else DOWNLOAD_HELP)
        raise DatasetNotFoundError(msg)
    df = pd.read_csv(path, sep=cfg["dataset"]["separator"] if not demo else ",")
    if df.shape[1] <= 1:  # wrong delimiter guess
        df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing expected columns: {missing}. "
                         f"Found: {list(df.columns)}")
    df = df[REQUIRED_COLUMNS].copy()
    for spec in ALL_FEATURES:
        if spec.kind == "binary":
            df[spec.name] = df[spec.name].astype(str).str.strip().str.lower()
    df.insert(0, RECORD_ID, range(len(df)))
    return df


def inspect_dataframe(df: pd.DataFrame) -> dict:
    """Shapes, types, missing values, duplicates and out-of-range values."""
    out_of_range: dict[str, int] = {}
    for spec in ALL_FEATURES:
        col = pd.to_numeric(df[spec.name], errors="coerce") if spec.kind == "numeric" else None
        if spec.kind == "numeric":
            bad = int(((col < spec.min) | (col > spec.max)).sum())
        else:
            bad = int((~df[spec.name].isin(spec.options)).sum())
        if bad:
            out_of_range[spec.name] = bad
    g3 = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    bad_g3 = int(((g3 < 0) | (g3 > 20)).sum())
    if bad_g3:
        out_of_range[TARGET_COLUMN] = bad_g3
    feature_cols = [c for c in df.columns if c != RECORD_ID]
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(v) for c, v in df.isna().sum().items()},
        "n_duplicate_rows": int(df.duplicated(subset=feature_cols).sum()),
        "out_of_range_values": out_of_range,
    }


def clean_dataframe(df: pd.DataFrame, pass_mark: float) -> tuple[pd.DataFrame, dict]:
    """Coerce types, blank out invalid values (they are imputed later) and add the label.

    Rows with no usable final grade are dropped because they have no target.
    """
    df = df.copy()
    notes = {"rows_in": int(len(df))}
    for spec in ALL_FEATURES:
        if spec.kind == "numeric":
            df[spec.name] = pd.to_numeric(df[spec.name], errors="coerce")
            invalid = (df[spec.name] < spec.min) | (df[spec.name] > spec.max)
            df.loc[invalid, spec.name] = float("nan")
        else:
            df.loc[~df[spec.name].isin(spec.options), spec.name] = float("nan")
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df.loc[(df[TARGET_COLUMN] < 0) | (df[TARGET_COLUMN] > 20), TARGET_COLUMN] = float("nan")
    before = len(df)
    df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    notes["rows_dropped_missing_target"] = int(before - len(df))
    df[LABEL_COLUMN] = (df[TARGET_COLUMN] < pass_mark).astype(int)
    notes["rows_out"] = int(len(df))
    notes["at_risk_count"] = int(df[LABEL_COLUMN].sum())
    notes["at_risk_share"] = float(df[LABEL_COLUMN].mean()) if len(df) else 0.0
    return df, notes


def resolve_dataset_for_app(cfg: dict, metadata: dict | None):
    """Pick which dataset file the dashboard should show. Returns (df, mode) or (None, None)."""
    mode = (metadata or {}).get("dataset_mode")
    candidates = [True] if mode == "demo_synthetic" else [False]
    for demo in candidates:
        if dataset_path(cfg, demo).exists():
            df = load_raw_dataset(cfg, demo=demo)
            df, _ = clean_dataframe(df, cfg["dataset"]["pass_mark"])
            return df, ("demo_synthetic" if demo else "real_uci")
    return None, None

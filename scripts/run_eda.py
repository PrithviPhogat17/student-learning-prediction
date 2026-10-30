"""Exploratory data analysis: writes figures to reports/figures/ and reports/eda_summary.txt."""
import _bootstrap  # noqa: F401

import argparse
import sys

from src.data_loader import DatasetNotFoundError, clean_dataframe, load_raw_dataset
from src.eda import make_eda_figures, summary_text
from src.utils import get_dirs, load_config

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use the synthetic demo dataset")
    args = ap.parse_args()
    cfg = load_config()
    try:
        df, _ = clean_dataframe(load_raw_dataset(cfg, demo=args.demo), cfg["dataset"]["pass_mark"])
    except DatasetNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    reports = get_dirs(cfg)["reports"]
    reports.mkdir(parents=True, exist_ok=True)
    for p in make_eda_figures(df, reports / "figures", cfg["dataset"]["pass_mark"]):
        print("wrote", p)
    text = summary_text(df, cfg["dataset"]["pass_mark"])
    (reports / "eda_summary.txt").write_text(text, encoding="utf-8")
    print(text)

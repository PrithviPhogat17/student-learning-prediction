"""Evaluate the saved model on the held-out test set and write reports/ + figures."""
import _bootstrap  # noqa: F401

import sys

from src.evaluate_model import evaluate_artifacts
from src.utils import load_config

if __name__ == "__main__":
    try:
        evaluate_artifacts(load_config())
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

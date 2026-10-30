"""Inference: load saved artifacts, validate inputs, predict. No Streamlit code here."""
from __future__ import annotations

import math
from pathlib import Path

import joblib
import pandas as pd

from .model import load_model
from .schema import FeatureSpec, get_feature_specs
from .train_model import METADATA_FILE, PREPROCESSOR_FILE
from .utils import get_dirs, load_config, read_json


class ArtifactsMissingError(FileNotFoundError):
    """Raised when the trained model files are not available."""


def artifacts_available(models_dir: Path) -> bool:
    models_dir = Path(models_dir)
    return (models_dir / METADATA_FILE).exists() and (models_dir / PREPROCESSOR_FILE).exists()


def risk_band(probability: float, low: float = 0.30, high: float = 0.60) -> str:
    if probability >= high:
        return "higher"
    if probability >= low:
        return "moderate"
    return "lower"


def validate_input(raw: dict, specs: list[FeatureSpec]) -> tuple[dict, list[str]]:
    """Check every field against the schema. Returns (clean_values, error_messages)."""
    clean: dict = {}
    errors: list[str] = []
    for s in specs:
        if s.name not in raw or raw[s.name] is None or raw[s.name] == "":
            errors.append(f"{s.label}: a value is required.")
            continue
        value = raw[s.name]
        if s.kind == "binary":
            v = str(value).strip().lower()
            if v not in s.options:
                errors.append(f"{s.label}: must be one of {list(s.options)} (got {value!r}).")
            else:
                clean[s.name] = v
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            errors.append(f"{s.label}: must be a number (got {value!r}).")
            continue
        if not math.isfinite(num):
            errors.append(f"{s.label}: must be a finite number.")
        elif num < s.min or num > s.max:
            errors.append(f"{s.label}: must be between {s.min:g} and {s.max:g} (got {num:g}).")
        elif s.integer and num != int(num):
            errors.append(f"{s.label}: must be a whole number (got {num:g}).")
        else:
            clean[s.name] = int(num) if s.integer else num
    return clean, errors


class StudentRiskPredictor:
    """Loads the trained ANN + preprocessing pipeline once and serves predictions."""

    def __init__(self, metadata: dict, preprocessor, trained_model, cfg: dict):
        self.metadata = metadata
        self.preprocessor = preprocessor
        self.model = trained_model
        self.cfg = cfg
        self.specs = get_feature_specs(metadata["include_prior_grades"])
        self.threshold = metadata["decision_threshold"]

    @classmethod
    def from_artifacts(cls, cfg: dict | None = None) -> "StudentRiskPredictor":
        cfg = cfg or load_config()
        models_dir = get_dirs(cfg)["models"]
        if not artifacts_available(models_dir):
            raise ArtifactsMissingError(
                f"No trained model found in {models_dir}. Train one first: python scripts/train.py")
        meta = read_json(models_dir / METADATA_FILE)
        try:
            model = load_model(meta["backend"], models_dir)
        except FileNotFoundError as exc:
            raise ArtifactsMissingError(str(exc)) from exc
        return cls(meta, joblib.load(models_dir / PREPROCESSOR_FILE), model, cfg)

    def predict_one(self, raw: dict) -> dict:
        clean, errors = validate_input(raw, self.specs)
        if errors:
            raise ValueError("Invalid input:\n- " + "\n- ".join(errors))
        prob = float(self.model.predict_proba(self.preprocessor.transform(pd.DataFrame([clean])))[0])
        rec = self.cfg["recommendations"]
        return {
            "probability_at_risk": prob,
            "predicted_class": "At risk of not passing" if prob >= self.threshold else "Likely to pass",
            "at_risk": bool(prob >= self.threshold),
            "threshold": self.threshold,
            "risk_band": risk_band(prob, rec["risk_low"], rec["risk_high"]),
            "inputs": clean,
        }

    def predict_dataframe(self, df: pd.DataFrame):
        return self.model.predict_proba(self.preprocessor.transform(df))

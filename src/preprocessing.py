"""Preprocessing: encoding, imputation, scaling and the train/validation/test split.

The fitted pipeline is learned from TRAINING data only and then re-used,
unchanged, for validation, test and dashboard inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .schema import LABEL_COLUMN, FeatureSpec


class StudentFeatureEncoder(BaseEstimator, TransformerMixin):
    """Select the model's columns in a fixed order and encode yes/no as 1/0.

    Unknown values become NaN and are filled by the imputer that follows.
    """

    def __init__(self, feature_names: list[str], binary_names: list[str]):
        self.feature_names = feature_names
        self.binary_names = binary_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        missing = [c for c in self.feature_names if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing input columns: {missing}")
        out = pd.DataFrame(index=frame.index)
        for name in self.feature_names:
            if name in self.binary_names:
                mapped = frame[name].astype(str).str.strip().str.lower().map({"no": 0.0, "yes": 1.0})
                out[name] = mapped.astype(float)
            else:
                out[name] = pd.to_numeric(frame[name], errors="coerce").astype(float)
        return out.to_numpy(dtype=float)


def build_preprocessor(specs: list[FeatureSpec]) -> Pipeline:
    names = [s.name for s in specs]
    binary = [s.name for s in specs if s.kind == "binary"]
    return Pipeline([
        ("encode", StudentFeatureEncoder(names, binary)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])


def split_data(df: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """Stratified train / validation / test split.

    Every row is a different student (one course file), so a plain stratified split
    cannot put the same student in two sets. Do not merge the Math and Portuguese
    files before splitting: many students appear in both.
    """
    seed = cfg["split"]["seed"]
    test_size = cfg["split"]["test_size"]
    val_size = cfg["split"]["val_size"]
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[LABEL_COLUMN], random_state=seed)
    train, val = train_test_split(
        train_val, test_size=val_size / (1.0 - test_size),
        stratify=train_val[LABEL_COLUMN], random_state=seed)
    return {"train": train.reset_index(drop=True),
            "val": val.reset_index(drop=True),
            "test": test.reset_index(drop=True)}


def fit_transform_splits(splits: dict[str, pd.DataFrame], specs: list[FeatureSpec]):
    """Fit the preprocessor on train only; return it plus transformed arrays."""
    pre = build_preprocessor(specs)
    x_train = pre.fit_transform(splits["train"])
    arrays = {"train": x_train,
              "val": pre.transform(splits["val"]),
              "test": pre.transform(splits["test"])}
    labels = {k: v[LABEL_COLUMN].to_numpy(dtype=int) for k, v in splits.items()}
    return pre, arrays, labels


def class_weights(y: np.ndarray) -> dict[int, float]:
    """'Balanced' weights: n_samples / (n_classes * count_of_class)."""
    n = len(y)
    out = {}
    for cls in (0, 1):
        count = int((y == cls).sum())
        out[cls] = n / (2.0 * count) if count else 1.0
    return out

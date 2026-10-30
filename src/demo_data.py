"""Synthetic DEMO data generator (NOT real student data).

Used only when the real UCI dataset cannot be downloaded (or for automated
tests). Records are random numbers with a plausible structure. Any metric
computed on this data is illustrative and must never be reported as real
predictive accuracy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import REQUIRED_COLUMNS

DEMO_FILENAME = "synthetic_student_mat_DEMO.csv"


def generate_demo_dataframe(n_rows: int = 395, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g1 = np.clip(np.round(rng.normal(10.9, 3.3, n_rows)), 0, 20)
    studytime = rng.choice([1, 2, 3, 4], n_rows, p=[0.27, 0.50, 0.16, 0.07])
    failures = rng.choice([0, 1, 2, 3], n_rows, p=[0.79, 0.12, 0.04, 0.05])
    absences = np.clip(np.round(rng.exponential(5.0, n_rows)), 0, 60)
    absences[rng.random(n_rows) < 0.3] = 0
    g2 = np.clip(np.round(g1 + rng.normal(-0.1, 1.8, n_rows)
                          + 0.3 * (studytime - 2) - 0.7 * failures), 0, 20)
    g3 = g2 + rng.normal(0.2, 1.5, n_rows) - 0.04 * absences - 0.3 * failures
    g3 = np.clip(np.round(g3), 0, 20)
    dropped = (g2 < 8) & (rng.random(n_rows) < 0.35)   # a few students with final grade 0
    g3[dropped] = 0

    def yn(p):
        return np.where(rng.random(n_rows) < p, "yes", "no")

    df = pd.DataFrame({
        "G1": g1, "G2": g2, "studytime": studytime, "absences": absences,
        "failures": failures,
        "traveltime": rng.choice([1, 2, 3, 4], n_rows, p=[0.65, 0.27, 0.06, 0.02]),
        "freetime": rng.choice([1, 2, 3, 4, 5], n_rows, p=[0.05, 0.16, 0.40, 0.29, 0.10]),
        "goout": rng.choice([1, 2, 3, 4, 5], n_rows, p=[0.06, 0.26, 0.33, 0.22, 0.13]),
        "schoolsup": yn(0.13), "famsup": yn(0.61), "paid": yn(0.46),
        "activities": yn(0.51), "higher": yn(0.95), "internet": yn(0.83),
        "G3": g3,
    })
    for col in df.columns:
        if col not in ("schoolsup", "famsup", "paid", "activities", "higher", "internet"):
            df[col] = df[col].astype(int)
    return df[REQUIRED_COLUMNS]

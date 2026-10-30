"""Exploratory data analysis figures and summary (used by scripts/run_eda.py)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .schema import ALL_FEATURES, LABEL_COLUMN, TARGET_COLUMN  # noqa: E402

NAVY, TEAL = "#0B2545", "#13A89E"


def make_eda_figures(df: pd.DataFrame, out_dir: Path, pass_mark: float) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(df[TARGET_COLUMN], bins=range(0, 22), color=TEAL, edgecolor="white")
    axes[0].axvline(pass_mark, color=NAVY, ls="--", label=f"Pass mark ({pass_mark:g})")
    axes[0].set_xlabel("Final grade G3 (0-20)"); axes[0].set_ylabel("Students"); axes[0].legend()
    axes[0].set_title("Final grade distribution")
    counts = df[LABEL_COLUMN].value_counts().reindex([0, 1]).fillna(0)
    axes[1].bar(["Not at risk", "At risk"], counts.values, color=[TEAL, NAVY])
    for i, v in enumerate(counts.values):
        axes[1].text(i, v, f"{int(v)}\n({v / len(df):.0%})", ha="center", va="bottom")
    axes[1].set_title("Target classes (at risk = G3 below pass mark)")
    fig.tight_layout(); p = out_dir / "target_distribution.png"; fig.savefig(p, dpi=140); plt.close(fig); paths.append(p)

    miss = df.drop(columns=["record_id"], errors="ignore").isna().sum()
    fig, ax = plt.subplots(figsize=(7, 4))
    if miss.sum() == 0:
        ax.text(0.5, 0.5, "No missing values found", ha="center", va="center", fontsize=14, color=NAVY)
        ax.axis("off")
    else:
        miss[miss > 0].plot.barh(ax=ax, color=TEAL); ax.set_xlabel("Missing values")
    ax.set_title("Missing-value summary")
    fig.tight_layout(); p = out_dir / "missing_values.png"; fig.savefig(p, dpi=140); plt.close(fig); paths.append(p)

    numeric = [s.name for s in ALL_FEATURES if s.kind == "numeric"]
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.5))
    for ax, col in zip(axes.ravel(), numeric):
        ax.hist(df[col].dropna(), bins=min(20, max(5, int(df[col].nunique()))), color=TEAL, edgecolor="white")
        ax.set_title(col, fontsize=10)
    for ax in axes.ravel()[len(numeric):]:
        ax.axis("off")
    fig.suptitle("Numeric feature distributions"); fig.tight_layout()
    p = out_dir / "feature_distributions.png"; fig.savefig(p, dpi=140); plt.close(fig); paths.append(p)

    corr_cols = numeric + [TARGET_COLUMN]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_cols)), corr_cols, rotation=60, ha="right")
    ax.set_yticks(range(len(corr_cols)), corr_cols)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8); ax.set_title("Correlation heatmap (association, not causation)")
    fig.tight_layout(); p = out_dir / "correlation_heatmap.png"; fig.savefig(p, dpi=140); plt.close(fig); paths.append(p)
    return paths


def summary_text(df: pd.DataFrame, pass_mark: float) -> str:
    lines = [f"Rows: {len(df)}", f"At-risk (G3 < {pass_mark:g}): {int(df[LABEL_COLUMN].sum())} "
             f"({df[LABEL_COLUMN].mean():.1%})", "", "Numeric summary:",
             df[[s.name for s in ALL_FEATURES if s.kind == 'numeric'] + [TARGET_COLUMN]].describe().round(2).to_string()]
    corr = df[["G1", "G2", TARGET_COLUMN]].corr().round(2)
    lines += ["", "Correlation among G1, G2, G3 (G3 is strongly tied to earlier grades):", corr.to_string()]
    return "\n".join(lines) + "\n"

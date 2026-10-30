"""Evaluation on the held-out test set, baselines, bootstrap intervals, permutation importance."""
from __future__ import annotations

import datetime as dt

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,  # noqa: E402
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)

from .model import load_model  # noqa: E402
from .schema import LABEL_COLUMN, get_feature_specs  # noqa: E402
from .train_model import METADATA_FILE, PREPROCESSOR_FILE  # noqa: E402
from .utils import get_dirs, read_json, write_json  # noqa: E402

NAVY, TEAL, GREY = "#0B2545", "#13A89E", "#8D99AE"


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    auc = float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) == 2 else None
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),   # rows = actual [not at risk, at risk], cols = predicted
    }


def bootstrap_ci(y_true, proba, threshold: float, n_boot: int = 1000, seed: int = 0) -> dict:
    """95% percentile bootstrap intervals; shows how noisy a small test set is."""
    rng = np.random.default_rng(seed)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    vals: dict[str, list[float]] = {"accuracy": [], "f1": [], "roc_auc": []}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt, pr = y_true[idx], proba[idx]
        pred = (pr >= threshold).astype(int)
        vals["accuracy"].append(accuracy_score(yt, pred))
        vals["f1"].append(f1_score(yt, pred, zero_division=0))
        if len(np.unique(yt)) == 2:
            vals["roc_auc"].append(roc_auc_score(yt, pr))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None
            for k, v in vals.items()}


def permutation_importance(predict_fn, pre, df: pd.DataFrame, y, feature_names, n_repeats=30, seed=0):
    """Drop in ROC-AUC when one ORIGINAL input column is shuffled (before preprocessing)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    base = roc_auc_score(y, predict_fn(pre.transform(df)))
    rows = []
    for name in feature_names:
        drops = []
        for _ in range(n_repeats):
            shuffled = df.copy()
            shuffled[name] = rng.permutation(shuffled[name].to_numpy())
            drops.append(base - roc_auc_score(y, predict_fn(pre.transform(shuffled))))
        rows.append({"feature": name, "importance_mean": float(np.mean(drops)),
                     "importance_std": float(np.std(drops))})
    return sorted(rows, key=lambda r: r["importance_mean"], reverse=True)


# ------------------------------------------------------------------ plots
def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_training_curves(history: dict, path, best_epoch: int):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(history["loss"], color=NAVY, label="Training loss")
    ax.plot(history["val_loss"], color=TEAL, label="Validation loss")
    ax.axvline(best_epoch - 1, color=GREY, ls="--", label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Binary cross-entropy"); ax.set_title("Training vs validation loss")
    ax.legend(); _save(fig, path)


def plot_confusion(cm, path, title):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    labels = ["Not at risk", "At risk"]
    ax.set_xticks([0, 1], labels); ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center",
                    color="white" if cm[i][j] > np.max(cm) / 2 else NAVY, fontsize=14)
    _save(fig, path)


def plot_roc(curves: dict, path):
    fig, ax = plt.subplots(figsize=(5, 4.4))
    colors = [NAVY, TEAL, GREY]
    for (name, (fpr, tpr, auc)), c in zip(curves.items(), colors):
        ax.plot(fpr, tpr, color=c, label=f"{name} (AUC={auc:.2f})")
    ax.plot([0, 1], [0, 1], ls=":", color="black", label="Chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test set)"); ax.legend(loc="lower right"); _save(fig, path)


def plot_comparison(models: dict, path):
    names = list(models)
    metrics = ["accuracy", "f1", "roc_auc"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    width = 0.25
    for i, (m, c) in enumerate(zip(metrics, [NAVY, TEAL, GREY])):
        vals = [models[n][m] or 0 for n in names]
        ax.bar(np.arange(len(names)) + i * width, vals, width, label=m, color=c)
    ax.set_xticks(np.arange(len(names)) + width, names, rotation=12)
    ax.set_ylim(0, 1.05); ax.set_title("Model vs baselines (test set)"); ax.legend()
    _save(fig, path)


def plot_importance(rows, path):
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.barh([r["feature"] for r in rows], [r["importance_mean"] for r in rows],
            xerr=[r["importance_std"] for r in rows], color=TEAL)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Drop in ROC-AUC when shuffled"); ax.set_title("Permutation importance (test set)")
    _save(fig, path)


# ------------------------------------------------------------ orchestrator
def evaluate_artifacts(cfg: dict, n_boot: int = 1000, n_repeats: int = 30, verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    dirs = get_dirs(cfg)
    meta_path = dirs["models"] / METADATA_FILE
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found. Train first: python scripts/train.py")
    meta = read_json(meta_path)
    pre = joblib.load(dirs["models"] / PREPROCESSOR_FILE)
    trained = load_model(meta["backend"], dirs["models"])
    train_df = pd.read_csv(dirs["processed"] / "train.csv")
    test_df = pd.read_csv(dirs["processed"] / "test.csv")
    specs = get_feature_specs(meta["include_prior_grades"])
    names = [s.name for s in specs]
    thr = meta["decision_threshold"]
    y_train, y_test = train_df[LABEL_COLUMN].to_numpy(), test_df[LABEL_COLUMN].to_numpy()
    x_train, x_test = pre.transform(train_df), pre.transform(test_df)

    log("Scoring the ANN and baselines on the held-out test set")
    scores = {"ANN": trained.predict_proba(x_test)}
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced").fit(x_train, y_train)
    scores["Logistic regression"] = logreg.predict_proba(x_test)[:, 1]
    dummy = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    scores["Majority class"] = dummy.predict_proba(x_test)[:, 1]
    if "G2" in names:   # simple rule: predict "at risk" when the latest grade is below the pass mark
        scores["Rule: G2 below pass mark"] = (test_df["G2"].to_numpy() < meta["pass_mark"]).astype(float)

    results = {n: classification_metrics(y_test, p, thr) for n, p in scores.items()}
    results["ANN"]["confidence_intervals_95"] = bootstrap_ci(y_test, scores["ANN"], thr, n_boot)
    results["Logistic regression"]["confidence_intervals_95"] = bootstrap_ci(
        y_test, scores["Logistic regression"], thr, n_boot)

    val_df = pd.read_csv(dirs["processed"] / "val.csv")
    val_metrics = classification_metrics(val_df[LABEL_COLUMN], trained.predict_proba(pre.transform(val_df)), thr)

    log("Permutation importance")
    importance = permutation_importance(trained.predict_proba, pre, test_df, y_test, names, n_repeats)

    figs = dirs["reports"] / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    history = read_json(dirs["reports"] / "training_history.json")
    plot_training_curves(history, figs / "training_curves.png", meta["best_epoch"])
    plot_confusion(results["ANN"]["confusion_matrix"], figs / "confusion_matrix.png", "ANN confusion matrix (test)")
    curves = {}
    for name in ("ANN", "Logistic regression"):
        fpr, tpr, _ = roc_curve(y_test, scores[name])
        curves[name] = (fpr, tpr, results[name]["roc_auc"])
    plot_roc(curves, figs / "roc_curves.png")
    plot_comparison(results, figs / "model_comparison.png")
    plot_importance(importance, figs / "permutation_importance.png")

    report = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "backend": meta["backend"], "backend_description": meta["backend_description"],
        "dataset_mode": meta["dataset_mode"], "threshold": thr,
        "n_train": len(train_df), "n_test": len(test_df),
        "test_at_risk_share": float(y_test.mean()),
        "include_prior_grades": meta["include_prior_grades"],
        "validation_ann": val_metrics, "test": results,
        "permutation_importance": importance,
        "notes": [
            "Metrics come from a single stratified split; the test set is small, so see the bootstrap intervals.",
            "Permutation importance shows what the model relies on, not what causes outcomes.",
        ] + (["DEMO MODE: synthetic data - these numbers are illustrative only and say nothing about real students."]
             if meta["dataset_mode"] == "demo_synthetic" else []),
    }
    write_json(report, dirs["reports"] / "model_evaluation.json")
    (dirs["reports"] / "model_evaluation.txt").write_text(format_text_report(report), encoding="utf-8")
    (dirs["reports"] / "results_section.md").write_text(format_markdown_results(report), encoding="utf-8")
    log(format_text_report(report))
    return report


def _fmt(v):
    return "n/a" if v is None else f"{v:.3f}"


def format_text_report(r: dict) -> str:
    lines = [f"Model: {r['backend_description']}", f"Dataset mode: {r['dataset_mode']}",
             f"Train rows: {r['n_train']}  Test rows: {r['n_test']}  Test at-risk share: {r['test_at_risk_share']:.2%}",
             f"Decision threshold: {r['threshold']}  Uses G1/G2: {r['include_prior_grades']}", "",
             f"{'Model':<26}{'Acc':>7}{'BalAcc':>8}{'Prec':>7}{'Rec':>7}{'F1':>7}{'AUC':>7}"]
    for name, m in r["test"].items():
        lines.append(f"{name:<26}{_fmt(m['accuracy']):>7}{_fmt(m['balanced_accuracy']):>8}"
                     f"{_fmt(m['precision']):>7}{_fmt(m['recall']):>7}{_fmt(m['f1']):>7}{_fmt(m['roc_auc']):>7}")
    ci = r["test"]["ANN"]["confidence_intervals_95"]
    lines += ["", "ANN 95% bootstrap intervals: " + ", ".join(
        f"{k} [{v[0]:.2f}, {v[1]:.2f}]" for k, v in ci.items() if v)]
    lines += ["ANN confusion matrix [[TN, FP], [FN, TP]]: " + str(r["test"]["ANN"]["confusion_matrix"]), "",
              "Permutation importance (drop in ROC-AUC):"]
    lines += [f"  {x['feature']:<12}{x['importance_mean']:+.4f} +/- {x['importance_std']:.4f}"
              for x in r["permutation_importance"]]
    lines += [""] + ["NOTE: " + n for n in r["notes"]]
    return "\n".join(lines) + "\n"


def format_markdown_results(r: dict) -> str:
    out = ["| Model | Accuracy | Balanced acc. | Precision | Recall | F1 | ROC-AUC |", "|---|---|---|---|---|---|---|"]
    for name, m in r["test"].items():
        out.append(f"| {name} | {_fmt(m['accuracy'])} | {_fmt(m['balanced_accuracy'])} | {_fmt(m['precision'])} | "
                   f"{_fmt(m['recall'])} | {_fmt(m['f1'])} | {_fmt(m['roc_auc'])} |")
    cm = r["test"]["ANN"]["confusion_matrix"]
    out += ["", f"ANN confusion matrix on {r['n_test']} test students: TN={cm[0][0]}, FP={cm[0][1]}, "
            f"FN={cm[1][0]}, TP={cm[1][1]}.", "", f"Backend: {r['backend_description']}; dataset mode: {r['dataset_mode']}."]
    out += [""] + ["- " + n for n in r["notes"]]
    return "\n".join(out) + "\n"

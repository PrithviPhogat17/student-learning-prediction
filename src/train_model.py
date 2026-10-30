"""End-to-end training: load -> inspect -> clean -> split -> preprocess -> train -> save."""
from __future__ import annotations

import copy
import datetime as dt
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import model as model_lib
from .data_loader import (clean_dataframe, dataset_path, file_md5, inspect_dataframe,
                          load_raw_dataset)
from .preprocessing import fit_transform_splits, split_data
from .schema import LABEL_COLUMN, get_feature_specs
from .utils import get_dirs, set_global_seed, write_json

PREPROCESSOR_FILE = "preprocessor.joblib"
METADATA_FILE = "metadata.json"


def run_training(cfg: dict, backend: str = "keras", demo: bool = False,
                 include_prior_grades: bool | None = None, seed: int | None = None,
                 verbose: bool = True) -> dict:
    """Train the ANN and save every artifact the dashboard needs. Returns the metadata dict."""
    if backend not in model_lib.BACKENDS:
        raise ValueError(f"backend must be one of {model_lib.BACKENDS}")
    cfg = copy.deepcopy(cfg)
    if include_prior_grades is not None:
        cfg["task"]["include_prior_grades"] = include_prior_grades
    if seed is not None:
        cfg["split"]["seed"] = seed
    seed = cfg["split"]["seed"]
    log = print if verbose else (lambda *a, **k: None)
    dirs = get_dirs(cfg)
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    if backend == "keras" and not model_lib.tensorflow_available():
        model_lib._import_tf()  # raises a helpful ModelBackendError

    set_global_seed(seed)
    path = dataset_path(cfg, demo)
    log(f"[1/6] Loading {'DEMO (synthetic)' if demo else 'real'} dataset: {path}")
    raw = load_raw_dataset(cfg, demo=demo)
    inspection = inspect_dataframe(raw)
    write_json(inspection, dirs["reports"] / "data_inspection.json")
    log(f"      rows={inspection['n_rows']} duplicates={inspection['n_duplicate_rows']} "
        f"out-of-range={inspection['out_of_range_values'] or 'none'}")

    log("[2/6] Cleaning and defining the target (at_risk = G3 < pass mark)")
    df, clean_notes = clean_dataframe(raw, cfg["dataset"]["pass_mark"])
    log(f"      {clean_notes}")

    specs = get_feature_specs(cfg["task"]["include_prior_grades"])
    log(f"[3/6] Stratified split + preprocessing ({len(specs)} input features)")
    splits = split_data(df, cfg)
    pre, arrays, labels = fit_transform_splits(splits, specs)
    for name, frame in splits.items():
        frame.to_csv(dirs["processed"] / f"{name}.csv", index=False)
    log("      " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    log(f"[4/6] Training ANN (backend={backend})")
    params = cfg["model"]
    ckpt = dirs["models"] / "checkpoints" / "best.keras"
    if backend == "keras":
        trained, history, summary = model_lib.train_keras(
            arrays["train"], labels["train"], arrays["val"], labels["val"], params, seed, ckpt)
    else:
        trained, history, summary = model_lib.train_sklearn_mlp(
            arrays["train"], labels["train"], arrays["val"], labels["val"], params, seed)
    epochs_run = len(history["loss"])
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    log(f"      epochs run={epochs_run}, best epoch={best_epoch}, "
        f"best val_loss={min(history['val_loss']):.4f}")

    log("[5/6] Saving artifacts")
    model_path = model_lib.save_model(trained, dirs["models"])
    joblib.dump(pre, dirs["models"] / PREPROCESSOR_FILE)
    write_json(history, dirs["reports"] / "training_history.json")

    metadata = {
        "backend": backend,
        "backend_description": ("TensorFlow/Keras Multi-Layer Perceptron" if backend == "keras"
                                else "scikit-learn MLP FALLBACK (not TensorFlow/Keras)"),
        "dataset_mode": "demo_synthetic" if demo else "real_uci",
        "dataset_file": path.name,
        "dataset_md5": file_md5(path),
        "n_rows_after_cleaning": clean_notes["rows_out"],
        "at_risk_share": clean_notes["at_risk_share"],
        "pass_mark": cfg["dataset"]["pass_mark"],
        "include_prior_grades": cfg["task"]["include_prior_grades"],
        "features": [s.name for s in specs],
        "n_features": len(specs),
        "decision_threshold": params["decision_threshold"],
        "seed": seed,
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
        "hyperparameters": params,
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "architecture_summary": summary,
        "model_file": model_path.name,
        "trained_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
    }
    if backend == "keras":
        import tensorflow as tf
        metadata["versions"]["tensorflow"] = tf.__version__
    write_json(metadata, dirs["models"] / METADATA_FILE)
    log("[6/6] Done. Next: python scripts/evaluate.py")
    return metadata

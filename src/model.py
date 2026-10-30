"""Neural-network backends.

* "keras"       - the project's main model: a TensorFlow/Keras Multi-Layer Perceptron.
* "sklearn_mlp" - an EXPLICIT, opt-in fallback (scikit-learn MLPClassifier) for computers
                  where TensorFlow cannot be installed. It is a real neural network too,
                  but it is NOT TensorFlow/Keras, has no dropout and no class weights.
                  The backend name is stored in metadata.json and shown in the dashboard,
                  so the two are never confused.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier

from .preprocessing import class_weights

KERAS_FILE = "ann_model.keras"
SKLEARN_FILE = "sklearn_mlp.joblib"
BACKENDS = ("keras", "sklearn_mlp")


class ModelBackendError(RuntimeError):
    """Raised when a backend cannot be used (e.g. TensorFlow missing)."""


def _import_tf():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf  # noqa: WPS433
        return tf
    except Exception as exc:  # ImportError or a broken installation
        raise ModelBackendError(
            "TensorFlow could not be imported. Install it with 'pip install -r requirements.txt' "
            "(Python 3.10-3.12 recommended). If it cannot be installed on this computer you may "
            "train the explicit fallback instead: python scripts/train.py --backend sklearn_mlp"
        ) from exc


def tensorflow_available() -> bool:
    try:
        _import_tf()
        return True
    except ModelBackendError:
        return False


@dataclass
class TrainedModel:
    backend: str
    model: Any

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Probability of the positive class ("at risk") for each row, shape (n,)."""
        x = np.asarray(x, dtype="float32" if self.backend == "keras" else float)
        if self.backend == "keras":
            return np.asarray(self.model(x, training=False)).reshape(-1)
        return self.model.predict_proba(x)[:, 1]


# ----------------------------------------------------------------- Keras ANN
def build_keras_model(n_features: int, params: dict):
    """Input -> Dense(64, ReLU) -> Dropout -> Dense(32, ReLU) -> Dropout -> Dense(1, sigmoid)."""
    tf = _import_tf()
    keras = tf.keras
    reg = keras.regularizers.l2(params["l2"])
    inputs = keras.Input(shape=(n_features,), name="features")
    x = inputs
    for i, units in enumerate(params["hidden_units"], start=1):
        x = keras.layers.Dense(units, activation="relu", kernel_regularizer=reg,
                               name=f"hidden_{i}")(x)
        if params["dropout"] > 0:
            x = keras.layers.Dropout(params["dropout"], name=f"dropout_{i}")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="at_risk_probability")(x)
    model = keras.Model(inputs, outputs, name="student_risk_ann")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy"), keras.metrics.AUC(name="auc")],
    )
    return model


def describe_keras_model(model) -> list[str]:
    """Plain-text architecture description (avoids version-specific model.summary() formatting)."""
    lines = [f"Model: {model.name}"]
    for layer in model.layers:
        kind = type(layer).__name__
        if kind == "Dense":
            detail = f"units={layer.units}, activation={layer.activation.__name__}"
        elif kind == "Dropout":
            detail = f"rate={layer.rate}"
        else:
            detail = ""
        lines.append(f"  {layer.name:<22}{kind:<12}{detail}")
    lines.append(f"Total parameters: {model.count_params()}")
    return lines


def train_keras(x_tr, y_tr, x_val, y_val, params: dict, seed: int, checkpoint_path: Path):
    tf = _import_tf()
    keras = tf.keras
    keras.utils.set_random_seed(seed)
    model = build_keras_model(x_tr.shape[1], params)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=params["patience"],
                                      restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath=str(checkpoint_path), monitor="val_loss",
                                        save_best_only=True),
    ]
    weights = class_weights(y_tr) if params.get("use_class_weights") else None
    hist = model.fit(
        np.asarray(x_tr, dtype="float32"), np.asarray(y_tr, dtype="float32"),
        validation_data=(np.asarray(x_val, dtype="float32"), np.asarray(y_val, dtype="float32")),
        epochs=params["max_epochs"], batch_size=params["batch_size"],
        class_weight=weights, callbacks=callbacks, verbose=0,
    )
    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    return TrainedModel("keras", model), history, describe_keras_model(model)


# ------------------------------------------------- scikit-learn fallback MLP
def train_sklearn_mlp(x_tr, y_tr, x_val, y_val, params: dict, seed: int):
    """Epoch-by-epoch training with manual early stopping, so loss curves can be recorded."""
    mlp = MLPClassifier(hidden_layer_sizes=tuple(params["hidden_units"]), activation="relu",
                        solver="adam", alpha=params["l2"], batch_size=params["batch_size"],
                        learning_rate_init=params["learning_rate"], random_state=seed)
    history = {"loss": [], "val_loss": []}
    best, best_state, wait = float("inf"), None, 0
    for _ in range(params["max_epochs"]):
        mlp.partial_fit(x_tr, y_tr, classes=[0, 1])
        tr = log_loss(y_tr, mlp.predict_proba(x_tr)[:, 1], labels=[0, 1])
        va = log_loss(y_val, mlp.predict_proba(x_val)[:, 1], labels=[0, 1])
        history["loss"].append(float(tr))
        history["val_loss"].append(float(va))
        if va < best - 1e-4:
            best, wait = va, 0
            best_state = (copy.deepcopy(mlp.coefs_), copy.deepcopy(mlp.intercepts_))
        else:
            wait += 1
            if wait >= params["patience"]:
                break
    if best_state is not None:
        mlp.coefs_, mlp.intercepts_ = best_state
    summary = [f"sklearn MLPClassifier hidden_layer_sizes={tuple(params['hidden_units'])}, "
               f"activation=relu, solver=adam, alpha(L2)={params['l2']}",
               "Output: logistic (sigmoid) unit; no dropout; no class weights in this fallback."]
    return TrainedModel("sklearn_mlp", mlp), history, summary


# ------------------------------------------------------------ save / load
def save_model(trained: TrainedModel, models_dir: Path) -> Path:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    if trained.backend == "keras":
        path = models_dir / KERAS_FILE
        trained.model.save(str(path))
    else:
        path = models_dir / SKLEARN_FILE
        joblib.dump(trained.model, path)
    return path


def load_model(backend: str, models_dir: Path) -> TrainedModel:
    models_dir = Path(models_dir)
    if backend == "keras":
        path = models_dir / KERAS_FILE
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        tf = _import_tf()
        return TrainedModel("keras", tf.keras.models.load_model(str(path), compile=False))
    if backend == "sklearn_mlp":
        path = models_dir / SKLEARN_FILE
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        return TrainedModel("sklearn_mlp", joblib.load(path))
    raise ValueError(f"Unknown backend: {backend}")

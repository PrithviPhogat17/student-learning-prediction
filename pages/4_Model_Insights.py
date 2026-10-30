"""Page 4: real evaluation results, baselines and explainability."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import ui_components as ui  # noqa: E402

ui.setup_page("Model Insights", "🔍")
ctx = ui.get_context()
ui.page_header("Model Performance and Explainability", "All numbers on this page are read from files written by scripts/evaluate.py.")
ui.demo_banner(ctx.metadata)

report = ui.load_report(ctx)
if report is None:
    st.error("Evaluation results not found (reports/model_evaluation.json).")
    ui.empty_state(ui.NO_MODEL_HELP)
    st.stop()

ann = report["test"]["ANN"]
c1, c2, c3, c4 = st.columns(4)
for col, (label, key) in zip((c1, c2, c3, c4), (("Accuracy", "accuracy"), ("Recall (at risk)", "recall"),
                                                 ("F1 (at risk)", "f1"), ("ROC-AUC", "roc_auc"))):
    with col:
        ui.kpi_card(label, "n/a" if ann[key] is None else f"{ann[key]:.2f}", f"test set, n={report['n_test']}")
ci = ann.get("confidence_intervals_95", {})
if ci:
    st.caption("95% bootstrap intervals (test set is small, so expect wide ranges): " + "; ".join(
        f"{k} {v[0]:.2f}-{v[1]:.2f}" for k, v in ci.items() if v))

st.subheader("Model vs baselines")
table = pd.DataFrame({name: {k: m[k] for k in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc")}
                      for name, m in report["test"].items()}).T.round(3)
st.dataframe(table)
log = report["test"].get("Logistic regression")
if log:
    st.write(f"On this split the ANN reached F1 {ann['f1']:.2f} and the logistic-regression baseline {log['f1']:.2f}. "
             "On small tabular datasets a neural network often performs about the same as a simple model; "
             "do not read a small difference as proof that one is better.")


def show(name: str, caption: str) -> None:
    path = ctx.dirs["reports"] / "figures" / name
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.warning(f"Figure {name} is missing. Re-run scripts/evaluate.py.")


left, right = st.columns(2)
with left:
    show("confusion_matrix.png", "Confusion matrix (rows: actual, columns: predicted)")
    show("training_curves.png", "Training vs validation loss")
with right:
    show("roc_curves.png", "ROC curves")
    show("model_comparison.png", "Model vs baselines")

st.subheader("What does the model rely on?")
show("permutation_importance.png", "Permutation importance")
st.caption("Permutation importance measures how much the test ROC-AUC falls when one input is shuffled. It describes "
           "the model's behaviour on this data, not causes. A high value does not mean changing that factor would "
           "change a student's result.")

st.subheader("Architecture")
if ctx.metadata:
    st.code("\n".join(ctx.metadata["architecture_summary"]))
st.markdown("Inputs (scaled) -> Dense 64 + ReLU -> Dropout -> Dense 32 + ReLU -> Dropout -> 1 sigmoid unit "
            "(probability of being at risk). Loss: binary cross-entropy. Optimizer: Adam. Early stopping on "
            "validation loss.")
for note in report["notes"]:
    st.caption("Note: " + note)

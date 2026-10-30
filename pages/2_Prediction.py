"""Page 2: enter student indicators and get the ANN's risk estimate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import ui_components as ui  # noqa: E402

ui.setup_page("Prediction", "🎯")
ctx = ui.get_context()
ui.page_header("Prediction", "Estimate whether a student is at risk of finishing below the pass mark.")
ui.demo_banner(ctx.metadata)

if ctx.predictor is None:
    st.error(ctx.error or "Model unavailable.")
    ui.empty_state(ui.NO_MODEL_HELP)
    st.stop()

pred = ctx.predictor
specs = pred.specs
pass_mark = ctx.metadata["pass_mark"]
st.caption(f"Model: {ctx.metadata['backend_description']} | inputs: {len(specs)} | "
           f"prior grades used: {'yes' if ctx.metadata['include_prior_grades'] else 'no'}")

source = st.radio("Starting values", ["Neutral defaults", "Example record from the held-out test data"],
                  horizontal=True)
defaults = {s.name: s.default for s in specs}
example_id = None
if source.startswith("Example"):
    test_path = ctx.dirs["processed"] / "test.csv"
    if test_path.exists():
        test_df = pd.read_csv(test_path)
        example_id = st.selectbox("Example record (anonymous id)", test_df["record_id"].tolist())
        row = test_df[test_df["record_id"] == example_id].iloc[0]
        defaults = {s.name: (str(row[s.name]) if s.kind == "binary" else int(row[s.name])) for s in specs}
        st.info(f"EXAMPLE: values copied from anonymous test record #{example_id}. They are only starting "
                "values - you can edit them. The record's real final grade is not shown to the model.")
    else:
        st.warning("Test records are not available; using neutral defaults.")

suffix = f"{source[:7]}_{example_id}"
values: dict = {}
with st.form("prediction_form"):
    for group in dict.fromkeys(s.group for s in specs):
        st.markdown(f"**{group}**")
        cols = st.columns(3)
        for i, s in enumerate([x for x in specs if x.group == group]):
            with cols[i % 3]:
                key = f"in_{s.name}_{suffix}"
                if s.kind == "binary":
                    opts = list(s.options)
                    values[s.name] = st.selectbox(s.label, opts, index=opts.index(str(defaults[s.name])),
                                                  help=s.help, key=key)
                else:
                    values[s.name] = st.number_input(s.label, min_value=int(s.min), max_value=int(s.max),
                                                     value=int(defaults[s.name]), step=1, help=s.help, key=key)
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        result = pred.predict_one(values)
        st.session_state["last_prediction"] = result
    except ValueError as exc:
        st.error(str(exc))

result = st.session_state.get("last_prediction")
if result:
    st.subheader("Result")
    c1, c2, c3 = st.columns(3)
    with c1:
        ui.kpi_card("Probability of being at risk", f"{result['probability_at_risk']:.0%}",
                    f"final grade below {pass_mark:g}")
    with c2:
        ui.kpi_card("Model's call", result["predicted_class"], f"threshold {result['threshold']}")
    with c3:
        ui.kpi_card("Risk band", result["risk_band"].capitalize(), "lower / moderate / higher")
    st.progress(min(max(result["probability_at_risk"], 0.0), 1.0))
    st.caption("The probability comes from the trained ANN. It reflects patterns in past student records, "
               "is not a guarantee, and says nothing about the student's ability. With a small training "
               "set the estimate is uncertain; treat it as a prompt to plan revision.")
    st.page_link("pages/3_Recommendations.py", label="See study recommendations for this prediction", icon="📝")

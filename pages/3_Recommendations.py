"""Page 3: rule-based study suggestions with the evidence behind each one."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import ui_components as ui  # noqa: E402
from src.recommendations import LIMITATION_NOTE, generate_recommendations  # noqa: E402

ui.setup_page("Recommendations", "📝")
ctx = ui.get_context()
ui.page_header("Personalized Recommendations",
               "Suggestions come from transparent rules - not from the neural network and not from generative AI.")
ui.demo_banner(ctx.metadata)

if ctx.predictor is None:
    st.error(ctx.error or "Model unavailable.")
    ui.empty_state(ui.NO_MODEL_HELP)
    st.stop()

last = st.session_state.get("last_prediction")
options = ["Example record from the held-out test data"]
if last:
    options.insert(0, "My latest prediction (from the Prediction page)")
choice = st.radio("Source of the student information", options, horizontal=True)

result = None
if choice.startswith("My latest"):
    result = last
else:
    test_path = ctx.dirs["processed"] / "test.csv"
    if not test_path.exists():
        ui.empty_state("No test records found. Make a prediction on the Prediction page first.")
        st.stop()
    test_df = pd.read_csv(test_path)
    rid = st.selectbox("Example record (anonymous id)", test_df["record_id"].tolist())
    row = test_df[test_df["record_id"] == rid].iloc[0]
    raw = {s.name: (str(row[s.name]) if s.kind == "binary" else int(row[s.name])) for s in ctx.predictor.specs}
    result = ctx.predictor.predict_one(raw)
    st.info(f"EXAMPLE: anonymous test record #{rid}.")

prob = result["probability_at_risk"]
rec_cfg = ctx.cfg["recommendations"]
c1, c2 = st.columns(2)
with c1:
    ui.kpi_card("Model's probability of being at risk", f"{prob:.0%}", result["predicted_class"])
with c2:
    ui.kpi_card("Risk band", result["risk_band"].capitalize(), "lower / moderate / higher")

st.subheader("Evidence used")
st.dataframe(pd.DataFrame([result["inputs"]]).T.rename(columns={0: "value"}))

recs = generate_recommendations(result["inputs"], prob, pass_mark=ctx.metadata["pass_mark"],
                                high_absences=rec_cfg["high_absences"], risk_high=rec_cfg["risk_high"])
st.subheader("Suggested next steps")
for r in recs:
    with st.container(border=True):
        st.markdown(f"**{r.title}** &nbsp; {ui.priority_badge(r.priority)}", unsafe_allow_html=True)
        st.write(r.advice)
        with st.expander("Why this suggestion?"):
            st.write(f"**Evidence:** {r.evidence}")
            st.write(f"**Rule:** `{r.rule_id}` (see src/recommendations.py)")
st.caption(LIMITATION_NOTE)

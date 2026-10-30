"""Home dashboard. Run with:  streamlit run app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

from src import ui_components as ui  # noqa: E402

ui.setup_page("Home", "🎓")
ctx = ui.get_context()
report = ui.load_report(ctx)

ui.hero("AI-Based Personalized Learning and Student Performance Prediction System",
        "A Data-Driven Approach to Identifying Learning Gaps and Guiding Focused Revision")
ui.demo_banner(ctx.metadata)

st.write("This dashboard explores secondary-school Mathematics results, uses a trained **Artificial Neural "
         "Network** to estimate whether a student is **at risk of not passing** (final grade below 10 on a "
         "0-20 scale), and turns the available indicators into **transparent study suggestions**. "
         "It supports revision planning; it does not measure a student's ability and its predictions are "
         "not guarantees.")

c1, c2, c3, c4 = st.columns(4)
with c1:
    n = ctx.metadata["n_rows_after_cleaning"] if ctx.metadata else (len(ctx.data) if ctx.data is not None else None)
    ui.kpi_card("Student records", f"{n}" if n else "n/a", "UCI Student Performance (Math)")
with c2:
    share = ctx.metadata["at_risk_share"] if ctx.metadata else None
    ui.kpi_card("At-risk share", f"{share:.0%}" if share is not None else "n/a", "final grade below pass mark")
with c3:
    if report:
        auc = report["test"]["ANN"]["roc_auc"]
        ui.kpi_card("ANN test ROC-AUC", f"{auc:.2f}" if auc is not None else "n/a", f"{report['n_test']} unseen students")
    else:
        ui.kpi_card("ANN test ROC-AUC", "not evaluated", "run scripts/evaluate.py")
with c4:
    if report:
        ui.kpi_card("ANN test F1", f"{report['test']['ANN']['f1']:.2f}", "at-risk class, threshold " + str(report["threshold"]))
    else:
        ui.kpi_card("ANN test F1", "not evaluated", "run scripts/evaluate.py")

if ctx.predictor is None:
    st.warning("The model has not been trained yet, so prediction pages are disabled.")
    st.code(ui.NO_MODEL_HELP)

st.subheader("How it works")
st.markdown("""
1. **Data**: anonymised school records (grades, study habits, support).
2. **Preprocessing**: encode yes/no answers, fill gaps, scale numbers. Learned from training data only.
3. **ANN**: a small neural network outputs the probability of being *at risk*.
4. **Recommendation rules**: a separate, human-readable rule set (not AI) turns the inputs into suggestions.
5. **Dashboard**: you explore, predict and read the evidence behind every suggestion.
""")

st.subheader("Go to")
a, b, c, d, e = st.columns(5)
with a:
    st.page_link("pages/1_Performance_Analysis.py", label="Performance Analysis", icon="📊")
with b:
    st.page_link("pages/2_Prediction.py", label="Prediction", icon="🎯")
with c:
    st.page_link("pages/3_Recommendations.py", label="Recommendations", icon="📝")
with d:
    st.page_link("pages/4_Model_Insights.py", label="Model Insights", icon="🔍")
with e:
    st.page_link("pages/5_About.py", label="About", icon="ℹ️")

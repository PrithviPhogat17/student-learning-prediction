"""Page 1: explore the (anonymised) dataset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import streamlit as st  # noqa: E402

from src import ui_components as ui  # noqa: E402
from src.schema import LABEL_COLUMN, TARGET_COLUMN  # noqa: E402

ui.setup_page("Performance Analysis", "📊")
ctx = ui.get_context()
ui.page_header("Student Performance Analysis", "Explore grades and study indicators. Records are anonymous row numbers.")
ui.demo_banner(ctx.metadata)

df = ctx.data
if df is None:
    ui.empty_state("No dataset found. Run `python scripts/download_data.py` (or `python scripts/make_demo_data.py` "
                   "for synthetic demo data), then refresh.")
    st.stop()

pass_mark = ctx.cfg["dataset"]["pass_mark"]
st.sidebar.header("Filters")
outcome = st.sidebar.radio("Outcome", ["All", "At risk (G3 below pass mark)", "Not at risk"])
study = st.sidebar.multiselect("Weekly study time band (1-4)", sorted(df["studytime"].dropna().unique().tolist()))
fails = st.sidebar.multiselect("Past failures", sorted(df["failures"].dropna().unique().tolist()))
max_abs = int(df["absences"].max())
abs_range = st.sidebar.slider("Absences", 0, max_abs, (0, max_abs))
g2_range = st.sidebar.slider("Period 2 grade (G2)", 0, 20, (0, 20))

view = df.copy()
if outcome.startswith("At risk"):
    view = view[view[LABEL_COLUMN] == 1]
elif outcome == "Not at risk":
    view = view[view[LABEL_COLUMN] == 0]
if study:
    view = view[view["studytime"].isin(study)]
if fails:
    view = view[view["failures"].isin(fails)]
view = view[view["absences"].between(*abs_range) & view["G2"].between(*g2_range)]

if view.empty:
    st.warning("No students match these filters. Widen the filters in the sidebar.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
with c1:
    ui.kpi_card("Students shown", f"{len(view)}", f"of {len(df)}")
with c2:
    ui.kpi_card("Mean final grade", f"{view[TARGET_COLUMN].mean():.1f}", "G3, scale 0-20")
with c3:
    ui.kpi_card("Mean period-2 grade", f"{view['G2'].mean():.1f}", "G2")
with c4:
    ui.kpi_card("At-risk share", f"{view[LABEL_COLUMN].mean():.0%}", f"G3 below {pass_mark:g}")

left, right = st.columns(2)
with left:
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.hist([view[view[LABEL_COLUMN] == 0][TARGET_COLUMN], view[view[LABEL_COLUMN] == 1][TARGET_COLUMN]],
            bins=range(0, 22), stacked=True, color=[ui.TEAL, ui.NAVY], label=["Not at risk", "At risk"])
    ax.axvline(pass_mark, ls="--", color="grey")
    ax.set_xlabel("Final grade (G3)"); ax.set_ylabel("Students"); ax.legend(); ax.set_title("Final grade distribution")
    st.pyplot(fig); plt.close(fig)
with right:
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    colors = view[LABEL_COLUMN].map({0: ui.TEAL, 1: ui.NAVY})
    ax.scatter(view["G2"], view[TARGET_COLUMN], c=colors, alpha=.6, s=18)
    ax.set_xlabel("Period 2 grade (G2)"); ax.set_ylabel("Final grade (G3)"); ax.set_title("Earlier vs final grade")
    st.pyplot(fig); plt.close(fig)

left, right = st.columns(2)
with left:
    by_study = view.groupby("studytime")[TARGET_COLUMN].mean()
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.bar(by_study.index.astype(str), by_study.values, color=ui.TEAL)
    ax.set_xlabel("Study-time band"); ax.set_ylabel("Mean G3"); ax.set_title("Mean final grade by weekly study time")
    st.pyplot(fig); plt.close(fig)
with right:
    by_fail = view.groupby("failures")[TARGET_COLUMN].mean()
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.bar(by_fail.index.astype(str), by_fail.values, color=ui.NAVY)
    ax.set_xlabel("Past failures"); ax.set_ylabel("Mean G3"); ax.set_title("Mean final grade by past failures")
    st.pyplot(fig); plt.close(fig)

corr = view[["G1", "G2", TARGET_COLUMN]].corr()
st.markdown(f"**Reading the charts.** Earlier grades track the final grade closely (G2 vs G3 correlation "
            f"{corr.loc['G2', TARGET_COLUMN]:.2f} in this selection). That is why predicting the final grade "
            "*without* earlier grades is a much harder, and more useful, problem. These are associations "
            "in one school sample, not proof that any factor causes better or worse results.")

st.subheader("Descriptive statistics")
st.dataframe(view[["G1", "G2", TARGET_COLUMN, "studytime", "absences", "failures"]].describe().round(2))
st.subheader("Selected records (first 100)")
st.dataframe(view.head(100), hide_index=True)

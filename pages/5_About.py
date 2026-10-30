"""Page 5: project information."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from src import ui_components as ui  # noqa: E402

ui.setup_page("About", "ℹ️")
ui.page_header("About the Project")

st.markdown("""
**Title:** AI-Based Personalized Learning and Student Performance Prediction System
**Subtitle:** A Data-Driven Approach to Identifying Learning Gaps and Guiding Focused Revision

### Objective and problem
Students often discover too late that they are falling behind. The project predicts, from earlier grades and
study-related indicators, whether a student is at risk of finishing below the pass mark, and turns the available
evidence into simple revision suggestions.

### Proposed solution
A trained Artificial Neural Network (TensorFlow/Keras) estimates the probability of being *at risk*. A separate
rule-based engine converts the inputs into suggestions and explains each one. A Streamlit dashboard ties it together.

### Technology stack
Python, TensorFlow/Keras, scikit-learn, pandas, NumPy, Matplotlib, Streamlit, PyYAML, joblib, unittest/pytest.

### Dataset
UCI *Student Performance* (Mathematics file), Cortez (2008), DOI 10.24432/C5TG7T, licensed CC BY 4.0.
Two Portuguese secondary schools; grades on a 0-20 scale; pass mark 10. Records are anonymous.

### Methodology
Stratified train/validation/test split; preprocessing fitted on training data only; ANN with early stopping;
comparison with logistic-regression, majority-class and a simple "G2 below pass mark" rule baseline;
bootstrap intervals and permutation importance.

### Known limitations
* About 395 records from two schools in one country: results may not transfer elsewhere.
* The test set is small, so metrics are noisy.
* The data has no topic-level results, so recommendations are general.
* G1/G2 strongly predict G3; without them the task is much harder.
* The model estimates risk; it does not measure ability, effort or potential, and it is not a decision tool.

### Future enhancements
Cross-validation, a harder no-prior-grade experiment, richer datasets (e.g. OULAD or ASSISTments) with
topic/skill-level signals, calibration analysis, and a fairness review.

### Student and course
Prithvi Phogat | Registration No. 2430010010 | Department of Data Science, Manipal University Jaipur
Course: Deep Learning (DSE3120), fifth semester | Supervisor: Dr. DP Sharma
""")

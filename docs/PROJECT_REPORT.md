# AI-Based Personalized Learning and Student Performance Prediction System
### A Data-Driven Approach to Identifying Learning Gaps and Guiding Focused Revision

---

## 1. Title Page

| | |
|---|---|
| **Project title** | AI-Based Personalized Learning and Student Performance Prediction System |
| **Subtitle** | A Data-Driven Approach to Identifying Learning Gaps and Guiding Focused Revision |
| **Student** | Prithvi Phogat |
| **Registration number** | 2430010010 |
| **Institution** | Manipal University Jaipur |
| **Department** | Data Science |
| **Course** | Deep Learning (DSE3120), fifth semester |
| **Supervisor** | Dr. DP Sharma |
| **Date of submission** | ____________ |

---

## 2. Abstract
This project builds a web-based educational analytics application that estimates whether a secondary-school student is at
risk of finishing a Mathematics course below the pass mark, and converts the available indicators into transparent revision
suggestions. A Multi-Layer Perceptron implemented in TensorFlow/Keras is trained on the UCI *Student Performance* dataset
(Mathematics file). The target is binary: final grade G3 below 10 on the 0-20 scale. Inputs are earlier period grades and
study-related indicators; sensitive and demographic attributes are excluded. The network is compared with majority-class,
logistic-regression and simple rule baselines on a held-out test set, with bootstrap intervals and permutation importance. A
separate rule-based engine, which is not a learned or generative model, provides suggestions. A Streamlit dashboard presents
data exploration, prediction, recommendations and model diagnostics. *Quantitative findings are reported in Section 15 after
the experiment is run; they are intentionally not stated here.*

## 3. Introduction
Early identification of students who may need additional support is a common goal of learning analytics. Tabular school
records (grades, attendance, study habits) are cheap to collect, and neural networks can model non-linear combinations of such
indicators. This project applies the standard Deep Learning workflow - data inspection, preprocessing, splitting, training,
evaluation, deployment in an application - to a small, public educational dataset, and is designed to be fully reproducible on a
laptop CPU.

## 4. Problem Statement
Given a student's earlier grades and study-related indicators, estimate the probability that the student's final grade falls
below the pass mark, and present understandable, evidence-linked study suggestions, without claiming to measure ability or
guaranteeing outcomes.

## 5. Objectives
1. Select a suitable open dataset and define a defensible prediction target.
2. Build a leakage-safe preprocessing pipeline.
3. Train and evaluate an ANN and compare it honestly with simpler baselines.
4. Build a rule-based recommendation engine separate from the ANN.
5. Deliver an interactive dashboard, automated tests and documentation.

## 6. Literature and Related Work
*Citations 1-2 were checked against their official pages while preparing this project; items 3-6 are well-known works cited
from the author's knowledge and **should be re-checked for exact page numbers before submission**.*

* Cortez and Silva [1] introduced the dataset used here and modelled it with classification and regression, noting that G3 is
  strongly correlated with G1 and G2 and that prediction without them is harder but more useful.
* Kuzilek, Hlosta and Zdrahal [2] released the Open University Learning Analytics Dataset (OULAD), a larger click-stream and
  assessment dataset, considered here as a future extension.
* Adam [3] is the optimizer used; dropout [4] and L2 regularisation are the regularisation techniques.
* Grinsztajn et al. [5] report that tree-based models often remain competitive with or better than neural networks on typical
  tabular data, which motivates the inclusion of simple baselines and cautious conclusions.
* scikit-learn [6] supplies preprocessing, baselines and metrics.

## 7. Dataset Description
UCI Student Performance (Cortez, 2008; DOI 10.24432/C5TG7T; CC BY 4.0). Data from two Portuguese secondary schools collected by
school reports and questionnaires. The Mathematics file (`student-mat.csv`, 395 students) is used; the Portuguese file is not
merged because many students appear in both files, which would leak information across splits. The source reports no missing
values. Grades use a 0-20 scale. G1, G2, G3 are the period-1, period-2 and final grades.
Dataset page: https://archive.ics.uci.edu/dataset/320/student+performance

**Target.** `at_risk = 1` if G3 < 10 else 0 (10 is the pass mark on the Portuguese scale; configurable).
**Inputs (14).** G1, G2, studytime, absences, failures, traveltime, freetime, goout, schoolsup, famsup, paid, activities, higher,
internet. **Excluded:** G3 (target definition), sex, age, address, famsize, Pstatus, Medu, Fedu, Mjob, Fjob, reason, guardian,
nursery, romantic, famrel, Dalc, Walc, health, school - to avoid profiling on personal characteristics and to keep suggestions
actionable.

## 8. Proposed Methodology
Dataset -> inspection -> cleaning -> stratified 60/20/20 split -> preprocessing fitted on train only -> ANN training with
validation-based early stopping -> test-set evaluation against baselines -> saved artifacts -> prediction -> rule-based
recommendations -> dashboard.

## 9. Data Preprocessing
* Inspection: shape, dtypes, missing values, duplicates, out-of-range values (`reports/data_inspection.json`).
* Cleaning: values outside the documented ranges become missing; rows without a final grade would be dropped (none expected).
* Encoding: yes/no -> 1/0. Ordinal scales (1-4, 1-5) are treated as numeric.
* Imputation (median) and standardisation, both fitted on the training split only and reused for validation, test and dashboard input.
* Split: stratified on the label, seed 42. One row is one student, so no student appears in two splits.

## 10. Deep Learning Architecture
`Input(14) -> Dense(64, ReLU) -> Dropout(0.3) -> Dense(32, ReLU) -> Dropout(0.3) -> Dense(1, sigmoid)`.
Loss: binary cross-entropy. Optimizer: Adam (learning rate 0.001). L2 regularisation 1e-4. Batch size 32, at most 300 epochs, early
stopping on validation loss (patience 25, best weights restored), checkpointing of the best model, balanced class weights.
The sigmoid output is the probability of being at risk; a threshold of 0.5 (fixed before testing) gives the class.

## 11. System Workflow
See the Mermaid diagram in the README. Training is run once (`scripts/train.py`); the dashboard only loads saved artifacts.

## 12. Application Design
Streamlit multi-page app: Home (KPIs from real evaluation files), Performance Analysis (filters, charts), Prediction (form built
from the feature schema, validated), Recommendations (rules, evidence, limitations), Model Insights (metrics, baselines, curves,
permutation importance), About. Navy/teal theme. Missing models or data lead to explanatory empty states. The recommendation
engine (`src/recommendations.py`) is separate from the ANN; each suggestion lists its rule and evidence. The dataset has no
topic-level information, so no topic-specific advice is given.

## 13. Experimental Setup
Python, TensorFlow/Keras, scikit-learn, pandas, NumPy on a CPU. Configuration in `config/config.yaml`. Seeds are fixed (split seed 42;
Keras global seed set before training); exact bit-for-bit repeatability across machines is not guaranteed by TensorFlow.
Baselines: majority class; logistic regression (balanced class weights); rule "at risk if G2 < 10". An additional experiment without
G1/G2 (`--no-prior-grades`) tests the harder, more useful task.

## 14. Evaluation Metrics
Accuracy, balanced accuracy, precision, recall, F1 (positive class = at risk), ROC-AUC, confusion matrix, 95% percentile bootstrap
intervals over the test set, permutation importance (drop in ROC-AUC when one input is shuffled). Accuracy alone is insufficient
if classes are imbalanced.

## 15. Results and Discussion
**STATUS: PENDING.** The experiment must be run on the student's computer (the project was authored in an environment without
TensorFlow or internet access, so no real results exist yet). After running

    python scripts/download_data.py
    python scripts/train.py
    python scripts/evaluate.py

paste the contents of `reports/results_section.md` here and insert `reports/figures/training_curves.png`, `confusion_matrix.png`,
`roc_curves.png`, `model_comparison.png`, `permutation_importance.png`. Then discuss, using only what the numbers show:
1. Does the ANN beat logistic regression and the G2 rule, and is the difference larger than the bootstrap intervals?
2. Recall and F1 for the at-risk class and the cost of false negatives versus false positives.
3. Training/validation curves: signs of overfitting? epoch of early stopping?
4. Which inputs the model relies on (not causal).
5. Results of the `--no-prior-grades` experiment, if run.

## 16. Limitations
Small sample from two schools; noisy test estimates; strong dependence on G1/G2; no topic-level data; ordinal variables treated as numeric;
self-reported questionnaire items; predictions are not guarantees and must not be used to make decisions about individual students;
importance scores are not causal; no real-world deployment or validation.

## 17. Conclusion
*(Write after Section 15 is filled in.)* The project delivers a complete, reproducible Deep Learning workflow and application. Whether the ANN
adds value over simpler models on this dataset is an empirical question answered by the results in Section 15.

## 18. Future Scope
Cross-validation and multiple seeds; probability calibration; tree-based baselines; OULAD or ASSISTments for activity- or skill-level
modelling; fairness analysis; SHAP; deployment with teacher feedback.

## 19. References
1. P. Cortez and A. Silva, "Using data mining to predict secondary school student performance," in *Proc. 5th Annual Future Business
   Technology Conference (FUBUTEC)*, 2008. Dataset: P. Cortez (2008), Student Performance, UCI Machine Learning Repository,
   https://doi.org/10.24432/C5TG7T.
2. J. Kuzilek, M. Hlosta and Z. Zdrahal, "Open University Learning Analytics dataset," *Scientific Data*, vol. 4, 170171, 2017,
   doi:10.1038/sdata.2017.171.
3. D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," ICLR 2015 (arXiv:1412.6980).
4. N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from
   overfitting," *Journal of Machine Learning Research*, vol. 15, pp. 1929-1958, 2014.
5. L. Grinsztajn, E. Oyallon and G. Varoquaux, "Why do tree-based models still outperform deep learning on typical tabular data?",
   NeurIPS 2022 Datasets and Benchmarks Track.
6. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.

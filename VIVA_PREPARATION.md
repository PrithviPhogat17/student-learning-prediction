# Viva Preparation - Student Performance Prediction (DSE3120)

Answers are tailored to **this** implementation. Fill in your own real numbers (from `reports/model_evaluation.txt`) wherever
a question asks about results; never quote numbers you did not obtain.

## A. The project and the data
**1. What problem does the project solve?**
It estimates whether a secondary-school student is at risk of finishing Mathematics below the pass mark, and turns the available indicators
into revision suggestions, so support can start earlier.

**2. Why did you choose the UCI Student Performance dataset?**
It is small and easy to download, openly licensed (CC BY 4.0), well documented, contains grades plus study indicators and is easy to explain. OULAD is
richer but large and multi-table; ASSISTments suits skill-level correctness prediction. I checked the UCI page for columns and licence; neither of the other two was
needed for a working, explainable project.

**3. What is the prediction target?**
`at_risk` = 1 if the final grade G3 is below 10 (pass mark on the 0-20 scale), otherwise 0. The model outputs the probability of being at risk.

**4. Why a classification and not a regression of G3?**
A pass/at-risk question is what a teacher acts on, it gives probabilities, a confusion matrix, recall and ROC-AUC, and it avoids implying false precision about an exact grade.

**5. What are the inputs and what did you leave out?**
14 inputs: G1, G2, studytime, absences, failures, traveltime, freetime, goout, schoolsup, famsup, paid, activities, higher, internet. G3 is excluded (it defines the target).
I also excluded sex, age, address, family structure, parents' education/jobs, alcohol use, romantic status and health, so the model does not profile students on personal
characteristics and so suggestions stay actionable.

**6. Is there target leakage?**
G3 is never an input. G1 and G2 are earlier grades, available before the final grade, so they are legitimate for an early-warning use, but they make the task easier;
I provide `--no-prior-grades` to test the harder task. Preprocessing is fitted only on training data.

**7. Why is it a Deep Learning project and not just a website?**
The prediction comes from a trained neural network (TensorFlow/Keras MLP) whose weights are learned by backpropagation from data. No external API or LLM is used.

## B. Neural network concepts
**8. What is an Artificial Neural Network?**
A model made of layers of simple units (neurons). Each computes a weighted sum of its inputs plus a bias and passes it through an activation function; stacking layers lets it learn
non-linear patterns.

**9. What are neurons, weights, biases and activation functions?**
A neuron computes z = w.x + b. Weights w say how strongly each input matters; the bias b shifts the result; the activation function (e.g. ReLU) adds non-linearity so that
several layers are more powerful than one.

**10. Why ReLU in the hidden layers?**
ReLU(z) = max(0, z): cheap, avoids the vanishing-gradient problem of sigmoid in hidden layers and works well in practice.

**11. What is the output activation and why?**
A single sigmoid unit, because the task is binary; it squashes the output to (0, 1), interpreted as the probability of being at risk.

**12. What is the loss function?**
Binary cross-entropy, which penalises confident wrong probabilities heavily and is the natural loss for a sigmoid output.

**13. What is an optimizer? Which one and why?**
It updates the weights to reduce the loss. I use Adam (learning rate 0.001): adaptive step sizes, robust defaults, little tuning.

**14. What is backpropagation?**
The chain-rule algorithm that computes how much each weight contributed to the loss (the gradients) by propagating the error backwards through the layers; the optimizer then uses
those gradients to update the weights.

**15. What is your architecture?**
Input(14) -> Dense 64 (ReLU) -> Dropout 0.3 -> Dense 32 (ReLU) -> Dropout 0.3 -> Dense 1 (sigmoid). It is small on purpose: the dataset has only ~395 rows.

**16. What is overfitting and how do you fight it?**
Overfitting = low training loss but poor performance on new data. I use a validation set with early stopping (patience 25, best weights restored), dropout 0.3, L2 weight decay 1e-4, and a small network.

**17. What is dropout?**
During training each hidden unit is randomly switched off with probability 0.3 so the network cannot rely on single units; at prediction time all units are used.

**18. What are epochs, batch size and learning rate?**
An epoch is one pass over the training data; batch size (32) is how many samples are used per weight update; learning rate (0.001) is the step size of each update.

**19. Why use class weights?**
The at-risk class is likely the minority; balanced weights make mistakes on it count more so the model does not just favour the majority class.

## C. Data handling and evaluation
**20. Why training, validation and test sets?**
Training data fits the weights; validation data is used during training for early stopping; test data is touched only at the end to estimate performance on unseen students. Mixing them
gives over-optimistic results. I used a stratified 60/20/20 split.

**21. What is data leakage?**
When information that would not be available at prediction time (or from the test set) influences training, producing unrealistically good results. I avoid it by excluding G3, fitting the
scaler/imputer on training data only, and not merging the Math and Portuguese files (many students are in both).

**22. Why is preprocessing necessary?**
Networks train poorly when inputs have very different scales (absences 0-93 vs. grades 0-20), and they cannot read text like "yes/no". I encode yes/no as 1/0, impute missing values with the median, and standardise.

**23. Why fit the scaler only on training data?**
Otherwise the test set's statistics leak into training. The same fitted scaler is then applied to validation, test and dashboard inputs; it is saved in `preprocessor.joblib`.

**24. What baselines did you use and why?**
Majority-class (sets a floor), logistic regression (simple, strong on tabular data) and the rule "at risk if G2 < 10". They show whether the ANN adds value; if it does not beat them, that is a legitimate finding.

**25. Which metrics did you use?**
Accuracy, balanced accuracy, precision, recall, F1, ROC-AUC and the confusion matrix. With a minority at-risk class accuracy alone can mislead, so recall and F1 for the at-risk class matter most.

**26. What does the confusion matrix show?**
Rows are actual classes and columns predicted: true negatives, false positives, false negatives, true positives. For this application false negatives (an at-risk student not flagged) are the costly error.

**27. What do precision, recall and F1 mean here?**
Precision: of students flagged at risk, the share truly at risk. Recall: of truly at-risk students, the share flagged. F1: their harmonic mean.

**28. What does ROC-AUC mean?**
The probability that a randomly chosen at-risk student gets a higher predicted risk than a randomly chosen not-at-risk student; 0.5 is chance, 1.0 is perfect ranking.

**29. Why bootstrap intervals?**
The test set has only about 79 students, so a single metric is noisy. Resampling the test set shows how much it could vary.

**30. What does permutation importance tell you?**
How much the test ROC-AUC drops when one input is shuffled, i.e. how much the trained model relies on it. It is not causal: a high value does not mean changing that factor changes a student's result.

## D. Recommendations, application and limits
**31. How does the recommendation engine work? Is it AI?**
It is a set of hand-written if-then rules in `src/recommendations.py` (e.g. G2 below 10, grade drop, many absences, low study time). It is not learned, not the ANN, and not generative AI. Each
suggestion displays the rule id and the evidence that triggered it.

**32. Is the system a generative-AI chatbot?**
No. There is no LLM or external API. The ANN outputs one probability; the rest is rules and a dashboard.

**33. Why can't you recommend specific weak topics?**
The dataset has only overall period grades and habit indicators, no topic-level results. Claiming topic mastery would be invented, so the suggestions stay general and say so.

**34. What are the limitations?**
Small sample from two schools; noisy test metrics; strong reliance on G1/G2; no topic data; self-reported variables; predictions are estimates, not guarantees; not validated in a real school; must not be used to decide
about individuals.

**35. Why might a neural network not outperform a simpler model?**
With few rows and mostly simple relationships, logistic regression can capture nearly everything and the network adds variance and tuning effort. Research on tabular data reports the same pattern. I therefore report the baselines honestly.

**36. Why did you use a scikit-learn fallback backend, and is it your main model?**
Only as an explicit, labelled option if TensorFlow cannot be installed on a computer. The main model is the Keras ANN; the backend name is stored in metadata and shown in the dashboard.

**37. How do you make sure the app uses the same preprocessing as training?**
The fitted preprocessing pipeline (encoder + imputer + scaler) is saved and loaded by the predictor, and every input is validated against the schema before transformation. A test checks reload consistency.

**38. How would you improve the project?**
Cross-validation and repeated seeds, calibration plots, tree-based baselines, a study without G1/G2, OULAD or ASSISTments for activity- or skill-level signals, SHAP, and a fairness review.

**39. What would you say if the ANN performs worse than logistic regression?**
That I report it as is: on a ~395-row tabular dataset a simple model can be as good or better, which is a useful result about model choice, not a failure of the project.

**40. What did you actually verify when building it?**
Data loading, leakage and split checks, preprocessing, validation, recommendations and the train-save-load-evaluate-predict chain were tested (using synthetic demo data and the scikit-learn fallback in an environment without TensorFlow).
The Keras training, the real dataset and the Streamlit UI in a browser must be run on my own machine - the README states this plainly.

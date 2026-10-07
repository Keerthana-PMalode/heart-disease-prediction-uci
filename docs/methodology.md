# Machine-Learning Methodology

## 1. Workflow

The project follows this sequence:

```text
Dataset validation
      ↓
Exploratory data analysis
      ↓
Target preparation
      ↓
Stratified train/test split
      ↓
Leakage-safe preprocessing pipelines
      ↓
Baseline models
      ↓
Advanced XGBoost model
      ↓
Five-fold stratified cross-validation
      ↓
Held-out test evaluation
      ↓
Model comparison and interpretation
```

## 2. Data Validation

The Cleveland dataset is loaded through `src.preprocessing.load_data()` and checked with `validate_columns()`. The expected schema contains 13 predictors and the original `num` target. Duplicate records, missing values, data types, target levels, and basic structural properties are examined before modeling.

## 3. Exploratory Data Analysis

EDA examines numerical and categorical distributions, target-stratified patterns, potential outliers, missingness, and associations. Numerical target-group comparisons use Mann–Whitney U tests with rank-biserial effect sizes. Categorical associations use Pearson chi-square where its approximation is adequate; `restecg` uses the documented Monte-Carlo chi-square procedure. Benjamini–Hochberg FDR correction is applied across the 13 final univariate tests.

EDA is descriptive and does not by itself establish causal relationships or predictive importance.

## 4. Train/Test Split

For modeling, the dataset is split into 80% training and 20% testing using stratification and `random_state=42`.

The resulting sizes are:

- Training: 242 observations
- Test: 61 observations
- Training target counts: 131 class 0 and 111 class 1
- Test target counts: 33 class 0 and 28 class 1

The test set is not used for model selection or preprocessing fitting.

## 5. Preprocessing

Reusable preprocessing is implemented in `src/preprocessing.py`.

### Numerical variables

- Median imputation
- `StandardScaler`

### Categorical variables

- Most-frequent imputation
- One-hot encoding with `handle_unknown="ignore"`

The transformed representation contains 28 features for the current dataset.

## 6. Pipeline and Leakage Prevention

Each classifier is wrapped in a scikit-learn `Pipeline` containing the shared `ColumnTransformer`. During cross-validation, the preprocessing transformer is fitted independently inside each training fold. For the final evaluation, it is fitted only on the training split.

This prevents test-set information from influencing imputation statistics, scaling parameters, or category encoding.

## 7. Models

### Baseline models

1. Logistic Regression — `max_iter=1000`, `random_state=42`
2. K-Nearest Neighbors — `n_neighbors=5`
3. Decision Tree — `random_state=42`
4. Gaussian Naive Bayes
5. Support Vector Machine — RBF kernel, `probability=True`, `random_state=42`
6. Random Forest — `n_estimators=200`, `random_state=42`

### Advanced model

XGBoost is used as the advanced/ensemble model with:

- `n_estimators=200`
- `max_depth=3`
- `learning_rate=0.05`
- `subsample=0.9`
- `colsample_bytree=0.9`
- `objective="binary:logistic"`
- `eval_metric="logloss"`
- `random_state=42`
- `n_jobs=1`

These are the established project configurations and are centralized in `src/models.py`.

## 8. Cross-Validation

Model development uses **stratified five-fold cross-validation** on the training set with `shuffle=True` and `random_state=42`. Accuracy, precision, recall, specificity, F1, and ROC-AUC are recorded. ROC-AUC variability is summarized using the fold mean and standard deviation.

The baseline selection criterion is mean cross-validation ROC-AUC.

## 9. Evaluation Metrics

- **Accuracy:** proportion of correctly classified observations.
- **Precision:** proportion of predicted-positive observations that are actually positive.
- **Recall/Sensitivity:** proportion of actual positive observations correctly identified.
- **Specificity:** true-negative rate, `TN / (TN + FP)`.
- **F1-score:** harmonic mean of precision and recall.
- **ROC-AUC:** area under the ROC curve. Positive-class probability scores are used when available; otherwise a supported decision score is used.
- **Confusion matrix:** counts of TN, FP, FN, and TP.

Metric implementation is centralized in `src/evaluation.py`.

## 10. Reproducibility

The project uses `random_state=42` for the stratified split, cross-validation shuffling, and stochastic model configurations where applicable. XGBoost uses `n_jobs=1` to keep the advanced experiment deterministic under the documented configuration as far as the underlying environment permits.

The project is organized so notebooks contain experiment workflow and interpretation, while reusable preprocessing, model definitions, evaluation, and visualization utilities reside under `src/`.

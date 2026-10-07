# Results and Discussion

## 1. Baseline Cross-Validation Results

The six baseline classifiers were evaluated using the same leakage-safe preprocessing and stratified five-fold cross-validation on the 242-observation training set. Mean values below are the established project results.

| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8471 | 0.8768 | 0.7834 | 0.9003 | 0.8245 | **0.9025** |
| Random Forest | 0.8139 | 0.8129 | 0.7739 | 0.8470 | 0.7900 | 0.8927 |
| SVM | 0.8139 | 0.8303 | 0.7648 | 0.8541 | 0.7893 | 0.8884 |
| Gaussian Naive Bayes | 0.7438 | 0.7049 | 0.8458 | 0.6564 | 0.7495 | 0.8773 |
| KNN | 0.8222 | 0.8232 | 0.7826 | 0.8544 | 0.7989 | 0.8712 |
| Decision Tree | 0.6980 | 0.6782 | 0.6842 | 0.7091 | 0.6768 | 0.6967 |

Logistic Regression has the highest mean cross-validation ROC-AUC and is therefore retained as the selected baseline under the project's predefined selection criterion.

## 2. Advanced XGBoost Results

XGBoost was evaluated using the same train/test methodology and preprocessing design.

### Five-fold cross-validation

| Metric | Mean | SD |
|---|---:|---:|
| Accuracy | 0.8016 | 0.0249 |
| Precision | 0.7848 | 0.0317 |
| Recall | 0.7834 | 0.0613 |
| Specificity | 0.8165 | 0.0387 |
| F1 | 0.7825 | 0.0345 |
| ROC-AUC | 0.8664 | 0.0330 |

**Note:** The ROC-AUC standard deviation reported in the primary cross-validation table (0.0330) differs slightly from the 0.0369 reported in the later fold-level XGBoost diagnostic. This is due to the use of different standard-deviation conventions: the primary cross-validation summary uses the population standard deviation (`ddof=0`), whereas the fold-level diagnostic uses the sample standard deviation (`ddof=1`). The mean ROC-AUC is identical in both reports (0.8664), so this difference does not indicate a discrepancy in model performance.

### XGBoost ROC-AUC fold-level summary

| Evaluation | ROC-AUC |
|---|---:|
| CV Fold 1 | 0.9080 |
| CV Fold 2 | 0.8367 |
| CV Fold 3 | 0.8427 |
| CV Fold 4 | 0.9056 |
| CV Fold 5 | 0.8392 |
| Mean CV | 0.8664 |
| CV Std | 0.0369 |
| Held-Out Test | 0.9545 |

### Held-out test set

| Metric | Logistic Regression | XGBoost |
|---|---:|---:|
| Accuracy | 0.8852 | **0.9016** |
| Precision | 0.8387 | **0.8438** |
| Recall / Sensitivity | 0.9286 | **0.9643** |
| Specificity | 0.8485 | 0.8485 |
| F1 | 0.8814 | **0.9000** |
| ROC-AUC | **0.9665** | 0.9545 |
One technical point
The explanation is mathematically correct:

- ddof=0 → population SD → 0.0330
- ddof=1 → sample SD → 0.0369

Both use the `same five fold ROC-AUC values`:

- 0.9080, 0.8367, 0.8427, 0.9056, 0.8392

and both produce the same mean:

- 0.8664

## 3. Confusion Matrices

### Logistic Regression

```text
[[28, 5],
 [ 2, 26]]
```

This corresponds to 28 true negatives, 5 false positives, 2 false negatives, and 26 true positives.

### XGBoost

```text
[[28, 5],
 [ 1, 27]]
```

This corresponds to 28 true negatives, 5 false positives, 1 false negative, and 27 true positives.

## 4. Logistic Regression vs XGBoost

On the single held-out test set, XGBoost produced higher accuracy, precision, recall, and F1-score than Logistic Regression, while both produced the same specificity. Logistic Regression produced the higher ROC-AUC on this test set.

The difference in ROC-AUC is important because ROC-AUC evaluates ranking performance across thresholds, whereas accuracy, precision, recall, specificity, and F1 at the default classification threshold are threshold-dependent. Consequently, the models do not have to have the same relative ordering on every metric.

For the project-level baseline selection, Logistic Regression remains the selected baseline because its mean five-fold training-set ROC-AUC was 0.9025 compared with 0.8664 for XGBoost.

The held-out test set contains only 61 observations, so the test-set differences should not be interpreted as proof of universal model superiority.

## 5. Interpretation of Metrics

- **Accuracy** summarizes overall classification correctness but can hide differences between positive and negative class errors.
- **Precision** is informative when false positive predictions matter.
- **Recall/Sensitivity** measures how many positive cases are detected; it is particularly useful when missed positive cases are an important concern.
- **Specificity** measures how well negative cases are identified.
- **F1-score** balances precision and recall and is useful when both types of positive-class performance matter.
- **ROC-AUC** evaluates discrimination across decision thresholds and is the project's primary baseline-selection metric.

The results demonstrate why the project reports multiple metrics rather than accuracy alone.

## 6. Strengths and Weaknesses

### Logistic Regression

**Strengths:** highest mean cross-validation ROC-AUC among the six baselines; strong specificity and precision during baseline cross-validation; simple and reproducible configuration.

**Limitations:** the model assumes a linear decision relationship in its transformed feature space and may not capture nonlinear interactions as directly as tree-based ensemble methods.

### XGBoost

**Strengths:** higher held-out test accuracy, recall, and F1-score than Logistic Regression under the fixed experiment; one fewer false negative in the held-out confusion matrix.

**Limitations:** its mean cross-validation ROC-AUC was lower than Logistic Regression in this experiment, and the observed test-set advantage on several threshold-dependent metrics is based on only 61 test observations.

## 7. Limitations and Potential Improvements

The experiment uses the 303-record Cleveland subset and a single 80:20 train/test split. Therefore, performance estimates can be sensitive to the particular sample partition. The project also evaluates fixed baseline and XGBoost configurations rather than conducting an exhaustive hyperparameter search.

Potential next steps include repeated or nested cross-validation, justified hyperparameter tuning, feature-selection experiments, calibration analysis, external validation on an independent dataset, and a controlled comparison with selected literature. Such extensions should preserve the project's leakage-prevention principles.

No clinical deployment claim is made from these experiments.

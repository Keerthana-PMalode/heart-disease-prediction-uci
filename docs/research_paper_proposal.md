# Heart Disease Prediction Using Machine Learning: A Controlled Comparison of Baseline Classifiers and XGBoost on the UCI Cleveland Dataset

## Abstract

Heart disease prediction is an important machine-learning classification problem in which predictive models can be evaluated using demographic, clinical, and diagnostic attributes. This study presents a controlled experimental comparison of six baseline machine-learning classifiers and an XGBoost model using the Cleveland subset of the UCI Heart Disease dataset. The dataset contains 303 observations and 13 predictor variables. The original five-level diagnosis variable was converted into a binary target, with `num = 0` representing absence of heart disease and `num = 1–4` representing presence.

To reduce information leakage, missing-value imputation, numerical standardization, and categorical one-hot encoding were implemented within machine-learning pipelines. The dataset was divided using an 80:20 stratified train/test split, producing 242 training observations and 61 held-out test observations. Model development used stratified five-fold cross-validation on the training set. Logistic Regression, K-Nearest Neighbors, Decision Tree, Gaussian Naive Bayes, Support Vector Machine, and Random Forest were evaluated using accuracy, precision, recall, specificity, F1-score, and ROC-AUC. Logistic Regression achieved the highest mean cross-validation ROC-AUC among the six baseline classifiers at 0.9025 and was therefore selected as the baseline model according to the predefined selection criterion.

XGBoost was subsequently evaluated using a fixed configuration and the same preprocessing and validation framework. Its mean cross-validation ROC-AUC was 0.8664. On the single held-out test set, XGBoost achieved higher accuracy, precision, recall, and F1-score than Logistic Regression, while both models achieved the same specificity. Logistic Regression, however, achieved the higher held-out ROC-AUC. These findings demonstrate that model rankings can differ across evaluation metrics and between cross-validation and a single held-out test set. The results are specific to the Cleveland dataset and the stated experimental configuration and should not be interpreted as evidence of universal model superiority or clinical validity.

**Keywords:** Heart disease prediction; machine learning; Logistic Regression; XGBoost; UCI Cleveland dataset; ROC-AUC; cross-validation; classification

---

# 1. Introduction

## 1.1 Background

Heart disease prediction represents a classification problem in which machine-learning algorithms can be used to identify patterns associated with the presence or absence of disease. Machine-learning methods provide a framework for comparing different predictive algorithms under common experimental conditions.

A major challenge in comparing machine-learning approaches is that published studies frequently differ in their datasets, preprocessing procedures, feature-selection strategies, train/test partitions, validation procedures, model configurations, and evaluation metrics. Consequently, reported performance values from different studies cannot always be compared directly.

## 1.2 Motivation

The present study was motivated by the need for a controlled and reproducible comparison of commonly used classification algorithms using a single dataset and a common preprocessing and validation framework.

The study therefore evaluates six baseline classifiers before evaluating XGBoost as an advanced ensemble model.

The analysis emphasizes multiple performance measures rather than accuracy alone, including:

* Accuracy
* Precision
* Recall/sensitivity
* Specificity
* F1-score
* ROC-AUC
* Confusion matrices

## 1.3 Research Question

The central research question is:

> **How effectively can machine-learning algorithms predict heart disease using the UCI Cleveland dataset, and how do the baseline Logistic Regression model and XGBoost differ under a controlled preprocessing, cross-validation, and held-out test evaluation framework?**

## 1.4 Study Contribution

The contribution of this study is methodological rather than a claim of a new state-of-the-art algorithm.

The study provides:

1. A reproducible experiment using the Cleveland dataset.
2. A common leakage-safe preprocessing pipeline.
3. A controlled comparison of six baseline classifiers.
4. Evaluation of XGBoost under the same experimental framework.
5. Multiple evaluation metrics rather than accuracy alone.
6. Explicit separation between cross-validation results and held-out test results.
7. Cautious comparison with previously published studies.

This is a controlled machine-learning experiment and **not a clinical validation study**.

---

# 2. Related Work and Literature Review

## 2.1 Overview

The project literature review contains 20 research papers covering traditional classifiers, ensemble methods, feature selection, hybrid approaches, deep-learning methods, and Cleveland-dataset experiments.

The studies include:

1. Palaniappan and Awang (2008)
2. Sultana, Haider and Uddin (2016)
3. Abdar et al. (2015)
4. Arabasadi et al. (2017)
5. Uyar and Ilhan (2017)
6. Pahwa and Kumar (2017)
7. Sharma and Saxena (2017)
8. Haq et al. (2018)
9. Dwivedi (2018)
10. Amin, Chiam and Varathan (2019)
11. Mohan, Thirumalai and Srivastava (2019)
12. Latha and Jeeva (2019)
13. Ali et al. (2019)
14. Al-Makhadmeh and Tolba (2019)
15. Tougui, Jilbab and El Mhamdi (2020)
16. Khan et al. (2025)
17. Kavitha et al. (2021)
18. Bharti et al. (2021)
19. Rani et al. (2021)
20. Teekaraman et al. (2022)

## 2.2 Traditional Classifiers

The reviewed literature repeatedly investigates Logistic Regression, Support Vector Machines, KNN, Decision Trees, Naive Bayes, and Random Forest.

This supports establishing multiple conventional classifiers as baseline models rather than selecting a single algorithm in isolation.

## 2.3 Ensemble and Advanced Methods

Random Forest, boosting, hybrid models, genetic-algorithm approaches, fuzzy systems, and neural-network approaches occur throughout the literature.

Teekaraman et al. (2022) is particularly relevant because the supplied literature review identifies a Cleveland-dataset experiment involving SVM, Gaussian Naive Bayes, Logistic Regression, LightGBM, XGBoost, and Random Forest.

## 2.4 Feature Selection

Feature-selection approaches appear in several reviewed studies.

For example, Haq et al. investigated Relief, mRMR, and LASSO, while Amin et al. investigated significant features and voting-based approaches.

These studies motivate feature-selection analysis as a potential extension, but feature selection was **not used to alter the current 13-feature experiment solely on the basis of EDA significance**.

## 2.5 Research Gaps

The supplied literature review identifies five major gaps:

1. Lack of uniform experimental conditions.
2. Differences between datasets and experimental settings.
3. Excessive emphasis on accuracy in some studies.
4. Reproducibility challenges.
5. Need for controlled comparisons.

The present study addresses these gaps at the project level by maintaining a common preprocessing pipeline, fixed train/test split, stratified cross-validation, explicit metrics, and reproducible model configurations.

---

# 3. Dataset Description

## 3.1 Dataset Source

The experiment uses the **Cleveland subset of the UCI Heart Disease dataset**.

The project documentation identifies:

* Dataset: UCI Heart Disease — Cleveland subset
* Dataset ID: 45
* Observations: 303
* Predictor variables: 13
* Original target: `num`

The project uses the Cleveland data file as the modeling source.

## 3.2 Predictor Variables

The 13 predictors are:

### Numerical variables

1. `age`
2. `trestbps`
3. `chol`
4. `thalach`
5. `oldpeak`

### Categorical variables

1. `sex`
2. `cp`
3. `fbs`
4. `restecg`
5. `exang`
6. `slope`
7. `ca`
8. `thal`

The distinction is important because the categorical codes represent discrete categories rather than continuous measurements.

In particular, `ca` and `thal` are treated as categorical predictors in the machine-learning pipeline.

## 3.3 Original Target

The original target is `num`.

The observed values are:

| Original `num` | Interpretation | Binary target |
| -------------: | -------------- | ------------: |
|              0 | Absence        |             0 |
|              1 | Presence       |             1 |
|              2 | Presence       |             1 |
|              3 | Presence       |             1 |
|              4 | Presence       |             1 |

The original target distribution is:

|     `num` |   Count |
| --------: | ------: |
|         0 |     164 |
|         1 |      55 |
|         2 |      36 |
|         3 |      35 |
|         4 |      13 |
| **Total** | **303** |

After binary conversion:

| Binary target |   Count | Percentage |
| ------------: | ------: | ---------: |
|             0 |     164 |     54.13% |
|             1 |     139 |     45.87% |
|     **Total** | **303** |   **100%** |

## 3.4 Missing Values

The dataset contains six missing predictor cells:

* `ca`: 4 missing values
* `thal`: 2 missing values

No target values are missing.

No observations were deleted because of missing predictor values.

The six missing cells represent approximately 0.15% of all predictor cells.

No claim regarding MCAR, MAR, or MNAR is made because the available descriptive analysis cannot establish the underlying missingness mechanism.

---

# 4. Exploratory Data Analysis

## 4.1 Dataset Structure

The EDA confirmed:

* 303 observations
* 13 predictors
* 5 numerical predictors
* 8 categorical predictors
* 6 missing predictor cells
* No duplicate rows
* Binary target distribution of 164/139

## 4.2 Numerical Association Analysis

Numerical predictors were compared between the two target groups using two-sided Mann–Whitney U tests.

Rank-biserial correlation was used as the effect-size measure.

The project used the following convention:

`r_rb = (2U / (n₀n₁)) − 1`

Under this convention, a positive value indicates greater ranks in target 0, while a negative value indicates greater ranks in target 1.

The observed results were:

| Feature    |       U | Rank-biserial | FDR-adjusted p |
| ---------- | ------: | ------------: | -------------: |
| `age`      |  8274.5 |       -0.2740 |         0.0001 |
| `trestbps` |  9710.0 |       -0.1481 |         0.0307 |
| `chol`     |  9798.5 |       -0.1403 |         0.0383 |
| `thalach`  | 16989.5 |        0.4906 |         <0.001 |
| `oldpeak`  |  6037.0 |       -0.4703 |         <0.001 |

All five numerical variables remained statistically significant after Benjamini–Hochberg FDR adjustment at α = 0.05.

These results describe univariate distributional associations. They do not establish causal relationships, independent predictive importance, or clinical significance.

## 4.3 Categorical Association Analysis

Categorical predictors were evaluated using contingency-table analysis.

Pearson chi-square tests were used when the approximation was considered adequate.

For `restecg`, sparse expected frequencies led to a Monte-Carlo procedure.

| Feature   |      χ² | df | Unadjusted p | Bias-corrected Cramér's V | FDR-adjusted p |
| --------- | ------: | -: | -----------: | ------------------------: | -------------: |
| `sex`     | 22.0426 |  1 |       <0.001 |                    0.2639 |         <0.001 |
| `cp`      | 81.8158 |  3 |       <0.001 |                    0.5108 |         <0.001 |
| `fbs`     |  0.0771 |  1 |       0.7813 |                    0.0000 |         0.7813 |
| `restecg` | 10.0515 |  2 |      0.0044* |                    0.1632 |         0.0057 |
| `exang`   | 54.6864 |  1 |       <0.001 |                    0.4216 |         <0.001 |
| `slope`   | 45.7846 |  2 |       <0.001 |                    0.3807 |         <0.001 |
| `ca`      | 71.8431 |  3 |       <0.001 |                    0.4806 |         <0.001 |
| `thal`    | 83.2895 |  2 |       <0.001 |                    0.5205 |         <0.001 |

* The `restecg` p-value is Monte-Carlo-derived.

Seven of the eight categorical variables were statistically significant after FDR correction:

* `sex`
* `cp`
* `restecg`
* `exang`
* `slope`
* `ca`
* `thal`

`fbs` was not statistically significant.

## 4.4 Monte-Carlo Procedure for `restecg`

The observed Pearson chi-square statistic was:

$$
\chi^2 = 10.0515
$$

with:

$$
df=2
$$

Because of sparse expected frequencies, the p-value was obtained using 10,000 Monte-Carlo simulations with `random_state=42`.

The resulting p-value was:

$$
p=0.0044
$$

and this value was used as the unadjusted p-value in the Benjamini–Hochberg correction.

The resulting FDR-adjusted p-value was:

$$
p_{FDR}=0.0057
$$

Therefore, the statistic, degrees of freedom, raw p-value, and adjusted p-value should not be conflated.

## 4.5 Spearman Correlation

Spearman correlation was used for the five numerical predictors and binary target.

| Feature    | Spearman ρ with target |
| ---------- | ---------------------: |
| `age`      |                 0.2367 |
| `trestbps` |                 0.1282 |
| `chol`     |                 0.1211 |
| `thalach`  |                -0.4235 |
| `oldpeak`  |                 0.4134 |

The strongest absolute associations were:

* `thalach`: ρ = -0.4235
* `oldpeak`: ρ = 0.4134

These are descriptive rank-based associations and are not equivalent to machine-learning feature importance.

## 4.6 EDA and Feature Retention

Although several variables showed statistically significant associations with the target, no predictor was removed solely because of its univariate statistical result.

This avoids treating statistical association as equivalent to predictive usefulness.

All 13 predictors were therefore retained for the controlled machine-learning experiments.

---

# 5. Preprocessing Methodology

## 5.1 Train/Test Split

The dataset was divided using:

* 80% training
* 20% testing
* Stratification by binary target
* `random_state=42`

The resulting partitions were:

| Partition | Observations | Class 0 | Class 1 |
| --------- | -----------: | ------: | ------: |
| Training  |          242 |     131 |     111 |
| Test      |           61 |      33 |      28 |
| Total     |          303 |     164 |     139 |

The test set was reserved for final evaluation.

It was not used to fit preprocessing transformations, perform cross-validation, or select the baseline model.

## 5.2 Numerical Preprocessing

The numerical variables were processed using:

1. Median imputation.
2. StandardScaler.

The imputer and scaler were fitted only on the appropriate training data.

## 5.3 Categorical Preprocessing

The categorical variables were processed using:

1. Most-frequent imputation.
2. One-hot encoding.
3. `handle_unknown="ignore"`.

This prevents arbitrary numerical relationships from being imposed on categorical codes.

## 5.4 Transformed Feature Representation

The resulting representation contains:

* 5 numerical features
* 23 one-hot encoded categorical features

for a total of:

$$
28\text{ transformed features}
$$

## 5.5 Leakage Prevention

All preprocessing was implemented through scikit-learn `Pipeline` and `ColumnTransformer` objects.

During cross-validation, preprocessing was fitted independently within each training fold.

This ensures that:

* Test-set information is not used to estimate imputation values.
* Test-set information is not used to calculate scaling parameters.
* Category encoding is learned within the appropriate training data.

---

# 6. Experimental Design

## 6.1 Baseline Classifiers

Six baseline classifiers were evaluated:

1. Logistic Regression
2. K-Nearest Neighbors
3. Decision Tree
4. Gaussian Naive Bayes
5. Support Vector Machine
6. Random Forest

## 6.2 Model Configurations

| Model                | Configuration                                     |
| -------------------- | ------------------------------------------------- |
| Logistic Regression  | `max_iter=1000`, `random_state=42`                |
| KNN                  | `n_neighbors=5`                                   |
| Decision Tree        | `random_state=42`                                 |
| Gaussian Naive Bayes | Default configuration                             |
| SVM                  | RBF kernel, `probability=True`, `random_state=42` |
| Random Forest        | `n_estimators=200`, `random_state=42`             |

## 6.3 Cross-Validation

The training data were evaluated using:

* StratifiedKFold
* 5 folds
* `shuffle=True`
* `random_state=42`

The following metrics were calculated:

* Accuracy
* Precision
* Recall
* Specificity
* F1-score
* ROC-AUC

The predefined model-selection criterion was **mean cross-validation ROC-AUC**.

---

# 7. Baseline Results

The six classifiers produced the following mean five-fold cross-validation results:

| Model                | Accuracy | Precision | Recall | Specificity |     F1 |    ROC-AUC |
| -------------------- | -------: | --------: | -----: | ----------: | -----: | ---------: |
| Logistic Regression  |   0.8471 |    0.8768 | 0.7834 |      0.9003 | 0.8245 | **0.9025** |
| Random Forest        |   0.8139 |    0.8129 | 0.7739 |      0.8470 | 0.7900 |     0.8927 |
| SVM                  |   0.8139 |    0.8303 | 0.7648 |      0.8541 | 0.7893 |     0.8884 |
| Gaussian Naive Bayes |   0.7438 |    0.7049 | 0.8458 |      0.6564 | 0.7495 |     0.8773 |
| KNN                  |   0.8222 |    0.8232 | 0.7826 |      0.8544 | 0.7989 |     0.8712 |
| Decision Tree        |   0.6980 |    0.6782 | 0.6842 |      0.7091 | 0.6768 |     0.6967 |

## 7.1 Baseline Selection

Logistic Regression achieved the highest mean five-fold cross-validation ROC-AUC:

$$
ROC\text{-}AUC_{CV}=0.9025
$$

It was therefore selected as the baseline model according to the predefined selection criterion.

This does **not** establish Logistic Regression as universally superior to the other algorithms.

---

# 8. Advanced XGBoost Model

XGBoost was evaluated as the advanced ensemble model.

The fixed configuration was:

```text
n_estimators = 200
max_depth = 3
learning_rate = 0.05
subsample = 0.9
colsample_bytree = 0.9
objective = "binary:logistic"
eval_metric = "logloss"
random_state = 42
n_jobs = 1
```

The same preprocessing architecture and stratified five-fold cross-validation framework were used.

---

# 9. XGBoost Cross-Validation Results

| Metric      |   Mean | Population SD (`ddof=0`) |
| ----------- | -----: | -----------------------: |
| Accuracy    | 0.8016 |                   0.0249 |
| Precision   | 0.7848 |                   0.0317 |
| Recall      | 0.7834 |                   0.0613 |
| Specificity | 0.8165 |                   0.0387 |
| F1          | 0.7825 |                   0.0345 |
| ROC-AUC     | 0.8664 |                   0.0330 |

## 9.1 Fold-Level ROC-AUC

The five fold-level ROC-AUC values were:

| Fold | ROC-AUC |
| ---: | ------: |
|    1 |  0.9080 |
|    2 |  0.8367 |
|    3 |  0.8427 |
|    4 |  0.9056 |
|    5 |  0.8392 |

Their mean is:

$$
0.8664
$$

The population standard deviation used by the primary `cross_validate` summary is:

$$
SD_{ddof=0}=0.0330
$$

The later diagnostic calculates the sample standard deviation:

$$
SD_{ddof=1}=0.0369
$$

Both values are derived from the same five fold scores.

Therefore, the difference is a statistical convention rather than a difference in experiments or model performance.

---

# 10. Held-Out Test Results

The final evaluation was conducted on the untouched 61-observation test set.

| Metric               | Logistic Regression |    XGBoost |
| -------------------- | ------------------: | ---------: |
| Accuracy             |              0.8852 | **0.9016** |
| Precision            |              0.8387 | **0.8438** |
| Recall / Sensitivity |              0.9286 | **0.9643** |
| Specificity          |              0.8485 |     0.8485 |
| F1                   |              0.8814 | **0.9000** |
| ROC-AUC              |          **0.9665** |     0.9545 |

## 10.1 Logistic Regression Confusion Matrix

```text
[[28, 5],
 [ 2, 26]]
```

Therefore:

* True negatives = 28
* False positives = 5
* False negatives = 2
* True positives = 26

## 10.2 XGBoost Confusion Matrix

```text
[[28, 5],
 [ 1, 27]]
```

Therefore:

* True negatives = 28
* False positives = 5
* False negatives = 1
* True positives = 27

XGBoost therefore produced one fewer false negative on this particular held-out test set.

---

# 11. Results Discussion

## 11.1 Cross-Validation Comparison

Logistic Regression achieved the strongest mean cross-validation ROC-AUC among the six baseline classifiers at 0.9025.

The ranking was:

1. Logistic Regression — 0.9025
2. Random Forest — 0.8927
3. Support Vector Machine — 0.8884
4. Gaussian Naive Bayes — 0.8773
5. KNN — 0.8712
6. Decision Tree — 0.6967

This ranking is specific to the stated preprocessing, training data, model configurations, five-fold validation strategy, and random state.

## 11.2 Logistic Regression Versus XGBoost

XGBoost had a lower mean cross-validation ROC-AUC than Logistic Regression:

* Logistic Regression: 0.9025
* XGBoost: 0.8664

However, on the single held-out test set, XGBoost achieved higher:

* Accuracy
* Precision
* Recall
* F1-score

Both models had the same specificity of 0.8485.

Logistic Regression achieved the higher held-out ROC-AUC:

* Logistic Regression: 0.9665
* XGBoost: 0.9545

This demonstrates why the study reports multiple metrics.

## 11.3 Threshold-Dependent and Threshold-Independent Measures

Accuracy, precision, recall, specificity, and F1-score depend on the selected classification threshold.

ROC-AUC instead summarizes ranking/discrimination performance across thresholds.

Consequently, a model can achieve stronger threshold-dependent metrics at one operating threshold while another model achieves a higher ROC-AUC.

The observed results illustrate this distinction but should not be interpreted as evidence that either model is universally superior.

## 11.4 Interpretation of the Held-Out Difference

XGBoost produced one fewer false negative than Logistic Regression on the 61-observation test set.

However, the test set is small, and the experiment uses a single train/test partition.

Therefore, the observed difference should be interpreted as an experimental observation rather than a statistically established universal advantage.

---

# 12. Discussion of EDA and Modelling

The EDA identified statistically significant univariate associations for all five numerical predictors and seven of the eight categorical predictors after FDR correction.

However, these findings were not directly converted into feature elimination.

This decision was deliberate because:

* Statistical significance does not imply predictive importance.
* Association does not establish causality.
* Univariate tests do not account for interactions between predictors.
* A variable with weak univariate association can still contribute to multivariate prediction.
* Removing variables based solely on EDA could introduce an unjustified modelling assumption.

The study therefore retained all 13 predictors and allowed the machine-learning experiments to evaluate their usefulness within the complete modelling pipeline.

---

# 13. Comparison with Literature

The literature comparison should distinguish between studies that are structurally comparable and studies that merely provide contextual evidence.

## 13.1 Direct or Relatively Direct Cleveland Comparisons

Particularly relevant studies in the supplied review include:

* Uyar and Ilhan (2017)
* Haq et al. (2018)
* Amin et al. (2019)
* Tougui et al. (2020)
* Kavitha et al. (2021)
* Teekaraman et al. (2022)

These studies involve the Cleveland dataset or a structurally similar 303-instance/13-feature setting according to the supplied literature review.

## 13.2 Contextual Studies

Other reviewed studies use:

* Different heart-disease datasets.
* Different sample sizes.
* Different feature sets.
* Different preprocessing procedures.
* Different train/test partitions.
* Different evaluation metrics.
* Hybrid or deep-learning architectures.

Their reported performance values should therefore not be directly ranked against the present experiment.

## 13.3 Literature Comparison Principle

A reported accuracy such as 97% from another paper cannot be interpreted as demonstrating that its model is superior to the 90.16% XGBoost test accuracy obtained here unless the datasets, target definitions, preprocessing, validation procedure, and evaluation protocol are sufficiently comparable.

The literature therefore provides context and methodological comparison rather than a universal leaderboard.

---

# 14. Research Gap and Study Positioning

The study is positioned as a **controlled and reproducible comparison**, rather than as a claim of a novel predictive algorithm.

Its methodological emphasis is:

* Common preprocessing.
* Fixed experimental configuration.
* Stratified cross-validation.
* Held-out test evaluation.
* Multiple evaluation metrics.
* Explicit leakage prevention.
* Reproducible random states.
* Separation between exploratory statistics and predictive performance.

The study does not claim state-of-the-art performance.

---

# 15. Limitations

The following limitations should be stated explicitly.

## 15.1 Dataset Size

Only 303 observations are available.

The relatively small dataset limits the precision with which model performance can be estimated.

## 15.2 Dataset Scope

The experiment uses only the Cleveland subset.

Therefore, the findings cannot establish generalization to other populations or institutions.

## 15.3 Single Held-Out Split

The final evaluation uses one 80:20 stratified split.

Although five-fold cross-validation is used on the training data, the final test estimate is based on one held-out partition.

## 15.4 Small Test Set

Only 61 observations were held out for final evaluation.

Small changes in individual predictions can therefore noticeably affect the reported metrics.

## 15.5 Fixed Model Configurations

The study does not conduct exhaustive hyperparameter optimization.

The results therefore represent the specified configurations rather than fully optimized versions of the algorithms.

## 15.6 No External Validation

No independent external dataset was used.

Therefore, external generalization has not been demonstrated.

## 15.7 No Clinical Validation

The study does not establish:

* Clinical validity.
* Clinical utility.
* Clinical safety.
* Suitability for deployment.
* Replacement of medical diagnosis.

## 15.8 Statistical Interpretation

EDA associations are not evidence of:

* Causality.
* Clinical significance.
* Independent predictive importance.

## 15.9 Model Comparison

Differences observed on the single held-out test set should not be interpreted as universal superiority.

---

# 16. Future Work

Future studies could extend the experiment using:

1. Repeated stratified cross-validation.
2. Nested cross-validation.
3. Controlled hyperparameter tuning.
4. Feature-selection experiments.
5. Calibration analysis.
6. Explainability and model-interpretation techniques.
7. Independent external validation.
8. Larger datasets.
9. Additional Cleveland-comparable benchmark studies.
10. Statistical comparison of model performance across repeated experimental partitions.

Future work should preserve the leakage-prevention principles used in the present experiment.

---

# 17. Conclusion

This study evaluated six baseline machine-learning classifiers and an XGBoost model for binary heart-disease prediction using the UCI Cleveland dataset.

The experiment used 303 observations and 13 predictor variables, with the original five-level `num` target converted into a binary target. Missing values were handled through leakage-safe pipeline preprocessing, numerical variables were median-imputed and standardized, and categorical variables were most-frequently imputed and one-hot encoded.

Under the predefined mean five-fold cross-validation ROC-AUC criterion, Logistic Regression was the strongest baseline model, achieving a mean ROC-AUC of 0.9025.

XGBoost achieved a lower mean cross-validation ROC-AUC of 0.8664. Nevertheless, on the single held-out test set, XGBoost produced higher accuracy, precision, recall, and F1-score than Logistic Regression, while both models achieved the same specificity. Logistic Regression achieved the higher held-out ROC-AUC of 0.9665 compared with 0.9545 for XGBoost.

These differences demonstrate that model performance depends on the evaluation metric and experimental setting. In particular, held-out threshold-dependent metrics and cross-validation ROC-AUC should not be treated as interchangeable evidence.

The findings are specific to the Cleveland dataset, the fixed preprocessing pipeline, model configurations, random state, five-fold cross-validation procedure, and single 80:20 train/test split. They therefore should not be interpreted as evidence of universal model superiority or clinical validity.

Overall, the experiment demonstrates the importance of controlled preprocessing, leakage prevention, stratified validation, multiple evaluation metrics, and cautious interpretation when comparing machine-learning models for heart-disease prediction.

---

# References

The supplied literature files establish the following 20 study records:

[1] S. Palaniappan and R. Awang, “Intelligent heart disease prediction system using data mining techniques,” in 2008 IEEE/ACS International Conference on Computer Systems and Applications, pp. 108–115, 2008, doi: 10.1109/AICCSA.2008.4493524.

[2] M. Sultana, A. Haider, and M. S. Uddin, “Analysis of data mining techniques for heart disease prediction,” in 2016 3rd International Conference on Electrical Engineering and Information Communication Technology (ICEEICT), pp. 1–5, 2016, doi: 10.1109/CEEICT.2016.7873142.

[3] M. Abdar, S. R. Niakan Kalhori, T. Sutikno, I. M. I. Subroto, and G. Arji, “Comparing performance of data mining algorithms in prediction heart diseases,” International Journal of Electrical and Computer Engineering, vol. 5, no. 6, pp. 1569–1576, 2015, doi: 10.11591/ijece.v5i6.pp1569-1576.

[4] Z. Arabasadi, R. Alizadehsani, M. Roshanzamir, H. Moosaei, and A. A. Yarifard, “Computer aided decision making for heart disease detection using hybrid neural network-genetic algorithm,” Computer Methods and Programs in Biomedicine, vol. 141, pp. 19–26, 2017, doi: 10.1016/j.cmpb.2017.01.004.

[5] K. Uyar and A. Ilhan, “Diagnosis of heart disease using genetic algorithm based trained recurrent fuzzy neural networks,” Procedia Computer Science, vol. 120, pp. 588–593, 2017, doi: 10.1016/j.procs.2017.11.283.

[6] K. Pahwa and R. Kumar, “Prediction of heart disease using hybrid technique for selecting features,” in 2017 4th IEEE Uttar Pradesh Section International Conference on Electrical, Computer and Electronics (UPCON), pp. 500–504, 2017, doi: 10.1109/UPCON.2017.8251100.

[7] P. Sharma and K. Saxena, “Application of fuzzy logic and genetic algorithm in heart disease risk level prediction,” International Journal of System Assurance Engineering and Management, vol. 8, no. 2, pp. 1109–1125, 2017, doi: 10.1007/s13198-017-0578-8.

[8] A. U. Haq, J. P. Li, M. H. Memon, S. Nazir, and R. Sun, “A hybrid intelligent system framework for the prediction of heart disease using machine learning algorithms,” Mobile Information Systems, vol. 2018, Art. no. 3860146, 21 pages, 2018, doi: 10.1155/2018/3860146.

[9] A. K. Dwivedi, “Performance evaluation of different machine learning techniques for prediction of heart disease,” Neural Computing and Applications, vol. 29, no. 10, pp. 685–693, 2018, doi: 10.1007/s00521-016-2604-1.

[10] M. S. Amin, Y. K. Chiam, and K. D. Varathan, “Identification of significant features and data mining techniques in predicting heart disease,” Telematics and Informatics, vol. 36, pp. 82–93, 2019, doi: 10.1016/j.tele.2018.11.007.

[11] S. Mohan, C. Thirumalai, and G. Srivastava, “Effective heart disease prediction using hybrid machine learning techniques,” IEEE Access, vol. 7, pp. 81542–81554, 2019, doi: 10.1109/ACCESS.2019.2923707.

[12] C. B. C. Latha and S. C. Jeeva, “Improving the accuracy of prediction of heart disease risk based on ensemble classification techniques,” Informatics in Medicine Unlocked, vol. 16, Art. no. 100203, 2019, doi: 10.1016/j.imu.2019.100203. 

[13] L. Ali, A. Rahman, A. Khan, M. Zhou, A. Javeed, and J. A. Khan, “An automated diagnostic system for heart disease prediction based on χ² statistical model and optimally configured deep neural network,” IEEE Access, vol. 7, pp. 34938–34945, 2019, doi: 10.1109/ACCESS.2019.2904800. 

[14] Z. Al-Makhadmeh and A. Tolba, “Utilizing IoT wearable medical device for heart disease prediction using higher order Boltzmann model: A classification approach,” Measurement, vol. 147, Art. no. 106815, 2019, doi: 10.1016/j.measurement.2019.07.043.

[15] I. Tougui, A. Jilbab, and J. El Mhamdi, “Heart disease classification using data mining tools and machine learning techniques,” Health and Technology, vol. 10, no. 5, pp. 1137–1144, 2020, doi: 10.1007/s12553-020-00438-1.

[16] M. Sabir, T. F. Khan, and M. Azam, “A comparative study of traditional and hybrid models for text classification,” *Journal of Computers and Intelligent Systems*, vol. 3, no. 1, pp. 81–91, 2025.

[17] M. Kavitha, G. Gnaneswar, R. Dinesh, Y. R. Sai, and R. S. Suraj, “Heart disease prediction using hybrid machine learning model,” in 2021 6th International Conference on Inventive Computation Technologies (ICICT), Coimbatore, India, pp. 1329–1333, 2021, doi: 10.1109/ICICT50816.2021.9358597.

[18] R. Bharti, A. Khamparia, M. Shabaz, G. Dhiman, S. Pande, and P. Singh, “Prediction of heart disease using a combination of machine learning and deep learning,” Computational Intelligence and Neuroscience, vol. 2021, Art. no. 8387680, 11 pages, 2021, doi: 10.1155/2021/8387680.

[19] P. Rani, R. Kumar, N. M. O. S. Ahmed, and A. Jain, “A decision support system for heart disease prediction based upon machine learning,” Journal of Reliable Intelligent Environments, vol. 7, no. 3, pp. 263–275, 2021, doi: 10.1007/s40860-021-00133-6.

[20] K. Karthick, S. K. Aruna, R. Samikannu, R. Kuppusamy, Y. Teekaraman, and A. R. Thelkar, “[Retracted] Implementation of a heart disease risk prediction model using machine learning,” *Computational and Mathematical Methods in Medicine*, vol. 2022, Art. no. 6517716, 14 pages, 2022, doi: 10.1155/2022/6517716.

The dataset citation supplied by the project documentation is:

A. Janosi, W. Steinbrunn, M. Pfisterer, and R. Detrano, “Heart Disease,” *UCI Machine Learning Repository*, 1989. doi: [10.24432/C52P4X](https://doi.org/10.24432/C52P4X).

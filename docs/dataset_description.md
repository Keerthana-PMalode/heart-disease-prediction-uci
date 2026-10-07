# Dataset Description

## 1. Dataset Source

The project uses the **Cleveland subset of the UCI Heart Disease dataset** from the UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/45/heart+disease

The dataset contains **303 observations and 13 predictor variables**. The original diagnosis field is `num`.

## 2. Predictor Variables

| Variable | Type used in preprocessing | Description |
|---|---|---|
| `age` | Numerical | Age in years |
| `sex` | Categorical | Sex indicator |
| `cp` | Categorical | Chest-pain type |
| `trestbps` | Numerical | Resting blood pressure |
| `chol` | Numerical | Serum cholesterol |
| `fbs` | Categorical | Fasting blood sugar indicator |
| `restecg` | Categorical | Resting electrocardiographic result |
| `thalach` | Numerical | Maximum heart rate achieved |
| `exang` | Categorical | Exercise-induced angina indicator |
| `oldpeak` | Numerical | ST depression induced by exercise |
| `slope` | Categorical | Slope of the peak exercise ST segment |
| `ca` | Categorical | Number of major vessels identified by fluoroscopy |
| `thal` | Categorical | Thalassemia-related coded attribute |

The project treats five variables as numerical (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) and eight as categorical.

## 3. Original Target

The original `num` variable has five coded values:

- `0` — absence of heart disease
- `1`–`4` — presence of heart disease at different diagnostic levels

For binary classification, the project uses:

```text
Original num     Binary target
0                0
1                1
2                1
3                1
4                1
```

The binary conversion is implemented centrally in `src/preprocessing.py`.

## 4. Missing Values

Missing values are represented by `?` in the source file and are loaded as `NaN`. The current Cleveland file contains six missing predictor cells: four in `ca` and two in `thal`. No target values are missing.

The project does not delete observations solely because of these missing values. Instead, missing-value handling is learned within each training fold/model pipeline.

## 5. Data Preparation Decisions

1. The original predictor columns are retained; no predictor is removed solely because of a univariate EDA result.
2. The original target is converted to a binary target for modeling.
3. Numerical predictors receive median imputation followed by standardization.
4. Categorical predictors receive most-frequent imputation followed by one-hot encoding with `handle_unknown="ignore"`.
5. Preprocessing is fitted only on training data or inside cross-validation folds through `Pipeline` and `ColumnTransformer`.
6. No automatic outlier deletion is performed.
7. The modeling split is 80:20 and stratified with `random_state=42`.

## 6. Dataset Scope

The experiments are specific to the Cleveland subset and should not be interpreted as external clinical validation or as evidence of generalization to other populations or datasets.
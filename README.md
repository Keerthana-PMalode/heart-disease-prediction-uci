# Heart Disease Prediction Using Machine Learning

## Overview

This project investigates machine-learning approaches for predicting heart disease using the **Cleveland subset of the UCI Heart Disease dataset**. The project emphasizes reproducible experimentation, leakage-safe preprocessing, controlled model comparison, and multiple evaluation metrics rather than accuracy alone.

The project uses 303 observations and 13 predictor variables. The original `num` target is converted from the UCI 0–4 representation to a binary classification target. 

## Dataset

Source: UCI Machine Learning Repository — Heart Disease Dataset

https://archive.ics.uci.edu/dataset/45/heart+disease

The project uses `data/raw/processed.cleveland.data` as the reproducible modeling input. Missing `?` values are loaded as `NaN` and handled inside training pipelines.

## Project Structure

```text
heart-disease-prediction/
├── data/
│   ├── processed/
│   ├── raw/
│   │   ├── cleveland.data
│   │   └── processed.cleveland.data
│   ├── heart_disease.csv
│   └── README.md
├── docs/
│   ├── dataset_description.md
│   ├── methodology.md
│   └── results_and_discussion.md
├── literature/
│   ├── literature_comparison.csv
│   └── literature_review.md
├── notebooks/
│   ├── 01_data_validation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_baseline_models.ipynb
│   ├── 05_advanced_model.ipynb
│   └── 06_model_comparison.ipynb
├── results/
│   ├── confusion_matrices/
│   ├── figures/
│   └── model_results.csv
├── src/
│   ├── __init__.py
│   ├── evaluation.py
│   ├── models.py
│   ├── preprocessing.py
│   └── visualization.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

The notebook progression follows validation → EDA → preprocessing → baseline models → advanced model → model comparison, consistent with the existing project methodology. 

## Installation

Create and activate a Python environment, then install the project dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The advanced-model notebook requires `xgboost`.

## Running the notebooks

Run the notebooks in order from the repository root or through Jupyter:

```bash
jupyter notebook
```

Then execute:

1. `01_data_validation.ipynb`
2. `02_eda.ipynb`
3. `03_preprocessing.ipynb`
4. `04_baseline_models.ipynb`
5. `05_advanced_model.ipynb`
6. `06_model_comparison.ipynb`

Each notebook locates the project root and imports reusable code from `src/`. The notebooks do not depend on execution state from another notebook.

## Reusable Source Package

The `src/` package separates reusable implementation from notebook-specific analysis:

- **`preprocessing.py`** — dataset loading, column validation, target preparation, feature groups, leakage-safe preprocessing, and model-pipeline construction.
- **`models.py`** — baseline/advanced model definitions, established hyperparameters, pipeline builders, and cross-validation scoring configuration.
- **`evaluation.py`** — prediction generation, confusion matrices, classification reports, Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, and model-comparison tables.
- **`visualization.py`** — reusable EDA, confusion-matrix, ROC, cross-validation, feature-importance, and model-comparison plots.

## Preprocessing

The 13 predictors are separated into five numerical and eight categorical variables. Numerical variables use median imputation and standardization. Categorical variables use most-frequent imputation and one-hot encoding with unknown-category handling.

All transformations are fitted within model pipelines. During cross-validation, each fold fits its preprocessing only on its corresponding training portion. The held-out test set is not used to fit preprocessing or select the baseline model.

## Models Evaluated

### Baselines

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Gaussian Naive Bayes
- Support Vector Machine
- Random Forest

### Advanced model

- XGBoost

The model configurations are centralized in `src/models.py`.

## Evaluation

The project reports:

- Accuracy
- Precision
- Recall/Sensitivity
- Specificity
- F1-score
- ROC-AUC
- Confusion matrix

Specificity is defined as `TN / (TN + FP)`. ROC-AUC uses positive-class probability scores when available. The evaluation implementation is centralized in `src/evaluation.py`.

## Established Results

The six baseline models are compared using stratified five-fold cross-validation on the training set. Logistic Regression has the highest mean ROC-AUC of **0.9025** and is the selected baseline under the project's predefined criterion.

The final held-out test results for Logistic Regression and XGBoost are:

| Metric | Logistic Regression | XGBoost |
|---|---:|---:|
| Accuracy | 0.8852 | 0.9016 |
| Precision | 0.8387 | 0.8438 |
| Recall | 0.9286 | 0.9643 |
| Specificity | 0.8485 | 0.8485 |
| F1 | 0.8814 | 0.9000 |
| ROC-AUC | **0.9665** | 0.9545 |

These results describe this specific Cleveland-dataset experiment and should not be interpreted as external clinical validation.

## Reproducibility

The established project random state is **42**. It is used for the stratified 80:20 train/test split, shuffled five-fold cross-validation, and stochastic model configurations where applicable. The project keeps preprocessing inside pipelines to prevent leakage.

## Documentation

- `docs/dataset_description.md` — dataset and data-preparation details
- `docs/methodology.md` — complete machine-learning methodology
- `docs/results_and_discussion.md` — experimental results and interpretation
- `data/README.md` — data-artifact documentation

## Scope and Limitations

This is an academic machine-learning experiment using a small benchmark dataset. The project does not claim clinical deployment or universal model superiority. Further validation on independent data and additional methodological studies would be required before drawing broader conclusions.
